"""In-process Hugging Face causal LM: real weights, credit-gated generate.

Each Interest is ``max_new_tokens=W``. Zero credit ⇒ no forward.
Default demo model is SmolLM2-135M-Instruct (CPU). Optional extra: ``pip install -e ".[hf]"``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import Thread
from typing import Any

from dart.cip.merkle import bytes_per_extent, extent_digest, root_from_extents
from dart.core.errors import EngineError
from dart.core.types import EngineState, KVExtent, ModelConfig, Prompt
from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.stats import KernelStats

DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

TokenizeFn = Callable[[Prompt], list[int]]
GenerateFn = Callable[[list[int], int], tuple[list[int], str, bool]]


def resolve_hf_model_id(model: str | None) -> str:
    if not model or model in {"dart-synth-8b", "synthetic", "local"}:
        return DEFAULT_HF_MODEL
    return model


class HuggingFaceEngine:
    """Greedy causal LM behind the same CIP decode gate as vLLM/llama.cpp."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        config: ModelConfig | None = None,
        device: str = "cpu",
        tokenize_fn: TokenizeFn | None = None,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        self.model_id = resolve_hf_model_id(model_id)
        self.device = device
        self.config = config or ModelConfig(
            model_id=self.model_id,
            tokenizer_hash="hf",
            dtype="f32",
        )
        self.stats = KernelStats()
        self.kernel_launches = 0
        self.decode_tokens = 0
        self.supports_rollback = False
        self._tokenize_fn = tokenize_fn
        self._generate_fn = generate_fn
        self._tok: Any = None
        self._model: Any = None

    def _opaque_extents(self, state: EngineState) -> list[KVExtent]:
        cfg = self.config
        nbytes = bytes_per_extent(cfg.n_kv_heads, cfg.head_dim, cfg.block_size)
        n_blocks = max(1, (max(state.pos, 1) + cfg.block_size - 1) // cfg.block_size)
        extents: list[KVExtent] = []
        for layer in range(min(cfg.n_layers, 8)):
            for b in range(n_blocks):
                digest = extent_digest(layer, b, state.pos, state.all_ids[-32:])
                extents.append(KVExtent(layer=layer, block_id=b, digest=digest, nbytes=nbytes))
        return extents

    def _ensure_loaded(self) -> None:
        if self._tokenize_fn is not None and self._generate_fn is not None:
            return
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise EngineError(
                "HuggingFace engine needs transformers+torch. Install with: pip install -e '.[hf]'"
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        cfg = self._model.config
        n_heads = int(getattr(cfg, "num_attention_heads", 8) or 8)
        hidden = int(getattr(cfg, "hidden_size", n_heads * 64) or n_heads * 64)
        self.config = self.config.model_copy(
            update={
                "n_layers": int(getattr(cfg, "num_hidden_layers", 8) or 8),
                "n_kv_heads": int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads),
                "head_dim": max(1, hidden // max(1, n_heads)),
                "vocab_size": int(getattr(cfg, "vocab_size", 32000) or 32000),
            }
        )
        self._torch = torch

    def _ids_from_prompt(self, prompt: Prompt) -> list[int]:
        self._ensure_loaded()
        if self._tokenize_fn is not None:
            return self._tokenize_fn(prompt)
        messages: list[dict[str, str]]
        if prompt.messages:
            messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        else:
            messages = [{"role": "user", "content": prompt.as_text() or "Hello"}]
        tok = self._tok
        if getattr(tok, "chat_template", None):
            enc = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            ids = enc["input_ids"] if hasattr(enc, "keys") and "input_ids" in enc else enc
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(x) for x in ids]
        text = prompt.as_text() or "Hello"
        return [int(x) for x in tok.encode(text, add_special_tokens=True)]

    def _generate_sync(self, ids: list[int], n: int) -> tuple[list[int], str, bool]:
        self._ensure_loaded()
        if self._generate_fn is not None:
            return self._generate_fn(ids, n)
        torch = self._torch
        tok = self._tok
        model = self._model
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=attn,
                max_new_tokens=max(1, n),
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
        new = out[0, input_ids.shape[1] :].tolist()
        text = tok.decode(new, skip_special_tokens=True)
        stopped = bool(new) and int(new[-1]) == int(tok.eos_token_id)
        return [int(x) for x in new], text, stopped

    def _generate_stream_sync(
        self,
        ids: list[int],
        n: int,
        emit: Callable[[str], None],
    ) -> tuple[list[int], str, bool]:
        self._ensure_loaded()
        if self._generate_fn is not None:
            new_ids, text, stopped = self._generate_fn(ids, n)
            if text:
                emit(text)
            return new_ids, text, stopped
        from transformers import TextIteratorStreamer

        torch = self._torch
        tok = self._tok
        model = self._model
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_ids)
        holder: dict[str, Any] = {}

        def run() -> None:
            try:
                with torch.inference_mode():
                    holder["out"] = model.generate(
                        input_ids,
                        attention_mask=attn,
                        max_new_tokens=max(1, n),
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        use_cache=True,
                        streamer=streamer,
                    )
            except Exception as exc:
                holder["err"] = exc
                streamer.end()

        worker = Thread(target=run, daemon=True)
        worker.start()
        for piece in streamer:
            if piece:
                emit(piece)
        worker.join()
        if "err" in holder:
            raise holder["err"]
        out = holder["out"]
        new = out[0, input_ids.shape[1] :].tolist()
        text = tok.decode(new, skip_special_tokens=True)
        stopped = bool(new) and int(new[-1]) == int(tok.eos_token_id)
        return [int(x) for x in new], text, stopped

    def _apply_decode(
        self,
        state: EngineState,
        ids: list[int],
        new_ids: list[int],
        text: str,
        stopped: bool,
        *,
        grammar_span: str | None,
    ) -> DecodeResult:
        predicted = max(1, len(new_ids) or len(text.split()) or 1)
        self.stats.record_generate(predicted=predicted, prompt_tokens=len(ids), prefix_hit=True)
        self.kernel_launches = self.stats.kernel_launches
        self.decode_tokens = self.stats.tokens_predicted
        state.assistant_text += text
        if not new_ids:
            new_ids = list(range(state.pos, state.pos + predicted))
        state.output_ids.extend(new_ids)
        state.pos += len(new_ids)
        state.kv_extents = self._opaque_extents(state)
        state.kv_root = root_from_extents(state.kv_extents)
        state.stopped = stopped or (not text and not new_ids)
        return DecodeResult(
            token_ids=new_ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=state.stopped,
            grammar_span=bool(grammar_span),
        )

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        ids = await asyncio.to_thread(self._ids_from_prompt, prompt)
        text = prompt.as_text()
        state.prompt_ids = ids
        state.output_ids = []
        state.pos = len(ids)
        state.prefix_text = text
        state.assistant_text = ""
        state.kv_extents = self._opaque_extents(state)
        state.kv_root = root_from_extents(state.kv_extents)
        return PrefillResult(state=state, text=text, extents=state.kv_extents)

    async def decode(
        self,
        state: EngineState,
        n: int,
        *,
        grammar_span: str | None = None,
    ) -> DecodeResult:
        if n <= 0 and not grammar_span:
            return DecodeResult(
                token_ids=[],
                text="",
                extents=state.kv_extents,
                kv_root=state.kv_root,
                kernel_launched=False,
            )
        window = max(1, n if not grammar_span else max(n, 32))
        ids = list(state.all_ids) or [1]
        try:
            new_ids, text, stopped = await asyncio.to_thread(self._generate_sync, ids, window)
        except Exception as exc:
            raise EngineError(f"HuggingFace decode failed: {exc}") from exc
        return self._apply_decode(state, ids, new_ids, text, stopped, grammar_span=grammar_span)

    async def decode_stream(
        self,
        state: EngineState,
        n: int,
        *,
        grammar_span: str | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> DecodeResult:
        if n <= 0 and not grammar_span:
            return DecodeResult(
                token_ids=[],
                text="",
                extents=state.kv_extents,
                kv_root=state.kv_root,
                kernel_launched=False,
            )
        window = max(1, n if not grammar_span else max(n, 32))
        ids = list(state.all_ids) or [1]
        emit = on_text or (lambda _s: None)
        try:
            new_ids, text, stopped = await asyncio.to_thread(
                self._generate_stream_sync, ids, window, emit
            )
        except Exception as exc:
            raise EngineError(f"HuggingFace decode failed: {exc}") from exc
        return self._apply_decode(state, ids, new_ids, text, stopped, grammar_span=grammar_span)
