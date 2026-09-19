"""In-process Hugging Face causal LM: real weights, credit-gated generate.

Prefill stores ``past_key_values``. Each later Interest decodes only the
new window — pages do not re-encode the prefix. Zero credit ⇒ no forward.
Default demo model is SmolLM2-135M-Instruct (CPU). Optional: ``pip install -e ".[hf]"``.
"""

from __future__ import annotations

import asyncio
import uuid
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
        self._pkv: dict[str, Any] = {}
        self._cache_len: dict[str, int] = {}
        self.tokens_encoded = 0
        self.prefill_forwards = 0
        self.last_encode_len = 0
        self._last_prefix_hit = False

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

    def _rid(self, state: EngineState) -> str:
        if not state.engine_request_id:
            state.engine_request_id = uuid.uuid4().hex
        return state.engine_request_id

    def _pkv_len(self, cache: Any) -> int:
        if cache is None:
            return 0
        get = getattr(cache, "get_seq_length", None)
        if callable(get):
            try:
                return int(get())
            except Exception:
                pass
        try:
            return int(cache[0][0].shape[-2])
        except Exception:
            return 0

    def _prefill_cache_sync(self, ids: list[int]) -> Any:
        self._ensure_loaded()
        torch = self._torch
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(t)
        with torch.inference_mode():
            out = self._model(input_ids=t, attention_mask=attn, use_cache=True)
        return out.past_key_values

    def _record_encode(self, n: int, *, prefix_hit: bool) -> None:
        self.last_encode_len = n
        self.tokens_encoded += n
        self._last_prefix_hit = prefix_hit

    def _generate_sync(
        self,
        ids: list[int],
        n: int,
        *,
        rid: str = "",
    ) -> tuple[list[int], str, bool]:
        self._ensure_loaded()
        cache = self._pkv.get(rid) if rid else None
        if self._generate_fn is not None:
            if cache is not None:
                seed = [ids[-1]] if ids else [1]
                self._record_encode(len(seed), prefix_hit=True)
                new_ids, text, stopped = self._generate_fn(seed, n)
                self._cache_len[rid] = self._cache_len.get(rid, 0) + len(new_ids)
                return new_ids, text, stopped
            self._record_encode(len(ids), prefix_hit=False)
            return self._generate_fn(ids, n)
        new_ids, text, stopped, new_cache = self._generate_from_cache_sync(
            ids, n, cache=cache, emit=None
        )
        if rid and new_cache is not None:
            self._pkv[rid] = new_cache
            self._cache_len[rid] = self._pkv_len(new_cache) or (
                self._cache_len.get(rid, 0) + len(new_ids)
            )
        return new_ids, text, stopped

    def _generate_from_cache_sync(
        self,
        ids: list[int],
        n: int,
        *,
        cache: Any,
        emit: Callable[[str], None] | None,
    ) -> tuple[list[int], str, bool, Any]:
        torch = self._torch
        tok = self._tok
        model = self._model
        window = max(1, n)
        if cache is None:
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            attn = torch.ones_like(input_ids)
            prefix = input_ids.shape[1]
            self._record_encode(int(prefix), prefix_hit=False)
            past = None
        else:
            last = ids[-1] if ids else 1
            past_len = self._pkv_len(cache) or self._cache_len.get("", 0)
            input_ids = torch.tensor([[last]], dtype=torch.long, device=self.device)
            attn = torch.ones((1, max(past_len, 1) + 1), dtype=torch.long, device=self.device)
            prefix = 1
            self._record_encode(1, prefix_hit=True)
            past = cache
        holder: dict[str, Any] = {}
        streamer = None
        if emit is not None:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        def run() -> None:
            try:
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "max_new_tokens": window,
                    "do_sample": False,
                    "pad_token_id": tok.pad_token_id,
                    "eos_token_id": tok.eos_token_id,
                    "use_cache": True,
                    "return_dict_in_generate": True,
                }
                if past is not None:
                    kwargs["past_key_values"] = past
                if streamer is not None:
                    kwargs["streamer"] = streamer
                with torch.inference_mode():
                    holder["out"] = model.generate(**kwargs)
            except Exception as exc:
                holder["err"] = exc
                if streamer is not None:
                    streamer.end()

        if streamer is not None:
            worker = Thread(target=run, daemon=True)
            worker.start()
            for piece in streamer:
                if piece:
                    emit(piece)
            worker.join()
        else:
            run()
        if "err" in holder:
            raise holder["err"]
        out = holder["out"]
        seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
        new = seq[prefix:].tolist()
        text = tok.decode(new, skip_special_tokens=True)
        stopped = bool(new) and int(new[-1]) == int(tok.eos_token_id)
        new_cache = getattr(out, "past_key_values", None)
        if new_cache is None and past is not None and new:
            t = torch.tensor([new], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                fwd = model(
                    input_ids=t,
                    attention_mask=torch.ones((1, (self._pkv_len(past) or 0) + len(new))),
                    past_key_values=past,
                    use_cache=True,
                )
            new_cache = fwd.past_key_values
        return [int(x) for x in new], text, stopped, new_cache

    def _generate_stream_sync(
        self,
        ids: list[int],
        n: int,
        emit: Callable[[str], None],
        *,
        rid: str = "",
    ) -> tuple[list[int], str, bool]:
        self._ensure_loaded()
        cache = self._pkv.get(rid) if rid else None
        if self._generate_fn is not None:
            new_ids, text, stopped = self._generate_sync(ids, n, rid=rid)
            if text:
                emit(text)
            return new_ids, text, stopped
        new_ids, text, stopped, new_cache = self._generate_from_cache_sync(
            ids, n, cache=cache, emit=emit
        )
        if rid and new_cache is not None:
            self._pkv[rid] = new_cache
            self._cache_len[rid] = self._pkv_len(new_cache) or (
                self._cache_len.get(rid, 0) + len(new_ids)
            )
        return new_ids, text, stopped

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
        encoded = self.last_encode_len or len(ids)
        prefix_hit = getattr(self, "_last_prefix_hit", encoded <= 1)
        self.stats.record_generate(
            predicted=predicted, prompt_tokens=encoded, prefix_hit=prefix_hit
        )
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
        rid = self._rid(state)
        if self._generate_fn is not None:
            # Stub path still parks a cache handle so later pages encode
            # only the new window, not the prefix.
            self._pkv[rid] = ("stub", len(ids))
            self._cache_len[rid] = len(ids)
            self._record_encode(len(ids), prefix_hit=False)
        else:
            cache = await asyncio.to_thread(self._prefill_cache_sync, ids)
            self._pkv[rid] = cache
            self._cache_len[rid] = self._pkv_len(cache) or len(ids)
            self._record_encode(len(ids), prefix_hit=False)
            self.stats.kernel_launches += 1
            self.kernel_launches = self.stats.kernel_launches
        self.prefill_forwards += 1
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
        rid = self._rid(state)
        try:
            new_ids, text, stopped = await asyncio.to_thread(
                self._generate_sync, ids, window, rid=rid
            )
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
        rid = self._rid(state)
        try:
            new_ids, text, stopped = await asyncio.to_thread(
                self._generate_stream_sync, ids, window, emit, rid=rid
            )
        except Exception as exc:
            raise EngineError(f"HuggingFace decode failed: {exc}") from exc
        return self._apply_decode(state, ids, new_ids, text, stopped, grammar_span=grammar_span)

    async def abort_generation(self, state: EngineState) -> None:
        rid = state.engine_request_id
        if rid:
            self._pkv.pop(rid, None)
            self._cache_len.pop(rid, None)
