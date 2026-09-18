"""llama.cpp HTTP producer (`n_predict` = credit window, `cache_prompt` on)."""

from __future__ import annotations

import os

import httpx

from dart.engine.base import DecodeResult, PrefillResult
from dart.errors import EngineError
from dart.merkle import bytes_per_extent, extent_digest, root_from_extents
from dart.types import EngineState, KVExtent, ModelConfig, Prompt


class LlamaCppEngine:
    def __init__(
        self,
        model_id: str = "llama.cpp",
        *,
        base_url: str | None = None,
        config: ModelConfig | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = (base_url or os.environ.get("DART_LLAMACPP_URL") or "http://127.0.0.1:8080").rstrip(
            "/"
        )
        self.config = config or ModelConfig(model_id=model_id, tokenizer_hash="llama.cpp")
        self.timeout_s = timeout_s
        self._prefix: dict[int, str] = {}
        self.kernel_launches = 0
        self.decode_tokens = 0

    def _extents(self, state: EngineState) -> list[KVExtent]:
        cfg = self.config
        nbytes = bytes_per_extent(cfg.n_kv_heads, cfg.head_dim, cfg.block_size)
        n_blocks = max(1, (max(state.pos, 1) + cfg.block_size - 1) // cfg.block_size)
        out: list[KVExtent] = []
        for layer in range(min(cfg.n_layers, 4)):
            for b in range(n_blocks):
                out.append(
                    KVExtent(
                        layer=layer,
                        block_id=b,
                        digest=extent_digest(layer, b, state.pos, state.all_ids[-16:]),
                        nbytes=nbytes,
                    )
                )
        return out

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        text = prompt.as_text()
        self._prefix[id(state)] = text
        ids = prompt.token_ids or [1] * max(1, len(text.split()) + 4)
        state.prompt_ids = ids
        state.pos = len(ids)
        state.kv_extents = self._extents(state)
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
                token_ids=[], text="", extents=state.kv_extents, kv_root=state.kv_root, kernel_launched=False
            )
        prompt = self._prefix.get(id(state), "")
        body: dict[str, object] = {
            "prompt": prompt,
            "n_predict": max(1, n),
            "cache_prompt": True,
            "temperature": state.sampler.temperature,
        }
        if grammar_span:
            body["json_schema"] = {"type": "object"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(f"{self.base_url}/completion", json=body)
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # pragma: no cover
            raise EngineError(f"llama.cpp decode failed: {exc}") from exc
        self.kernel_launches += 1
        text = str(payload.get("content") or "")
        self._prefix[id(state)] = prompt + text
        new_ids = list(range(state.pos, state.pos + max(1, len(text.split()) or 1)))[: max(1, n)]
        state.output_ids.extend(new_ids)
        state.pos += len(new_ids)
        state.kv_extents = self._extents(state)
        state.kv_root = root_from_extents(state.kv_extents)
        state.stopped = bool(payload.get("stop"))
        self.decode_tokens += len(new_ids)
        return DecodeResult(
            token_ids=new_ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=state.stopped,
            grammar_span=bool(grammar_span),
        )
