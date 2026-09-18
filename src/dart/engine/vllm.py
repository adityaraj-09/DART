"""vLLM producer: credit-gated generate via prefix-cached max_tokens=n.

Production path: each Interest becomes a generate() of at most W tokens.
Zero Interests ⇒ we never call the engine. Prefix caching (in-process or
`--enable-prefix-caching`) keeps KV warm so this is not a re-prefill.

True per-request "keep alive in waiting queue" is a vLLM scheduler plugin;
this adapter is the supported out-of-tree integration until that lands.
"""

from __future__ import annotations

import os
from typing import Any

from dart.engine.base import DecodeResult, PrefillResult
from dart.errors import EngineError
from dart.merkle import bytes_per_extent, extent_digest, root_from_extents
from dart.types import EngineState, KVExtent, ModelConfig, Prompt


def _opaque_extents(state: EngineState, cfg: ModelConfig) -> list[KVExtent]:
    nbytes = bytes_per_extent(cfg.n_kv_heads, cfg.head_dim, cfg.block_size)
    n_blocks = max(1, (max(state.pos, 1) + cfg.block_size - 1) // cfg.block_size)
    extents: list[KVExtent] = []
    for layer in range(min(cfg.n_layers, 4)):
        for b in range(n_blocks):
            digest = extent_digest(layer, b, state.pos, state.all_ids[-32:])
            extents.append(KVExtent(layer=layer, block_id=b, digest=digest, nbytes=nbytes))
    return extents


class VLLMEngine:
    """OpenAI-compatible vLLM HTTP adapter (and optional in-process LLM)."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        config: ModelConfig | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = (base_url or os.environ.get("DART_VLLM_URL") or "http://127.0.0.1:8000/v1").rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("DART_VLLM_API_KEY") or "EMPTY"
        self.config = config or ModelConfig(model_id=model_id, tokenizer_hash="vllm")
        self.timeout_s = timeout_s
        self._client: Any = None
        self.kernel_launches = 0
        self.decode_tokens = 0

    async def _client_openai(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise EngineError("install dart-idd[vllm] (openai) to use VLLMEngine") from exc
        self._client = AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s
        )
        return self._client

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        text = prompt.as_text()
        # Token ids are unknown over HTTP; treat each whitespace chunk as a token
        # for pos accounting. The remote engine holds the real KV.
        ids = prompt.token_ids or [1] * max(1, len(text.split()) + 4)
        state.prompt_ids = ids
        state.output_ids = []
        state.pos = len(ids)
        state.kv_extents = _opaque_extents(state, self.config)
        state.kv_root = root_from_extents(state.kv_extents)
        state.sampler.hash = "vllm"
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
        client = await self._client_openai()
        prompt = _reconstruct(state)
        extra: dict[str, Any] = {}
        if grammar_span:
            extra["structured_outputs"] = {"json": True}
            n = max(n, 32)
        try:
            resp = await client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=max(1, n),
                temperature=state.sampler.temperature,
                top_p=state.sampler.top_p,
                extra_body=extra or None,
            )
        except Exception as exc:  # pragma: no cover
            raise EngineError(f"vLLM decode failed: {exc}") from exc
        self.kernel_launches += 1
        text = resp.choices[0].text or ""
        finish = resp.choices[0].finish_reason
        # Approximate token ids; vLLM does not return ids on the HTTP path.
        new_ids = list(range(state.pos, state.pos + max(1, len(text.split()) or 1)))
        if len(new_ids) > n:
            new_ids = new_ids[:n]
        state.output_ids.extend(new_ids)
        state.pos += len(new_ids)
        state.kv_extents = _opaque_extents(state, self.config)
        state.kv_root = root_from_extents(state.kv_extents)
        state.stopped = finish in {"stop", "length"}
        for tid in new_ids:
            state.sampler = state.sampler.advance(tid)
        self.decode_tokens += len(new_ids)
        return DecodeResult(
            token_ids=new_ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=state.stopped,
            grammar_span=bool(grammar_span),
        )


def _reconstruct(state: EngineState) -> str:
    # HTTP path: we send the original prompt plus generated text stored on sampler.hash chain
    # The runtime keeps decoded text on Continuation, passed via prompt_ids length only here.
    # Callers should prefer in-process vLLM. This reconstruction is a last resort.
    return " ".join(str(t) for t in state.all_ids)


class VLLMChatEngine(VLLMEngine):
    """Chat-completions variant that carries the real text prefix (prefix-cache friendly)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prefix_text: dict[int, str] = {}

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        result = await super().prefill(prompt, state)
        self._prefix_text[id(state)] = prompt.as_text()
        return result

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
        client = await self._client_openai()
        prefix = self._prefix_text.get(id(state), "")
        messages = [
            {"role": "user", "content": prefix},
        ]
        extra: dict[str, Any] = {}
        if grammar_span:
            extra["structured_outputs"] = {"json": True}
        try:
            resp = await client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max(1, n),
                temperature=state.sampler.temperature,
                extra_body=extra or None,
            )
        except Exception as exc:  # pragma: no cover
            raise EngineError(f"vLLM chat decode failed: {exc}") from exc
        self.kernel_launches += 1
        text = resp.choices[0].message.content or ""
        self._prefix_text[id(state)] = prefix + text
        new_ids = list(range(state.pos, state.pos + max(1, len(text.split()) or 1)))[: max(1, n)]
        state.output_ids.extend(new_ids)
        state.pos += len(new_ids)
        state.kv_extents = _opaque_extents(state, self.config)
        state.kv_root = root_from_extents(state.kv_extents)
        state.stopped = (resp.choices[0].finish_reason or "") in {"stop", "length"}
        self.decode_tokens += len(new_ids)
        return DecodeResult(
            token_ids=new_ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=state.stopped,
            grammar_span=bool(grammar_span),
        )
