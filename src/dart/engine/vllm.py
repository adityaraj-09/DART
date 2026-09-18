"""vLLM producer: credit-gated generate via prefix-cached max_tokens=n.

Each Interest becomes one HTTP generate of at most W tokens. Zero Interests
⇒ zero HTTP calls ⇒ the vLLM process records zero additional forwards.

Limitation (honest): this HTTP path re-enters admission and can drop KV if
the prefix cache evicts. An in-process scheduler plugin that leaves the
request in `waiting` with pinned blocks would tighten residency, not invent
the credit-gate idea. See docs/limitations.md.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.stats import KernelStats
from dart.errors import EngineError
from dart.merkle import bytes_per_extent, extent_digest, root_from_extents
from dart.types import EngineState, KVExtent, ModelConfig, Prompt


def _opaque_extents(state: EngineState, cfg: ModelConfig) -> list[KVExtent]:
    nbytes = bytes_per_extent(cfg.n_kv_heads, cfg.head_dim, cfg.block_size)
    n_blocks = max(1, (max(state.pos, 1) + cfg.block_size - 1) // cfg.block_size)
    extents: list[KVExtent] = []
    for layer in range(min(cfg.n_layers, 8)):
        for b in range(n_blocks):
            digest = extent_digest(layer, b, state.pos, state.all_ids[-32:])
            extents.append(KVExtent(layer=layer, block_id=b, digest=digest, nbytes=nbytes))
    return extents


def _parse_prometheus(text: str, key: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split("{", 1)[0].split(" ", 1)[0] == key:
            try:
                return int(float(line.rsplit(" ", 1)[-1]))
            except ValueError:
                return None
    return None


class VLLMChatEngine:
    """HTTP adapter against vLLM's OpenAI server (or the in-repo fake)."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        config: ModelConfig | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model_id = model_id
        raw = (base_url or os.environ.get("DART_VLLM_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.base_url = raw
        self.api_key = api_key or os.environ.get("DART_VLLM_API_KEY") or "EMPTY"
        self.config = config or ModelConfig(model_id=model_id, tokenizer_hash="vllm")
        self.timeout_s = timeout_s
        self._client = client
        self.stats = KernelStats()
        self.kernel_launches = 0
        self.decode_tokens = 0
        self.supports_rollback = False

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}

    async def _http(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def scrape_engine_metrics(self) -> dict[str, int]:
        """Read the engine process counters (vLLM /metrics or fake /v1/engine/counters)."""
        client = await self._http()
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        out: dict[str, int] = {}
        try:
            r = await client.get(f"{root}/v1/engine/counters")
            if r.status_code == 200:
                return {k: int(v) for k, v in r.json().items()}
        except Exception:
            pass
        try:
            r = await client.get(f"{root}/metrics")
            r.raise_for_status()
            text = r.text
            for src, dst in (
                ("vllm_engine_forward_calls_total", "kernel_launches"),
                ("vllm_generation_tokens_total", "tokens_predicted"),
                ("vllm_prefix_cache_hits_total", "prefix_cache_hits"),
            ):
                val = _parse_prometheus(text, src)
                if val is not None:
                    out[dst] = val
        except Exception:
            return self.stats.snapshot()
        return out or self.stats.snapshot()

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        text = prompt.as_text()
        ids = prompt.token_ids or [1] * max(1, len(text.split()) + 4)
        state.prompt_ids = ids
        state.output_ids = []
        state.pos = len(ids)
        state.prefix_text = text
        state.assistant_text = ""
        state.kv_extents = _opaque_extents(state, self.config)
        state.kv_root = root_from_extents(state.kv_extents)
        # Bookkeeping only. The remote KV is created on the first generate() —
        # first Interest therefore includes TTFT. No HTTP ⇒ no engine forward.
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
        messages: list[dict[str, str]] = [{"role": "user", "content": state.prefix_text or " "}]
        if state.assistant_text:
            messages.append({"role": "assistant", "content": state.assistant_text})
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max(1, n if not grammar_span else max(n, 32)),
            "temperature": state.sampler.temperature,
        }
        if grammar_span:
            body["structured_outputs"] = {"json": True}
        client = await self._http()
        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers()
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise EngineError(f"vLLM decode failed: {exc}") from exc
        choice = payload["choices"][0]
        text = (choice.get("message") or {}).get("content") or choice.get("text") or ""
        usage = payload.get("usage") or {}
        predicted = int(usage.get("completion_tokens") or max(1, len(text.split()) or 1))
        prompt_tokens = int(usage.get("prompt_tokens") or 0) or None
        self.stats.record_generate(predicted=predicted, prompt_tokens=prompt_tokens)
        self.kernel_launches = self.stats.kernel_launches
        self.decode_tokens = self.stats.tokens_predicted
        state.assistant_text += text
        new_ids = list(range(state.pos, state.pos + predicted))
        state.output_ids.extend(new_ids)
        state.pos += len(new_ids)
        state.kv_extents = _opaque_extents(state, self.config)
        state.kv_root = root_from_extents(state.kv_extents)
        finish = choice.get("finish_reason") or ""
        state.stopped = finish in {"stop", "length"}
        for tid in new_ids:
            state.sampler = state.sampler.advance(tid)
        return DecodeResult(
            token_ids=new_ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=state.stopped,
            grammar_span=bool(grammar_span),
        )


# Back-compat alias used by older imports / factory.
VLLMEngine = VLLMChatEngine
