"""llama.cpp HTTP producer (`n_predict` = credit window, `cache_prompt` on).

Kernel launches are the engine's `/completion` POSTs (and `/metrics`
`llamacpp_decode_calls_total` when present), not DART wrapper increments.
"""

from __future__ import annotations

import os

import httpx

from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.stats import KernelStats
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
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = (
            base_url or os.environ.get("DART_LLAMACPP_URL") or "http://127.0.0.1:8080"
        ).rstrip("/")
        self.config = config or ModelConfig(model_id=model_id, tokenizer_hash="llama.cpp")
        self.timeout_s = timeout_s
        self._client = client
        self.stats = KernelStats()
        self.kernel_launches = 0
        self.decode_tokens = 0
        self.supports_rollback = False

    async def _http(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    def _extents(self, state: EngineState) -> list[KVExtent]:
        cfg = self.config
        nbytes = bytes_per_extent(cfg.n_kv_heads, cfg.head_dim, cfg.block_size)
        n_blocks = max(1, (max(state.pos, 1) + cfg.block_size - 1) // cfg.block_size)
        out: list[KVExtent] = []
        for layer in range(min(cfg.n_layers, 8)):
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

    async def scrape_engine_metrics(self) -> dict[str, int]:
        client = await self._http()
        try:
            r = await client.get(f"{self.base_url}/engine/counters")
            if r.status_code == 200:
                return {k: int(v) for k, v in r.json().items()}
        except Exception:
            pass
        try:
            r = await client.get(f"{self.base_url}/metrics")
            r.raise_for_status()
            launches = 0
            predicted = 0
            for line in r.text.splitlines():
                if line.startswith("llamacpp_decode_calls_total "):
                    launches = int(float(line.rsplit(" ", 1)[-1]))
                if line.startswith("llamacpp_tokens_predicted "):
                    predicted = int(float(line.rsplit(" ", 1)[-1]))
            return {"kernel_launches": launches, "tokens_predicted": predicted}
        except Exception:
            return self.stats.snapshot()

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        text = prompt.as_text()
        ids = prompt.token_ids or [1] * max(1, len(text.split()) + 4)
        state.prompt_ids = ids
        state.pos = len(ids)
        state.prefix_text = text
        state.assistant_text = ""
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
                token_ids=[],
                text="",
                extents=state.kv_extents,
                kv_root=state.kv_root,
                kernel_launched=False,
            )
        prompt = (state.prefix_text or "") + (state.assistant_text or "")
        body: dict[str, object] = {
            "prompt": prompt,
            "n_predict": max(1, n if not grammar_span else max(n, 32)),
            "cache_prompt": True,
            "temperature": state.sampler.temperature,
        }
        if grammar_span:
            body["json_schema"] = {"type": "object"}
        client = await self._http()
        try:
            resp = await client.post(f"{self.base_url}/completion", json=body)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise EngineError(f"llama.cpp decode failed: {exc}") from exc
        text = str(payload.get("content") or "")
        predicted = int(
            payload.get("tokens_predicted")
            or (payload.get("timings") or {}).get("predicted_n")
            or max(1, len(text.split()) or 1)
        )
        prompt_tokens = int(
            payload.get("tokens_evaluated")
            or (payload.get("timings") or {}).get("prompt_n")
            or 0
        ) or None
        self.stats.record_generate(predicted=predicted, prompt_tokens=prompt_tokens)
        self.kernel_launches = self.stats.kernel_launches
        self.decode_tokens = self.stats.tokens_predicted
        state.assistant_text += text
        new_ids = list(range(state.pos, state.pos + predicted))
        state.output_ids.extend(new_ids)
        state.pos += len(new_ids)
        state.kv_extents = self._extents(state)
        state.kv_root = root_from_extents(state.kv_extents)
        state.stopped = bool(payload.get("stop"))
        return DecodeResult(
            token_ids=new_ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=state.stopped,
            grammar_span=bool(grammar_span),
        )
