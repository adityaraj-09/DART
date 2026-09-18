"""Cache-only producer: never runs a decode kernel.

A second process with a shared FileCAS answers Interests by reading named
Data. That is the not-Andes defense: the object exists in the address space
and a peer satisfies it without a GPU.
"""

from __future__ import annotations

from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.stats import KernelStats
from dart.errors import EngineError
from dart.types import EngineState, ModelConfig, Prompt


class CacheOnlyEngine:
    """Peer producer. Prefill is identity; decode is forbidden."""

    def __init__(self, model_id: str = "cas-peer") -> None:
        self.model_id = model_id
        self.config = ModelConfig(model_id=model_id, tokenizer_hash="cas-peer")
        self.stats = KernelStats()
        self.kernel_launches = 0
        self.decode_tokens = 0
        self.prefills = 0
        self.supports_rollback = False

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        self.prefills += 1
        state.prefix_text = prompt.as_text()
        state.kv_root = state.kv_root or "0" * 64
        return PrefillResult(state=state, text=state.prefix_text, extents=[])

    async def decode(
        self,
        state: EngineState,
        n: int,
        *,
        grammar_span: str | None = None,
    ) -> DecodeResult:
        raise EngineError(
            "CacheOnlyEngine cannot decode; satisfy the Interest from CAS or Nack"
        )
