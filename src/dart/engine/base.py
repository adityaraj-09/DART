"""Engine protocol: the producer that may run a decode kernel."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from dart.types import EngineState, KVExtent, ModelConfig, Prompt


class PrefillResult(BaseModel):
    state: EngineState
    text: str = ""
    extents: list[KVExtent] = Field(default_factory=list)


class DecodeResult(BaseModel):
    token_ids: list[int]
    text: str
    extents: list[KVExtent] = Field(default_factory=list)
    kv_root: str
    stopped: bool = False
    grammar_span: bool = False
    kernel_launched: bool = True


@runtime_checkable
class Engine(Protocol):
    """A compatible producer. Handover is fetching named KV, not thawing a process."""

    model_id: str
    config: ModelConfig

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult: ...

    async def decode(
        self,
        state: EngineState,
        n: int,
        *,
        grammar_span: str | None = None,
    ) -> DecodeResult: ...
