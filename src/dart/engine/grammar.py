"""Grammar as one named Data object vs token-at-a-time logit masking.

Jump-forward: Interest(grammar/span/S) → one kernel launch → one Data.
Masked: the same span is N tokens, each a kernel launch with a logit mask.
"""

from __future__ import annotations

from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.synthetic import SyntheticEngine
from dart.types import EngineState, Prompt

_SPAN_JSON = {
    "next-value": '{"status":"ok","id":7}',
    "json_value": '{"ok":true}',
    "tool-args": '{"name":"search","query":"dart idd"}',
    "string": '"alpha"',
}


class JumpForwardEngine(SyntheticEngine):
    """Forced span in a single Data object (typed Interest)."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.grammar_mode = "jump"


class LogitMaskedEngine(SyntheticEngine):
    """Ignores n>1 under grammar: one token per kernel, like a logit mask loop."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.grammar_mode = "mask"
        self._span_queue: list[int] = []
        self._span_text_queue: list[str] = []

    async def decode(
        self,
        state: EngineState,
        n: int,
        *,
        grammar_span: str | None = None,
    ) -> DecodeResult:
        if grammar_span and not self._span_queue:
            text = _SPAN_JSON.get(grammar_span, '{"ok":true}')
            ids = [ord(c) % 32000 for c in text]
            self._span_queue = ids
            self._span_text_queue = list(text)
        if self._span_queue:
            tid = self._span_queue.pop(0)
            piece = self._span_text_queue.pop(0) if self._span_text_queue else ""
            # Force n=1
            state.output_ids.append(tid)
            state.pos += 1
            self.kernel_launches += 1
            self.stats.kernel_launches = self.kernel_launches
            self.decode_tokens += 1
            self.stats.tokens_predicted = self.decode_tokens
            more = bool(self._span_queue)
            return DecodeResult(
                token_ids=[tid],
                text=piece,
                extents=state.kv_extents,
                kv_root=state.kv_root,
                stopped=not more,
                grammar_span=True,
                kernel_launched=True,
            )
        return await super().decode(state, n, grammar_span=None)


def _split_keep(text: str, n: int) -> list[str]:
    if n <= 1:
        return [text]
    # Even chunks so launches == tokens and concatenation reconstructs JSON.
    size = max(1, (len(text) + n - 1) // n)
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    while len(parts) < n:
        parts.append("")
    return parts[:n]


async def masked_span_launches(engine: LogitMaskedEngine, prompt: str, span: str) -> dict[str, int]:
    """Drive token-at-a-time grammar until the span is complete."""
    state = EngineState()
    await engine.prefill(Prompt(text=prompt), state)
    launches_before = engine.kernel_launches
    chunks: list[str] = []
    ids: list[int] = []
    for _ in range(64):
        result = await engine.decode(state, 1, grammar_span=span)
        chunks.append(result.text)
        ids.extend(result.token_ids)
        if result.stopped or not getattr(engine, "_span_queue", [1]):
            break
    return {
        "kernel_launches": engine.kernel_launches - launches_before,
        "tokens": len(ids),
        "text_len": len("".join(chunks)),
    }


async def jump_span_launches(engine: JumpForwardEngine | SyntheticEngine, prompt: str, span: str) -> dict[str, int]:
    state = EngineState()
    await engine.prefill(Prompt(text=prompt), state)
    before = engine.kernel_launches
    result = await engine.decode(state, 1, grammar_span=span)
    return {
        "kernel_launches": engine.kernel_launches - before,
        "tokens": len(result.token_ids),
        "text_len": len(result.text),
        "one_data": True,
    }
