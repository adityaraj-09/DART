"""Kernel telemetry the *engine* reports — not DART's wrapper counters.

Paper eval must show that credit-gating zeroes the producer's own
forward/generate count (vLLM HTTP POSTs, llama.cpp /completion, or a
scraped Prometheus counter), not merely `cont.metrics.decode_kernel_launches`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KernelStats:
    """Producer-side counters. HTTP adapters increment these only after a
    successful generate/completion round-trip to the engine process.
    """

    kernel_launches: int = 0
    tokens_predicted: int = 0
    prompt_tokens: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    admission_reentries: int = 0
    http_calls: int = 0
    last_prompt_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def record_generate(
        self,
        *,
        predicted: int,
        prompt_tokens: int | None = None,
        prefix_hit: bool | None = None,
    ) -> None:
        self.kernel_launches += 1
        self.http_calls += 1
        self.tokens_predicted += max(0, predicted)
        if prompt_tokens is not None:
            # If the engine re-sent a longer prompt than last time plus the
            # new tail, prefix cache likely missed → implicit re-prefill /
            # admission re-entry on the HTTP path.
            if self.last_prompt_tokens and prompt_tokens > self.last_prompt_tokens + predicted + 8:
                self.admission_reentries += 1
                self.prefix_cache_misses += 1
            elif prefix_hit is False:
                self.prefix_cache_misses += 1
            elif prefix_hit is True:
                self.prefix_cache_hits += 1
            self.prompt_tokens += prompt_tokens
            self.last_prompt_tokens = prompt_tokens

    def snapshot(self) -> dict[str, Any]:
        return {
            "engine_kernel_launches": self.kernel_launches,
            "engine_tokens_predicted": self.tokens_predicted,
            "engine_prompt_tokens": self.prompt_tokens,
            "engine_prefix_cache_hits": self.prefix_cache_hits,
            "engine_prefix_cache_misses": self.prefix_cache_misses,
            "engine_admission_reentries": self.admission_reentries,
            "engine_http_calls": self.http_calls,
        }


def stats_of(engine: object) -> KernelStats:
    raw = getattr(engine, "stats", None)
    if isinstance(raw, KernelStats):
        return raw
    # SyntheticEngine and older adapters expose kernel_launches on self.
    s = KernelStats()
    s.kernel_launches = int(getattr(engine, "kernel_launches", 0) or 0)
    s.tokens_predicted = int(getattr(engine, "decode_tokens", 0) or 0)
    return s
