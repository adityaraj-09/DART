"""Andes-complete control: push-then-pace vs refuse-to-decode.

Andes generates into a client pacer and pauses the kernel only when the
pacer depth exceeds a watermark. Tokens sitting in that pacer still cost
residual-stream steps and KV. IDD never generates them.

Kill criterion 1: if Andes watermark pause captures ≥90% of the joule/KV
win, IDD is an Andes engineering patch. This module is that experiment.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from dart.engine.base import Engine
from dart.engine.stats import stats_of
from dart.engine.synthetic import SyntheticEngine
from dart.types import EngineState, Prompt


async def run_andes(
    *,
    consume_tps: float = 30.0,
    watermark: int = 32,
    max_tokens: int = 128,
    duration_s: float = 1.0,
    segment_size: int = 16,
    step_latency_s: float = 0.0,
    engine: Engine | None = None,
    prompt: str = "Write a long essay about named continuations.",
) -> dict[str, Any]:
    """Push decode into a pacer; pause only when len(buffer) >= watermark."""

    eng = engine or SyntheticEngine(step_latency_s=step_latency_s, seed=0)
    state = EngineState()
    await eng.prefill(Prompt(text=prompt), state)
    buffer: list[int] = []
    generated = 0
    consumed = 0
    paused_ticks = 0
    inventory_hw = 0
    t0 = time.monotonic()
    t_decode = max(step_latency_s, 0.01)
    consume_interval = 1.0 / max(consume_tps, 1e-6)
    stop = asyncio.Event()

    async def producer() -> None:
        nonlocal generated, paused_ticks, inventory_hw
        while not stop.is_set() and generated < max_tokens:
            if time.monotonic() - t0 >= duration_s:
                break
            if len(buffer) >= watermark:
                paused_ticks += 1
                await asyncio.sleep(t_decode)
                continue
            n = min(segment_size, watermark - len(buffer), max_tokens - generated)
            if n <= 0:
                await asyncio.sleep(t_decode)
                continue
            result = await eng.decode(state, n)
            if not result.kernel_launched:
                break
            buffer.extend(result.token_ids)
            generated += len(result.token_ids)
            inventory_hw = max(inventory_hw, len(buffer))
            if result.stopped:
                break

    async def consumer() -> None:
        nonlocal consumed, inventory_hw
        nxt = time.monotonic()
        while not stop.is_set():
            if time.monotonic() - t0 >= duration_s and not buffer:
                break
            if consumed >= max_tokens:
                break
            now = time.monotonic()
            if now < nxt:
                await asyncio.sleep(min(t_decode, nxt - now))
                continue
            nxt = now + consume_interval
            if buffer:
                buffer.pop(0)
                consumed += 1
                inventory_hw = max(inventory_hw, len(buffer))
            elif generated >= max_tokens or time.monotonic() - t0 >= duration_s:
                break

    await asyncio.gather(producer(), consumer())
    stop.set()
    st = stats_of(eng)
    inventory = generated - consumed
    return {
        "mode": "andes",
        "watermark": watermark,
        "consume_tps": consume_tps,
        "tokens_generated": generated,
        "tokens_consumed": consumed,
        "tokens_generated_unconsumed": max(0, inventory),
        "pacer_inventory": max(0, inventory),
        "pacer_inventory_hw": inventory_hw,
        "decode_kernel_launches": st.kernel_launches,
        "engine_kernel_launches": st.kernel_launches,
        "engine_tokens_predicted": st.tokens_predicted,
        "paused_ticks": paused_ticks,
        "kv_high_water": sum(e.nbytes for e in state.kv_extents),
        "duration_s": time.monotonic() - t0,
    }


async def run_push(
    *,
    max_tokens: int = 128,
    step_latency_s: float = 0.0,
    engine: Engine | None = None,
    prompt: str = "Write a long essay about named continuations.",
) -> dict[str, Any]:
    """Always-push: generate max_tokens as fast as the kernel allows."""

    eng = engine or SyntheticEngine(step_latency_s=step_latency_s, seed=0)
    state = EngineState()
    await eng.prefill(Prompt(text=prompt), state)
    t0 = time.monotonic()
    generated = 0
    while generated < max_tokens:
        n = min(16, max_tokens - generated)
        result = await eng.decode(state, n)
        if not result.kernel_launched or not result.token_ids:
            break
        generated += len(result.token_ids)
        if result.stopped:
            break
    st = stats_of(eng)
    return {
        "mode": "push",
        "tokens_generated": generated,
        "tokens_consumed": generated,
        "tokens_generated_unconsumed": 0,
        "pacer_inventory": 0,
        "pacer_inventory_hw": 0,
        "decode_kernel_launches": st.kernel_launches,
        "engine_kernel_launches": st.kernel_launches,
        "engine_tokens_predicted": st.tokens_predicted,
        "kv_high_water": sum(e.nbytes for e in state.kv_extents),
        "duration_s": time.monotonic() - t0,
    }
