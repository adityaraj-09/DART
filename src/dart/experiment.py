"""Kill-test and Andes-complete control experiment.

Slice 0/1 from the product plan: on one producer, compare always-push vs
credit-gated decode. If credit does not cut generated-but-unconsumed tokens
and KV high-water, the inversion is aesthetic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from dart.consumers import DrainPacer, JsonNeedPacer, ReadingPacer
from dart.engine.synthetic import SyntheticEngine
from dart.runtime import DartRuntime
from dart.types import RuntimeConfig


Mode = Literal["push", "reading", "api", "json"]


async def run_workload(
    mode: Mode,
    *,
    max_tokens: int = 512,
    duration_s: float = 2.0,
    step_latency_s: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    engine = SyntheticEngine(step_latency_s=step_latency_s, seed=seed)
    cfg = RuntimeConfig(
        decode_quota=max_tokens,
        w_init=16 if mode != "push" else 128,
        w_max=128 if mode != "push" else 512,
        startup_credit=16,
        poll_interval_s=0.002,
        t_decode_s=max(step_latency_s, 0.01),
        lease_ttl_s=max(60.0, duration_s + 10),
        interest_lifetime_s=5.0,
    )
    rt = DartRuntime(engine, cfg)
    await rt.start()
    t0 = time.monotonic()
    handle = await rt.open("Write a long essay about named continuations.", max_tokens=max_tokens)
    if mode == "push":
        pacer = DrainPacer(window=min(32, max_tokens))
    elif mode == "api":
        pacer = ReadingPacer(tokens_per_sec=200.0, burst=32)
    elif mode == "json":
        pacer = JsonNeedPacer(burst=64, pause_s=0.05)
    else:
        pacer = ReadingPacer(tokens_per_sec=30.0, burst=16)

    consumed = 0
    displayed = 0

    async def _consume() -> None:
        nonlocal consumed, displayed
        async for data in rt.consume(handle, pacer, max_tokens=max_tokens):
            consumed += data.token_count()
            displayed += data.token_count()
            if time.monotonic() - t0 >= duration_s:
                break

    try:
        await asyncio.wait_for(_consume(), timeout=duration_s + 2.0)
    except TimeoutError:
        pass
    await asyncio.sleep(0)  # let scheduler account skips
    snap = rt.metrics_snapshot()
    await rt.close(handle.cont_id)
    await rt.aclose()
    totals = snap["totals"]
    item = snap["items"][0] if snap["items"] else {}
    generated = totals["tokens_generated"]
    return {
        "mode": mode,
        "duration_s": time.monotonic() - t0,
        "tokens_generated": generated,
        "tokens_consumed": consumed,
        "tokens_displayed": displayed,
        "tokens_generated_unconsumed": max(0, generated - consumed),
        "decode_kernel_launches": totals["decode_kernel_launches"],
        "decode_steps_skipped": totals["decode_steps_skipped"],
        "kv_bytes": snap["kv_bytes"],
        "kv_high_water": snap["kv_high_water"],
        "kv_gpu_bytes": snap["kv_gpu_bytes"],
        "engine_kernel_launches": engine.kernel_launches,
        "engine_decode_tokens": engine.decode_tokens,
        "cwnd": item.get("cwnd"),
        "k": item.get("k"),
        "generated_over_consumed": (
            generated / consumed if consumed else (float("inf") if generated else 0.0)
        ),
    }


async def compare(
    *,
    duration_s: float = 2.0,
    max_tokens: int = 256,
    step_latency_s: float = 0.0,
) -> dict[str, Any]:
    push = await run_workload("push", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s)
    reading = await run_workload(
        "reading", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s
    )
    api = await run_workload("api", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s)
    bursty = await run_workload(
        "json", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s
    )
    def _cut(metric: str) -> float | None:
        a, b = push[metric], reading[metric]
        if not a:
            return None
        return (a - b) / a

    return {
        "push": push,
        "credit_reading_30tps": reading,
        "credit_api_200tps": api,
        "credit_json_bursty": bursty,
        "kill_test": {
            "unconsumed_cut_vs_push": _cut("tokens_generated_unconsumed")
            if push["tokens_generated_unconsumed"]
            else 1.0 if reading["tokens_generated_unconsumed"] == 0 else _cut("tokens_generated"),
            "kv_high_water_cut_vs_push": _cut("kv_high_water"),
            "generated_cut_vs_push": _cut("tokens_generated"),
            "survive": (
                (reading["tokens_generated"] < push["tokens_generated"] * 0.9)
                or (reading["kv_high_water"] < push["kv_high_water"])
            )
            and reading["tokens_consumed"] > 0,
        },
    }
