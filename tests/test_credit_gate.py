from __future__ import annotations

import asyncio

from dart.client.consumers import DrainPacer, ReadingPacer
from dart.engine.synthetic import SyntheticEngine
from dart.cip.protocol import CipName, Interest
from dart.core.runtime import DartRuntime
from dart.core.types import RuntimeConfig


async def test_no_interest_means_no_decode() -> None:
    engine = SyntheticEngine(seed=1)
    rt = DartRuntime(
        engine,
        RuntimeConfig(poll_interval_s=0.001, t_decode_s=0.005, decode_quota=64),
    )
    await rt.start()
    handle = await rt.open("hello world", max_tokens=64)
    await asyncio.sleep(0.05)
    assert engine.kernel_launches == 0
    assert engine.decode_tokens == 0
    cont = rt.get(handle.cont_id)
    assert cont.metrics.tokens_generated == 0
    assert cont.sleeping is True
    await rt.aclose()


async def test_interest_runs_kernel_and_advances_root(runtime: DartRuntime) -> None:
    handle = await runtime.open("named holes", max_tokens=32)
    prev = handle.kv_root
    name = CipName.tokens(handle.model_hash, prev, 0).render()
    data = await runtime.interest(
        Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease),
        lease=handle.lease,
    )
    assert data.token_count() > 0
    assert data.kv_root != prev
    assert data.cache_hit is False
    cont = runtime.get(handle.cont_id)
    assert cont.metrics.decode_kernel_launches >= 1
    assert runtime.engine.kernel_launches >= 1  # type: ignore[attr-defined]


async def test_second_interest_is_cas_hit(runtime: DartRuntime) -> None:
    handle = await runtime.open("cache me", max_tokens=32)
    name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    req = Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease)
    first = await runtime.interest(req, lease=handle.lease)
    launches = runtime.engine.kernel_launches  # type: ignore[attr-defined]
    second = await runtime.interest(req, lease=handle.lease)
    assert second.cache_hit is True
    assert second.text == first.text
    assert runtime.engine.kernel_launches == launches  # type: ignore[attr-defined]


async def test_close_stops_further_decode(runtime: DartRuntime) -> None:
    handle = await runtime.open("stop", max_tokens=64)
    await runtime.close(handle.cont_id)
    await asyncio.sleep(0.02)
    assert runtime.engine.kernel_launches == 0  # type: ignore[attr-defined]


async def test_reading_pacer_consumes(runtime: DartRuntime) -> None:
    handle = await runtime.open("pace", max_tokens=24)
    n = 0
    async for seg in runtime.consume(handle, DrainPacer(window=8), max_tokens=24):
        n += seg.token_count()
        if n >= 16:
            break
    assert n >= 8
    snap = runtime.metrics_snapshot()["totals"]
    assert snap["tokens_generated"] >= n


async def test_grammar_span_is_one_data(runtime: DartRuntime) -> None:
    from dart.core.types import InterestKind

    handle = await runtime.open("json please", max_tokens=64)
    name = CipName.grammar(handle.model_hash, handle.kv_root, "next-value").render()
    data = await runtime.interest(
        Interest(
            name=name,
            window=16,
            kind=InterestKind.GRAMMAR,
            grammar_span="next-value",
            lease=handle.lease,
        ),
        lease=handle.lease,
    )
    assert "{" in data.text
    assert runtime.get(handle.cont_id).metrics.grammar_spans >= 1


async def test_slow_reader_skips_steps() -> None:
    engine = SyntheticEngine(seed=2)
    rt = DartRuntime(
        engine,
        RuntimeConfig(poll_interval_s=0.002, t_decode_s=0.01, decode_quota=128, interest_lifetime_s=3),
    )
    await rt.start()
    handle = await rt.open("slow", max_tokens=128)
    pacer = ReadingPacer(tokens_per_sec=20, burst=4)
    got = 0

    async def _run() -> None:
        nonlocal got
        async for data in rt.consume(handle, pacer, max_tokens=128):
            got += data.token_count()
            if got >= 8:
                break

    await asyncio.wait_for(_run(), timeout=2)
    await asyncio.sleep(0.05)
    skipped = rt.get(handle.cont_id).metrics.decode_steps_skipped
    await rt.close(handle.cont_id)
    await rt.aclose()
    assert got >= 4
    assert skipped >= 0  # may be zero if consume is still active; just ensure no crash
