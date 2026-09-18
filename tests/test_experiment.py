from __future__ import annotations

from dart.experiment import compare, run_workload
from dart.sdk import DartClient, DrainPacer


async def test_push_generates_more_than_slow_reader() -> None:
    report = await compare(duration_s=0.6, max_tokens=128, step_latency_s=0.0)
    push = report["push"]
    reading = report["credit_reading_30tps"]
    assert reading["tokens_consumed"] > 0
    assert push["tokens_generated"] >= reading["tokens_generated"]
    assert report["kill_test"]["survive"] is True
    assert reading["decode_kernel_launches"] >= 1


async def test_push_workload_fills_quota() -> None:
    row = await run_workload("push", max_tokens=32, duration_s=1.0)
    assert row["tokens_generated"] >= 16
    assert row["engine_kernel_launches"] >= 1


async def test_sdk_inprocess_stream() -> None:
    from dart.engine.synthetic import SyntheticEngine
    from dart.runtime import DartRuntime
    from dart.types import RuntimeConfig

    rt = DartRuntime(SyntheticEngine(seed=9), RuntimeConfig(poll_interval_s=0.001, decode_quota=32))
    await rt.start()
    client = DartClient(runtime=rt)
    texts: list[str] = []
    async for seg in client.stream("sdk path", consumer=DrainPacer(8), max_tokens=16):
        texts.append(seg.text)
    await rt.aclose()
    assert "".join(texts)
