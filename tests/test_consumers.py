from __future__ import annotations

import asyncio

from dart.client.consumers import DrainPacer, JsonNeedPacer, ReadingPacer, ToolCallPacer, TtsPacer, ViewportPacer


async def test_reading_pacer_rate_bounds() -> None:
    p = ReadingPacer(tokens_per_sec=50, burst=10)
    t0 = asyncio.get_running_loop().time()
    got = 0
    while got < 20:
        got += await p.next_window()
    elapsed = asyncio.get_running_loop().time() - t0
    # burst of 10 is instant, remaining 10 at 50 tok/s ≈ 0.2s
    assert elapsed < 1.0


async def test_other_pacers() -> None:
    assert await DrainPacer(4).next_window() == 4
    tts = TtsPacer(realtime_factor=1.0, tokens_per_sec=40, burst=8)
    assert await tts.next_window() >= 1
    j = JsonNeedPacer(burst=7, pause_s=0.0)
    assert await j.next_window() == 7
    view = ViewportPacer(burst=3)
    view.observe(True)
    assert await view.next_window() == 3
    assert await ToolCallPacer(burst=2).next_window() == 2
