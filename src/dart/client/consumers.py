"""Consumer pacers: they are the source of credit, not the GPU."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Pacer(Protocol):
    name: str

    async def next_window(self) -> int:
        """Block until the consumer can absorb more tokens; return W."""
        ...


class DrainPacer:
    """API drainer: credit as fast as the caller pulls (HTTP/WS backpressure)."""

    name = "drain"

    def __init__(self, window: int = 32) -> None:
        self.window = window

    async def next_window(self) -> int:
        return self.window


class ReadingPacer:
    """Human compositor: typical reading speed ~20–40 tok/s."""

    name = "reading"

    def __init__(self, tokens_per_sec: float = 30.0, burst: int = 16) -> None:
        if tokens_per_sec <= 0:
            raise ValueError("tokens_per_sec must be positive")
        self.tokens_per_sec = tokens_per_sec
        self.burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()

    async def next_window(self) -> int:
        while True:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.burst * 2, self._tokens + elapsed * self.tokens_per_sec)
            if self._tokens >= 1:
                n = min(self.burst, int(self._tokens))
                self._tokens -= n
                return max(1, n)
            need = 1.0 - self._tokens
            await asyncio.sleep(need / self.tokens_per_sec)


class TtsPacer:
    """Mouth as credit source. realtime_factor=1.0 is wall-clock speech."""

    name = "tts"

    def __init__(self, realtime_factor: float = 1.0, tokens_per_sec: float = 18.0, burst: int = 12) -> None:
        rate = tokens_per_sec * realtime_factor
        self._inner = ReadingPacer(tokens_per_sec=rate, burst=burst)
        self.realtime_factor = realtime_factor

    async def next_window(self) -> int:
        return await self._inner.next_window()


class JsonNeedPacer:
    """Bursty parser: pull a span, then pause (structured JSON value)."""

    name = "json"

    def __init__(self, burst: int = 64, pause_s: float = 0.5) -> None:
        self.burst = burst
        self.pause_s = pause_s
        self._paused = False

    async def next_window(self) -> int:
        if self._paused:
            await asyncio.sleep(self.pause_s)
            self._paused = False
        self._paused = True
        return self.burst


class ViewportPacer:
    """Browser compositor credit: IntersectionObserver → observe(visible).

    ``next_window`` blocks while the continuation is off-screen. The JS
    companion (``dart.client.pacers``) calls ``observe(true)`` when the
    live page intersects the viewport.
    """

    name = "viewport"

    def __init__(self, burst: int = 16) -> None:
        self.burst = burst
        self._visible = asyncio.Event()

    def observe(self, visible: bool) -> None:
        if visible:
            self._visible.set()
        else:
            self._visible.clear()

    def unobserve(self) -> None:
        self._visible.clear()

    @property
    def visible(self) -> bool:
        return self._visible.is_set()

    async def next_window(self) -> int:
        await self._visible.wait()
        return self.burst


class ToolCallPacer:
    """Tool-call burst: grant W for the call payload, then wait for ack."""

    name = "tool"

    def __init__(self, burst: int = 64, pause_s: float = 0.0) -> None:
        self.burst = burst
        self.pause_s = pause_s
        self._armed = True
        self._ack = asyncio.Event()
        self._ack.set()

    def ack(self) -> None:
        """Consumer finished the tool call; more credit may issue."""
        self._ack.set()
        self._armed = True

    async def next_window(self) -> int:
        if not self._armed:
            if self.pause_s > 0:
                await asyncio.sleep(self.pause_s)
            await self._ack.wait()
        self._ack.clear()
        self._armed = False
        return self.burst
