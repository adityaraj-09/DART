"""CIP congestion control: speculative K, window, and timeout.

Transport signals map onto inference knobs:

    outstanding Interests / RTT  → speculative K
    InterestLifetime expiry      → stop decode, pin KV
    zero Interests               → GPU does zero decode
    ACK / next Interest          → grow cwnd (slow-start then AIMD)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CongestionController:
    w_init: int = 16
    w_max: int = 128
    k_max: int = 8
    t_decode_s: float = 0.02
    rtt_s: float = 0.05
    ssthresh: float = 64.0

    cwnd: float = field(init=False)
    credits: int = 0
    in_flight: int = 0
    last_interest_at: float = 0.0
    last_ack_at: float = 0.0
    timeouts: int = 0
    interests: int = 0

    def __post_init__(self) -> None:
        self.cwnd = float(self.w_init)

    def clamp_window(self, requested: int, lease_w_max: int) -> int:
        return max(0, min(requested, int(self.cwnd), lease_w_max, self.w_max))

    def on_interest(
        self,
        window: int,
        *,
        lease_w_max: int,
        now: float | None = None,
        rtt_sample_s: float | None = None,
    ) -> int:
        """Grant credits. Duplicate in-flight credit is not added twice."""
        now = time.monotonic() if now is None else now
        if rtt_sample_s is not None and rtt_sample_s > 0:
            self.rtt_s = 0.8 * self.rtt_s + 0.2 * rtt_sample_s
        self.last_interest_at = now
        self.interests += 1
        room = max(0, min(int(self.cwnd), lease_w_max, self.w_max) - self.in_flight - self.credits)
        granted = min(max(0, window), room)
        self.credits += granted
        return granted

    def take_decode(self, n: int) -> tuple[int, int]:
        """Return (committed, speculative) tokens the kernel may produce."""
        committed = min(n, self.credits)
        self.credits -= committed
        self.in_flight += committed
        extra = min(self.speculative_k(), max(0, n - committed))
        return committed, extra

    def on_produced(self, committed: int, drafted: int = 0) -> None:
        # in_flight already counted in take_decode for committed
        del drafted

    def on_ack(self, consumed: int, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.last_ack_at = now
        self.in_flight = max(0, self.in_flight - consumed)
        if consumed <= 0:
            return
        if self.cwnd < self.ssthresh:
            self.cwnd = min(self.w_max, self.cwnd + consumed)
        else:
            self.cwnd = min(self.w_max, self.cwnd + consumed / max(self.cwnd, 1.0))

    def on_timeout(self, now: float | None = None) -> int:
        """Halve the window. Returns credits that must not be decoded."""
        now = time.monotonic() if now is None else now
        self.timeouts += 1
        self.ssthresh = max(2.0, self.cwnd / 2.0)
        self.cwnd = max(1.0, self.cwnd / 2.0)
        leftover = self.credits
        self.credits = 0
        return leftover

    def speculative_k(self) -> int:
        """Draft ahead of credit to hide RTT, like TCP prefetch.

        K = clamp(0, ceil(W * RTT / t_decode) - W, K_max) using cwnd as W.
        """
        if self.t_decode_s <= 0:
            return 0
        w = max(1.0, self.cwnd)
        ahead = int(round(w * self.rtt_s / self.t_decode_s - w))
        return int(max(0, min(self.k_max, ahead)))

    def expired(self, lifetime_s: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.last_interest_at <= 0:
            return False
        return (now - self.last_interest_at) > lifetime_s

    @property
    def outstanding(self) -> int:
        return self.credits + self.in_flight
