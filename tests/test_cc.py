from __future__ import annotations

from dart.cc import CongestionController


def test_grant_and_ack_slow_start() -> None:
    cc = CongestionController(w_init=8, w_max=64, ssthresh=32.0)
    g = cc.on_interest(8, lease_w_max=64, now=1.0)
    assert g == 8
    committed, extra = cc.take_decode(8)
    assert committed == 8
    assert cc.credits == 0
    cc.on_ack(8, now=1.1)
    assert cc.cwnd >= 16  # slow start adds consumed


def test_cannot_amplify_past_cwnd() -> None:
    cc = CongestionController(w_init=4, w_max=8)
    assert cc.on_interest(100, lease_w_max=128) == 4
    assert cc.on_interest(100, lease_w_max=128) == 0


def test_timeout_halves_and_drops_credit() -> None:
    cc = CongestionController(w_init=16, w_max=64)
    cc.on_interest(16, lease_w_max=64, now=0.0)
    leftover = cc.on_timeout(now=3.0)
    assert leftover == 16
    assert cc.credits == 0
    assert cc.cwnd == 8
    assert cc.ssthresh == 8


def test_speculative_k_grows_with_rtt() -> None:
    slow = CongestionController(w_init=16, k_max=8, t_decode_s=0.02, rtt_s=0.02)
    fast_rtt = CongestionController(w_init=16, k_max=8, t_decode_s=0.02, rtt_s=0.2)
    assert slow.speculative_k() <= fast_rtt.speculative_k()
    assert 0 <= fast_rtt.speculative_k() <= 8
