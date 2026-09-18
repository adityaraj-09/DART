from __future__ import annotations

import time

import pytest

from dart.errors import AmplificationError, LeaseError
from dart.lease import ContinuationLease, sign_lease, verify_lease


def _lease(**kw: object) -> ContinuationLease:
    now = time.time()
    base = dict(
        cont_id="c1",
        model_hash="ab" * 8,
        model_id="m",
        kv_root="cd" * 16,
        expiry_unix=now + 60,
        issued_at=now,
        w_max=32,
        decode_quota=100,
    )
    base.update(kw)
    return ContinuationLease.model_validate(base)


def test_sign_verify_roundtrip() -> None:
    lease = _lease()
    token = sign_lease(lease, "secret")
    got = verify_lease(token, "secret")
    assert got.cont_id == "c1"


def test_forged_and_wrong_secret() -> None:
    token = sign_lease(_lease(), "secret")
    with pytest.raises(LeaseError):
        verify_lease(token, "other")
    with pytest.raises(LeaseError):
        verify_lease(token[:-2] + "aa", "secret")
    with pytest.raises(LeaseError):
        verify_lease("not-a-token", "secret")


def test_expired() -> None:
    lease = _lease(expiry_unix=time.time() - 1)
    token = sign_lease(lease, "secret")
    with pytest.raises(LeaseError, match="expired"):
        verify_lease(token, "secret")


def test_window_cap() -> None:
    lease = _lease(w_max=16)
    lease.assert_window(16)
    with pytest.raises(AmplificationError):
        lease.assert_window(17)
