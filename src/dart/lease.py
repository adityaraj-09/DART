"""HMAC-signed continuation capabilities.

An Interest is a compute capability. The lease bound to a continuation
caps window, decode quota, and lifetime so a client cannot Interest for
1e6 tokens on someone else's GPU.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dart.errors import AmplificationError, LeaseError


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


class ContinuationLease(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    cont_id: str
    model_hash: str
    model_id: str
    kv_root: str
    pos: int = 0
    w_max: int = 128
    decode_quota: int = 512
    expiry_unix: float
    issued_at: float = Field(default_factory=time.time)
    producer_hint: str = "local-0"
    startup_credit: int = 16

    def remaining_ttl(self, now: float | None = None) -> float:
        return self.expiry_unix - (now if now is not None else time.time())

    def assert_live(self, now: float | None = None) -> None:
        if self.remaining_ttl(now) <= 0:
            raise LeaseError("lease expired")

    def assert_window(self, window: int) -> None:
        if window > self.w_max:
            raise AmplificationError(
                f"Interest window {window} exceeds lease w_max {self.w_max}"
            )


def sign_lease(lease: ContinuationLease, secret: str | bytes) -> str:
    key = secret.encode() if isinstance(secret, str) else secret
    payload = json.dumps(lease.model_dump(), separators=(",", ":"), sort_keys=True).encode()
    sig = hmac.new(key, payload, hashlib.sha256).digest()
    return f"{_b64e(payload)}.{_b64e(sig)}"


def verify_lease(token: str, secret: str | bytes, *, now: float | None = None) -> ContinuationLease:
    key = secret.encode() if isinstance(secret, str) else secret
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError as exc:
        raise LeaseError("malformed lease") from exc
    payload = _b64d(payload_b64)
    expected = hmac.new(key, payload, hashlib.sha256).digest()
    actual = _b64d(sig_b64)
    if not hmac.compare_digest(expected, actual):
        raise LeaseError("forged lease")
    try:
        data: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LeaseError("malformed lease payload") from exc
    lease = ContinuationLease.model_validate(data)
    lease.assert_live(now)
    return lease
