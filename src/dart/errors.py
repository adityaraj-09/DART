"""Typed errors for the DART continuation runtime."""

from __future__ import annotations


class DartError(Exception):
    """Base class for DART errors."""


class NameParseError(DartError, ValueError):
    """CIP name is not a valid continuation address."""


class LeaseError(DartError):
    """Capability token is missing, expired, or forged."""


class AmplificationError(DartError):
    """Interest asked for more decode than the lease allows."""


class EngineError(DartError):
    """Producer failed to prefill or decode."""


class InterestNack(DartError):
    """Producer refused an Interest without running decode."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


class PinMissError(DartError):
    """No live pin for this kv_root; cannot adopt without re-prefill."""


class HandoverError(DartError):
    """Named KV transfer failed (connector miss or no mesh node)."""
