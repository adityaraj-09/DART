"""Optional upstream vLLM hook.

vLLM loads ``vllm.general_plugins`` entry points. We register a no-crash
callback that exposes DART's credit-gated waiting scheduler. We do **not**
replace vLLM's Scheduler class unless the import surface matches; a mismatch
would take down someone else's vLLM process.

In-process serving uses ``InProcessVLLMEngine`` (``--engine vllm-inprocess``).
HTTP ``--engine vllm`` is unchanged.
"""

from __future__ import annotations

from typing import Any


def vllm_installed() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def dart_scheduler_class() -> type | None:
    """Return a vLLM Scheduler subclass when the V1 API is present."""
    try:
        from vllm.v1.core.sched.scheduler import Scheduler  # type: ignore
    except ImportError:
        return None

    class DartCreditScheduler(Scheduler):  # type: ignore[misc,valid-type]
        """Leave zero-credit requests in waiting; do not free their KV."""

        dart_credit: dict[str, int] = {}

        def schedule(self) -> Any:  # pragma: no cover - needs real vLLM
            waiting = getattr(self, "waiting", None)
            running = getattr(self, "running", None)
            if running is not None and waiting is not None:
                keep = []
                park = []
                for req in list(running):
                    rid = getattr(req, "request_id", None) or getattr(req, "req_id", "")
                    credit = self.dart_credit.get(str(rid), 1)
                    if credit <= 0:
                        park.append(req)
                    else:
                        keep.append(req)
                if park:
                    running.clear()
                    running.extend(keep)
                    for req in park:
                        if hasattr(req, "status"):
                            try:
                                from vllm.v1.request import RequestStatus

                                req.status = RequestStatus.WAITING
                            except Exception:
                                pass
                        waiting.append(req)
            return super().schedule()

    return DartCreditScheduler


def register() -> dict[str, Any]:
    """vLLM general-plugin entry point. Safe when vLLM is absent."""
    installed = vllm_installed()
    cls = dart_scheduler_class() if installed else None
    return {
        "name": "dart_idd",
        "vllm_installed": installed,
        "scheduler": None if cls is None else f"{cls.__module__}.{cls.__name__}",
        "engine": "dart.engine.vllm_inprocess.InProcessVLLMEngine",
    }
