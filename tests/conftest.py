from __future__ import annotations

import pytest

from dart.engine.synthetic import SyntheticEngine
from dart.core.runtime import DartRuntime
from dart.core.types import RuntimeConfig


@pytest.fixture
async def runtime() -> DartRuntime:
    rt = DartRuntime(
        SyntheticEngine(seed=7),
        RuntimeConfig(
            poll_interval_s=0.001,
            t_decode_s=0.01,
            interest_lifetime_s=2.0,
            w_init=16,
            w_max=64,
            decode_quota=128,
            startup_credit=16,
            lease_ttl_s=60.0,
        ),
    )
    await rt.start()
    yield rt
    await rt.aclose()
