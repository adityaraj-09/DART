from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from dart.client.consumers import ToolCallPacer, ViewportPacer
from dart.core.errors import LeaseError, TenantQuotaError
from dart.core.lease import sign_lease, verify_lease
from dart.cip.protocol import CipName, Interest
from dart.core.runtime import DartRuntime
from dart.engine.synthetic import SyntheticEngine
from dart.kv.pin import FilePinnedKVPool
from dart.core.types import EngineState, RuntimeConfig


def _cfg(**kw: object) -> RuntimeConfig:
    base = dict(
        poll_interval_s=0.001,
        decode_quota=32,
        segment_size=8,
        interest_lifetime_s=4.0,
        lease_ttl_s=60.0,
        secret="current-secret",
        secret_previous="old-secret",
        tenant_interest_quota=2,
        default_tenant="acme",
    )
    base.update(kw)
    return RuntimeConfig(**base)  # type: ignore[arg-type]


def test_verify_lease_accepts_previous_secret() -> None:
    from dart.core.lease import ContinuationLease

    now = time.time()
    lease = ContinuationLease(
        cont_id="c1",
        model_hash="ab" * 8,
        model_id="m",
        kv_root="cd" * 16,
        expiry_unix=now + 60,
        issued_at=now,
        tenant_id="acme",
    )
    token = sign_lease(lease, "old-secret")
    got = verify_lease(token, ["current-secret", "old-secret"])
    assert got.tenant_id == "acme"
    with pytest.raises(LeaseError):
        verify_lease(token, ["current-secret"])


async def test_rotate_secret_and_tenant_quota() -> None:
    rt = DartRuntime(SyntheticEngine(seed=2), _cfg())
    await rt.start()
    handle = await rt.open("quota", max_tokens=32, tenant_id="acme")
    assert rt.get(handle.cont_id).lease.tenant_id == "acme"
    name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    d0 = await rt.interest(
        Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
    )
    name1 = CipName.tokens(handle.model_hash, d0.kv_root, 1).render()
    await rt.interest(
        Interest(name=name1, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
    )
    assert rt.metrics_snapshot()["tenants"]["acme"] == 2
    name2 = CipName.tokens(handle.model_hash, d0.kv_root, 2).render()
    with pytest.raises(TenantQuotaError):
        await rt.interest(
            Interest(name=name2, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
        )
    rt.rotate_secret("rotated")
    still = verify_lease(handle.lease, rt.config.secrets())
    assert still.cont_id == handle.cont_id
    await rt.close(handle.cont_id)
    cont = rt.get(handle.cont_id)
    assert cont.cc.credits == 0
    assert cont.cc.in_flight == 0
    await rt.aclose()


async def test_file_pins_survive_reload_and_adopt(tmp_path: Path) -> None:
    pool_a = FilePinnedKVPool(tmp_path)
    src = DartRuntime(
        SyntheticEngine(seed=1),
        _cfg(producer_id="node-0", tenant_interest_quota=0, secret="pin-secret", secret_previous=None),
        pin_pool=pool_a,
    )
    await src.start()
    handle = await src.open("persist pin", max_tokens=16)
    rec = pool_a.lookup(handle.kv_root)
    assert rec is not None
    await src.aclose()

    pool_b = FilePinnedKVPool(tmp_path)
    assert pool_b.lookup(handle.kv_root) is not None
    dst = DartRuntime(
        SyntheticEngine(seed=1),
        _cfg(producer_id="node-1", tenant_interest_quota=0, secret="pin-secret", secret_previous=None),
        pin_pool=pool_b,
    )
    await dst.start()
    adopted = await dst.adopt(lease=handle.lease, kv_root=handle.kv_root, from_node="node-0")
    assert adopted.cont_id == handle.cont_id
    assert dst.engine.prefills == 0  # type: ignore[attr-defined]
    await dst.aclose()


async def test_viewport_and_tool_pacers() -> None:
    view = ViewportPacer(burst=9)
    assert view.visible is False
    task = asyncio.create_task(view.next_window())
    await asyncio.sleep(0.01)
    assert not task.done()
    view.observe(True)
    assert await task == 9
    view.unobserve()

    tool = ToolCallPacer(burst=5)
    assert await tool.next_window() == 5
    pending = asyncio.create_task(tool.next_window())
    await asyncio.sleep(0.01)
    assert not pending.done()
    tool.ack()
    assert await pending == 5


def test_file_cas_and_pins_from_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dart.factory import build_runtime
    from dart.kv.pin import FilePinnedKVPool
    from dart.kv.store import FileCAS

    monkeypatch.delenv("DART_VLLM_URL", raising=False)
    rt = build_runtime("synthetic", cas_dir=str(tmp_path), pin_dir=str(tmp_path / "pins"))
    assert isinstance(rt.cas, FileCAS)
    assert isinstance(rt.pin_pool, FilePinnedKVPool)
    assert rt.config.pin_dir == str(tmp_path / "pins")
