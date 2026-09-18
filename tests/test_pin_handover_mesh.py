from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from dart.engine.cache_only import CacheOnlyEngine
from dart.engine.synthetic import SyntheticEngine
from dart.errors import HandoverError, InterestNack, PinMissError
from dart.gateway import create_app
from dart.kvconn import (
    FileKVConnector,
    KVBlob,
    LMCacheConnector,
    MemoryKVConnector,
    NixlConnector,
    build_connector,
)
from dart.mesh import InterestRouter, MeshNode, RouteKind, build_local_mesh
from dart.pin import PinnedKVPool
from dart.protocol import CipName, Interest
from dart.runtime import DartRuntime
from dart.store import MemoryCAS
from dart.types import EngineState, Prompt, RuntimeConfig


def _cfg(**kw: object) -> RuntimeConfig:
    base = dict(
        poll_interval_s=0.001,
        decode_quota=64,
        segment_size=8,
        interest_lifetime_s=4.0,
        lease_ttl_s=60.0,
        w_init=16,
        w_max=64,
    )
    base.update(kw)
    return RuntimeConfig(**base)  # type: ignore[arg-type]


def test_pin_by_kv_root_lookup_and_expiry() -> None:
    pool = PinnedKVPool()
    state = EngineState(kv_root="ab" * 32, pos=4)
    pool.pin(
        state.kv_root,
        state,
        holder_id="node-0",
        until=time.time() + 30,
        cont_id="c1",
        model_hash="mh",
    )
    rec = pool.lookup(state.kv_root)
    assert rec is not None
    assert rec.holder_id == "node-0"
    assert rec.clone_state().pos == 4
    assert pool.drop_if_unpinned(state.kv_root, now=time.time() + 1) is False
    expired = EngineState(kv_root="cd" * 32, pos=1)
    until = time.time() + 30
    pool.pin(
        expired.kv_root,
        expired,
        holder_id="node-0",
        until=until,
        cont_id="c2",
        model_hash="mh",
    )
    assert pool.lookup(expired.kv_root, now=until + 10.0) is None
    assert pool.drop_if_unpinned(expired.kv_root, now=until + 10.0) is True


def test_pin_adopt_transfers_holder_and_clones_state() -> None:
    pool = PinnedKVPool()
    state = EngineState(kv_root="ef" * 32, pos=9, output_ids=[1, 2])
    pool.pin(
        state.kv_root,
        state,
        holder_id="node-0",
        until=time.time() + 60,
        cont_id="c",
        model_hash="mh",
    )
    got = pool.adopt(state.kv_root, "node-2")
    assert got.holder_id == "node-2"
    got.state.output_ids.append(99)
    rec = pool.lookup(state.kv_root)
    assert rec is not None
    assert rec.state.output_ids == [1, 2]
    assert pool.adopts == 1


def test_pin_miss_and_sleep_wake() -> None:
    pool = PinnedKVPool()
    with pytest.raises(PinMissError):
        pool.adopt("00" * 32, "node-1")
    state = EngineState(kv_root="11" * 32)
    pool.pin(
        state.kv_root,
        state,
        holder_id="n",
        until=time.time() + 10,
        cont_id="c",
        model_hash="mh",
        on_gpu=True,
    )
    pool.sleep(state.kv_root)
    rec = pool.lookup(state.kv_root)
    assert rec is not None and rec.on_gpu is False
    pool.wake(state.kv_root)
    rec = pool.lookup(state.kv_root)
    assert rec is not None and rec.on_gpu is True


async def test_memory_and_lmcache_and_nixl_connectors() -> None:
    state = EngineState(kv_root="aa" * 32, pos=3)
    blob = KVBlob.from_state(
        kv_root=state.kv_root,
        model_hash="mh",
        cont_id="c",
        state=state,
        holder_id="node-0",
    )
    mem = MemoryKVConnector()
    await mem.put(blob)
    got = await mem.get(state.kv_root)
    assert got is not None and got.state.pos == 3
    moved = await mem.transfer(state.kv_root, src="node-0", dst="node-2")
    assert moved.holder_id == "node-2"
    assert mem.bytes_moved == blob.nbytes
    with pytest.raises(HandoverError):
        await mem.transfer("ff" * 32, src="a", dst="b")

    lmc = LMCacheConnector()
    await lmc.put(blob)
    assert await lmc.has(state.kv_root)
    assert lmc.lmcache_available is False or isinstance(lmc.lmcache_available, bool)

    nixl = NixlConnector(backend=MemoryKVConnector())
    await nixl.put(blob)
    moved2 = await nixl.transfer(state.kv_root, src="node-0", dst="node-1")
    assert moved2.holder_id == "node-1"
    assert nixl.transfers == 1
    assert nixl.rdma_available is False
    m = nixl.metrics()
    assert m["transport"] == "nixl-memcpy"


async def test_file_connector_survives_reload(tmp_path: Path) -> None:
    state = EngineState(kv_root="bb" * 32, pos=7, output_ids=[4])
    blob = KVBlob.from_state(
        kv_root=state.kv_root,
        model_hash="mh",
        cont_id="c",
        state=state,
        holder_id="node-0",
    )
    a = FileKVConnector(tmp_path)
    await a.put(blob)
    b = FileKVConnector(tmp_path)
    got = await b.get(state.kv_root)
    assert got is not None
    assert got.state.output_ids == [4]
    conn = build_connector("file", path=tmp_path)
    assert await conn.has(state.kv_root)


async def test_adopt_skips_prefill() -> None:
    pool = PinnedKVPool()
    conn = MemoryKVConnector()
    cas = MemoryCAS()
    src = DartRuntime(
        SyntheticEngine(seed=1),
        _cfg(producer_id="node-0"),
        cas=cas,
        pin_pool=pool,
        connector=conn,
    )
    dst = DartRuntime(
        SyntheticEngine(seed=1),
        _cfg(producer_id="node-1"),
        cas=cas,
        pin_pool=pool,
        connector=conn,
    )
    await src.start()
    await dst.start()
    handle = await src.open("resume without prefill", max_tokens=32)
    name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    data = await src.interest(Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease)
    src_prefills = src.engine.prefills  # type: ignore[attr-defined]
    await src.release_for_handover(handle.cont_id)
    adopted = await dst.adopt(lease=handle.lease, kv_root=data.kv_root, from_node="node-0")
    assert adopted.cont_id == handle.cont_id
    assert dst.engine.prefills == 0  # type: ignore[attr-defined]
    assert src_prefills == 1
    assert dst.get(handle.cont_id).adopted is True
    assert dst.get(handle.cont_id).metrics.prefills_skipped == 1
    name1 = CipName.tokens(handle.model_hash, data.kv_root, 1).render()
    more = await dst.interest(
        Interest(name=name1, window=8, lifetime_ms=2000, lease=handle.lease),
        lease=handle.lease,
    )
    assert more.token_count() > 0
    assert dst.engine.kernel_launches >= 1  # type: ignore[attr-defined]
    await src.aclose()
    await dst.aclose()


async def test_adopt_token_continuity_matches_single_node() -> None:
    async def _run(handover: bool) -> list[str]:
        pool = PinnedKVPool()
        conn = NixlConnector()
        cas = MemoryCAS()
        a = DartRuntime(SyntheticEngine(seed=4), _cfg(producer_id="a"), cas=cas, pin_pool=pool, connector=conn)
        b = DartRuntime(SyntheticEngine(seed=4), _cfg(producer_id="b"), cas=cas, pin_pool=pool, connector=conn)
        await a.start()
        await b.start()
        handle = await a.open("continuity", max_tokens=48)
        texts: list[str] = []
        kv = handle.kv_root
        rt = a
        for seg in range(3):
            if handover and seg == 2:
                await a.release_for_handover(handle.cont_id)
                await b.adopt(lease=handle.lease, kv_root=kv, from_node="a")
                rt = b
            name = CipName.tokens(handle.model_hash, kv, seg).render()
            data = await rt.interest(
                Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease),
                lease=handle.lease,
            )
            texts.append(data.text)
            kv = data.kv_root
        await a.aclose()
        await b.aclose()
        return texts

    control = await _run(False)
    handed = await _run(True)
    assert handed == control


async def test_router_cas_pin_then_handover_to_cheapest() -> None:
    mesh = build_local_mesh(
        3,
        connector="nixl",
        seed=5,
        costs=[10.0, 8.0, 1.0],
        poll_interval_s=0.001,
        decode_quota=48,
        segment_size=8,
        interest_lifetime_s=4,
    )
    await mesh.start()
    handle = await mesh.open("mesh routing", node_id="node-0", max_tokens=48)
    name0 = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    first, d0 = await mesh.route(
        Interest(name=name0, window=8, lifetime_ms=2500, lease=handle.lease, cont_id=handle.cont_id),
        lease=handle.lease,
    )
    assert d0.kind is RouteKind.PIN
    assert d0.node_id == "node-0"
    assert mesh.nodes[2].runtime.engine.kernel_launches == 0  # type: ignore[attr-defined]

    _, d_cas = await mesh.route(
        Interest(name=name0, window=8, lifetime_ms=2500, lease=handle.lease),
        lease=handle.lease,
    )
    assert d_cas.kind is RouteKind.CAS
    assert d_cas.cache_hit

    name1 = CipName.tokens(handle.model_hash, first.kv_root, 1).render()
    second, d_pin = await mesh.route(
        Interest(name=name1, window=8, lifetime_ms=2500, lease=handle.lease, cont_id=handle.cont_id),
        lease=handle.lease,
    )
    assert d_pin.node_id == "node-0"
    k2 = mesh.nodes[2].runtime.engine.kernel_launches  # type: ignore[attr-defined]
    assert k2 == 0

    gone = mesh.nodes[0].runtime
    mesh.unregister("node-0")
    name2 = CipName.tokens(handle.model_hash, second.kv_root, 2).render()
    third, d_ho = await mesh.route(
        Interest(name=name2, window=8, lifetime_ms=2500, lease=handle.lease, cont_id=handle.cont_id),
        lease=handle.lease,
    )
    assert d_ho.kind is RouteKind.HANDOVER
    assert d_ho.node_id == "node-2"
    assert d_ho.prefill_skipped
    assert mesh.nodes[1].runtime.engine.prefills == 0  # type: ignore[attr-defined]
    assert mesh.nodes[1].runtime.engine.kernel_launches >= 1  # type: ignore[attr-defined]
    assert third.token_count() > 0
    st = mesh.status()
    assert st["stats"]["handover_routes"] >= 1
    await gone.aclose()
    await mesh.aclose()


async def test_model_mismatch_on_adopt() -> None:
    pool = PinnedKVPool()
    conn = MemoryKVConnector()
    src = DartRuntime(
        SyntheticEngine(seed=1),
        _cfg(producer_id="node-0"),
        pin_pool=pool,
        connector=conn,
    )
    dst = DartRuntime(
        CacheOnlyEngine(),
        _cfg(producer_id="node-1"),
        pin_pool=pool,
        connector=conn,
    )
    await src.start()
    handle = await src.open("mismatch", max_tokens=16)
    with pytest.raises(InterestNack) as exc:
        await dst.adopt(lease=handle.lease, kv_root=handle.kv_root)
    assert exc.value.reason == "no_model"
    await src.aclose()
    await dst.aclose()


async def test_concurrent_handover_single_winner() -> None:
    mesh = build_local_mesh(
        3,
        connector="nixl",
        seed=6,
        costs=[9.0, 2.0, 1.0],
        poll_interval_s=0.001,
        decode_quota=32,
        segment_size=8,
    )
    await mesh.start()
    handle = await mesh.open("race", node_id="node-0", max_tokens=32)
    name0 = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    first, _ = await mesh.route(
        Interest(name=name0, window=8, lifetime_ms=2500, lease=handle.lease),
        lease=handle.lease,
    )
    await mesh.nodes[0].runtime.release_for_handover(handle.cont_id)
    name1 = CipName.tokens(handle.model_hash, first.kv_root, 1).render()
    req = Interest(name=name1, window=8, lifetime_ms=2500, lease=handle.lease, cont_id=handle.cont_id)

    results = await asyncio.gather(
        mesh.route(req, lease=handle.lease),
        mesh.route(req, lease=handle.lease),
        return_exceptions=True,
    )
    datas = [r for r in results if not isinstance(r, BaseException)]
    assert len(datas) == 2
    texts = {d[0].text for d in datas}
    assert len(texts) == 1
    prefills = [n.runtime.engine.prefills for n in mesh.nodes]  # type: ignore[attr-defined]
    assert prefills[1] == 0 and prefills[2] == 0
    await mesh.aclose()


async def test_gateway_kv_handover_and_mesh_interest() -> None:
    mesh = build_local_mesh(
        2,
        connector="memory",
        seed=8,
        costs=[1.0, 5.0],
        poll_interval_s=0.001,
        decode_quota=32,
        segment_size=8,
    )
    await mesh.start()
    app = create_app(mesh.nodes[0].runtime, router=mesh)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        opened = await client.post("/v1/continuations", json={"prompt": "gw mesh", "max_tokens": 24})
        assert opened.status_code == 200, opened.text
        body = opened.json()
        interest = await client.post(
            f"/v1/continuations/{body['cont_id']}/interest",
            json={"window": 8, "lease": body["lease"]},
        )
        assert interest.status_code == 200, interest.text
        data = interest.json()
        kv = await client.get("/v1/kv", params={"root": data["kv_root"]})
        assert kv.status_code == 200
        assert kv.json()["kv_root"] == data["kv_root"]
        status = await client.get("/v1/mesh")
        assert status.status_code == 200
        assert len(status.json()["nodes"]) == 2
        replay = await client.post(
            "/v1/mesh/interest",
            json={"name": data["name"], "window": 8, "lease": body["lease"]},
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["cache_hit"] is True
        assert replay.json()["route"]["kind"] == "cas"
        ho = await client.post(
            "/v1/handover",
            json={"lease": body["lease"], "kv_root": data["kv_root"], "from_node": "node-0"},
        )
        assert ho.status_code == 200, ho.text
        assert ho.json()["adopted"] is True
        metrics = await client.get("/v1/metrics")
        assert metrics.json()["pins"]["live"] >= 1
        assert "dart_kv_pins" in (await client.get("/metrics")).text
    await mesh.aclose()


async def test_experiment_mesh_suite_ok() -> None:
    from dart.experiment import mesh_handover_suite

    report = await mesh_handover_suite()
    assert report["ok"] is True
    assert report["cas_route"]["kind"] == "cas"
    assert report["handover_route"]["kind"] == "handover"
    assert report["prefills_after"][2] == 0


async def test_build_local_mesh_requires_node() -> None:
    with pytest.raises(ValueError):
        build_local_mesh(0)


def test_router_cheapest_and_unknown_node() -> None:
    a = DartRuntime(SyntheticEngine(), _cfg(producer_id="node-0"))
    b = DartRuntime(SyntheticEngine(), _cfg(producer_id="node-1"))
    router = InterestRouter(
        [MeshNode("node-0", a, cost=5), MeshNode("node-1", b, cost=1)],
        pin_pool=PinnedKVPool(),
        connector=MemoryKVConnector(),
        cas=MemoryCAS(),
        secret="dart-dev-secret-change-me",
    )
    assert router.cheapest().node_id == "node-1"
    with pytest.raises(HandoverError):
        router.node("nope")


async def test_unknown_kv_nacks() -> None:
    mesh = build_local_mesh(2, connector="memory", seed=0, poll_interval_s=0.001)
    await mesh.start()
    handle = await mesh.open("x", node_id="node-0", max_tokens=8)
    bogus = CipName.tokens(handle.model_hash, "ab" * 32, 0).render()
    with pytest.raises(InterestNack) as exc:
        await mesh.route(Interest(name=bogus, window=4, lifetime_ms=500, lease=handle.lease), lease=handle.lease)
    assert exc.value.reason == "unknown_name"
    await mesh.aclose()


async def test_prefill_not_called_on_zero_interest_still_pins() -> None:
    rt = DartRuntime(SyntheticEngine(seed=2), _cfg())
    await rt.start()
    handle = await rt.open("pin on open", max_tokens=16)
    rec = rt.pin_pool.lookup(handle.kv_root)
    assert rec is not None
    assert rt.engine.prefills == 1  # type: ignore[attr-defined]
    await asyncio.sleep(0.02)
    assert rt.engine.kernel_launches == 0  # type: ignore[attr-defined]
    await rt.aclose()
