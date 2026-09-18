from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from dart.andes import run_andes, run_push
from dart.engine.cache_only import CacheOnlyEngine
from dart.engine.fake_http import FakeCounters, create_fake_llamacpp_app, create_fake_vllm_app
from dart.engine.llamacpp import LlamaCppEngine
from dart.engine.stats import stats_of
from dart.engine.synthetic import SyntheticEngine
from dart.engine.vllm import VLLMChatEngine
from dart.experiment import andes_complete, cas_peer_hit, grammar_ablation
from dart.protocol import CipName, Interest
from dart.runtime import DartRuntime
from dart.store import FileCAS
from dart.types import EngineState, Prompt, RuntimeConfig


async def test_vllm_fake_kernel_launches_match_remote() -> None:
    ctr = FakeCounters()
    app = create_fake_vllm_app(seed=1, counters=ctr)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://vllm") as client:
        eng = VLLMChatEngine("fake-7b", base_url="http://vllm/v1", client=client)
        state = EngineState()
        await eng.prefill(Prompt(text="hello named decode"), state)
        assert ctr.kernel_launches == 0  # open() must not POST
        rt = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, decode_quota=24, segment_size=8))
        await rt.start()
        handle = await rt.open("hello named decode", max_tokens=24)
        # still no engine forward until Interest
        remote0 = await eng.scrape_engine_metrics()
        assert remote0["kernel_launches"] == 0
        name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
        await rt.interest(
            Interest(name=name, window=8, lifetime_ms=3000, lease=handle.lease),
            lease=handle.lease,
        )
        remote = await eng.scrape_engine_metrics()
        local = stats_of(eng)
        assert remote["kernel_launches"] == local.kernel_launches == 1
        assert local.kernel_launches == rt.get(handle.cont_id).metrics.decode_kernel_launches
        await rt.aclose()


async def test_vllm_zero_interest_zero_engine_forwards() -> None:
    ctr = FakeCounters()
    app = create_fake_vllm_app(counters=ctr)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://vllm") as client:
        eng = VLLMChatEngine("fake-7b", base_url="http://vllm/v1", client=client)
        rt = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, t_decode_s=0.005, decode_quota=64))
        await rt.start()
        await rt.open("no interest", max_tokens=64)
        await asyncio.sleep(0.04)
        assert ctr.kernel_launches == 0
        assert (await eng.scrape_engine_metrics())["kernel_launches"] == 0
        await rt.aclose()


async def test_llamacpp_fake_completion_is_the_kernel() -> None:
    ctr = FakeCounters()
    app = create_fake_llamacpp_app(counters=ctr)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://llama") as client:
        eng = LlamaCppEngine("llama.cpp", base_url="http://llama", client=client)
        rt = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, decode_quota=16, segment_size=4))
        await rt.start()
        handle = await rt.open("cpp", max_tokens=16)
        name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
        await rt.interest(Interest(name=name, window=4, lifetime_ms=3000, lease=handle.lease), lease=handle.lease)
        remote = await eng.scrape_engine_metrics()
        assert remote["kernel_launches"] == 1
        assert ctr.kernel_launches == 1
        await rt.aclose()


async def test_andes_holds_inventory_idd_does_not() -> None:
    report = await andes_complete(duration_s=0.5, max_tokens=64, watermark=16, consume_tps=30)
    assert report["andes"]["pacer_inventory_hw"] > 0
    assert report["idd"]["pacer_inventory_hw"] == 0
    assert report["idd_beats_andes_on_inventory"] is True
    # Engine counters exist on all three arms
    assert report["push"]["engine_kernel_launches"] >= 1
    assert report["andes"]["engine_kernel_launches"] >= 1
    assert report["idd"]["engine_kernel_launches"] >= 1


async def test_andes_watermark_pauses() -> None:
    row = await run_andes(consume_tps=20, watermark=8, max_tokens=40, duration_s=0.4, segment_size=8)
    assert row["paused_ticks"] >= 1
    assert row["pacer_inventory_hw"] >= 1
    push = await run_push(max_tokens=40)
    assert push["tokens_generated"] >= 16


async def test_grammar_jump_is_one_kernel_mask_is_many() -> None:
    report = await grammar_ablation("tool-args")
    assert report["one_data_object"] is True
    assert report["jump_forward"]["kernel_launches"] == 1
    assert report["logit_masking"]["kernel_launches"] > 1
    assert report["masking_more_launches"] is True


async def test_cas_peer_second_process_no_gpu(tmp_path: Path) -> None:
    report = await cas_peer_hit(str(tmp_path / "cas"))
    assert report["match"] is True
    assert report["peer_engine_launches"] == 0
    assert report["producer_engine_launches"] >= 1

    # True second OS process: read FileCAS without importing the producer runtime.
    name = report["name"]
    cas_dir = str(tmp_path / "cas")
    code = (
        "from dart.store import FileCAS\n"
        f"cas = FileCAS({cas_dir!r})\n"
        f"d = cas.get_data({name!r})\n"
        "assert d is not None and d.text\n"
        "print('peer-ok', d.cache_hit)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert "peer-ok" in proc.stdout


async def test_cache_only_refuses_decode() -> None:
    eng = CacheOnlyEngine()
    with pytest.raises(Exception):
        await eng.decode(EngineState(), 8)


async def test_peer_http_satisfies_without_kernel(tmp_path: Path) -> None:
    from dart.gateway import create_app, create_peer_app

    cas_dir = tmp_path / "cas"
    eng = SyntheticEngine(seed=4)
    rt = DartRuntime(
        eng,
        RuntimeConfig(poll_interval_s=0.001, decode_quota=16),
        cas=FileCAS(cas_dir),
    )
    await rt.start()
    handle = await rt.open("peer http", max_tokens=16)
    name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    data = await rt.interest(
        Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease),
        lease=handle.lease,
    )
    await rt.aclose()

    peer_app = create_peer_app(str(cas_dir))
    transport = ASGITransport(app=peer_app)
    async with AsyncClient(transport=transport, base_url="http://peer") as client:
        r = await client.post("/v1/peer/interest", json={"name": name})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cache_hit"] is True
        assert body["text"] == data.text
        missing = await client.get("/v1/cas", params={"name": "/cip/deadbeef/deadbeef/tokens/seg/0"})
        assert missing.status_code == 404
    _ = create_app  # imported for symmetry
