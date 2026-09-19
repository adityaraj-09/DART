from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from dart.engine.fake_http import FakeCounters, create_fake_vllm_app
from dart.engine.vllm import VLLMChatEngine
from dart.engine.vllm_inprocess import InProcessVLLMEngine
from dart.engine.vllm_plugin import register, vllm_installed
from dart.engine.vllm_sched import CreditGatedScheduler, RequestStatus
from dart.cip.protocol import CipName, Interest
from dart.core.runtime import DartRuntime
from dart.core.types import EngineState, Prompt, RuntimeConfig


def test_scheduler_waiting_pin_not_evicted() -> None:
    sched = CreditGatedScheduler(num_gpu_blocks=32, block_nbytes=64)
    live = sched.admit("live", n_tokens=16, n_blocks=4)
    sched.pause("live", mode="keep")
    assert live.status is RequestStatus.WAITING
    assert live.pinned is True
    sched.admit("cached", n_tokens=8, n_blocks=4, count_prefill=False)
    sched.blocks.unpin("cached")
    sched.blocks.evict_unpinned(8)
    assert sched.blocks.has_blocks("live")
    assert not sched.blocks.has_blocks("cached")
    sched.admit("live", n_tokens=16, n_blocks=4)
    assert sched.admission_reentries == 1


def test_scheduler_abort_frees() -> None:
    sched = CreditGatedScheduler(num_gpu_blocks=16, block_nbytes=8)
    sched.admit("r", n_tokens=4, n_blocks=2)
    assert sched.blocks.pinned_blocks() == 2
    sched.abort("r")
    assert sched.blocks.live_blocks() == 0
    req = sched.get("r")
    assert req is not None and req.status is RequestStatus.FINISHED


def test_plugin_register_is_safe_without_vllm() -> None:
    info = register()
    assert info["name"] == "dart_idd"
    assert info["vllm_installed"] is vllm_installed()
    assert "InProcessVLLMEngine" in info["engine"]


async def test_inprocess_open_waits_with_pinned_blocks() -> None:
    eng = InProcessVLLMEngine(seed=1)
    rt = DartRuntime(
        eng,
        RuntimeConfig(poll_interval_s=0.001, t_decode_s=0.005, decode_quota=32),
    )
    await rt.start()
    handle = await rt.open("keep the table", max_tokens=32)
    await asyncio.sleep(0.04)
    snap = eng.scheduler_snapshot()
    cont = rt.get(handle.cont_id)
    assert snap["admissions"] == 1
    assert snap["decode_forwards"] == 0
    assert snap["waiting"] >= 1
    assert snap["pinned_blocks"] > 0
    assert eng.request_status(cont.state) == "waiting"
    assert cont.sleeping is True
    assert rt.kv.gpu_bytes() > 0  # pause(keep): still on GPU
    assert eng.kernel_launches == 1  # prefill forward only
    await rt.aclose()


async def test_two_interests_one_admission() -> None:
    eng = InProcessVLLMEngine(seed=3)
    rt = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, decode_quota=32, segment_size=8))
    await rt.start()
    handle = await rt.open("no reentry", max_tokens=32)
    name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    d0 = await rt.interest(
        Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
    )
    name1 = CipName.tokens(handle.model_hash, d0.kv_root, 1).render()
    await rt.interest(
        Interest(name=name1, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
    )
    snap = eng.scheduler_snapshot()
    assert snap["admissions"] == 1
    assert snap["admission_reentries"] == 0
    assert snap["decode_forwards"] == 2
    assert rt.get(handle.cont_id).metrics.decode_kernel_launches == 2
    await rt.aclose()


async def test_http_vllm_reenters_each_interest() -> None:
    ctr = FakeCounters()
    app = create_fake_vllm_app(counters=ctr)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://vllm") as client:
        eng = VLLMChatEngine("fake-7b", base_url="http://vllm/v1", client=client)
        rt = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, decode_quota=24, segment_size=8))
        await rt.start()
        handle = await rt.open("http path", max_tokens=24)
        assert ctr.kernel_launches == 0
        name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
        d0 = await rt.interest(
            Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
        )
        name1 = CipName.tokens(handle.model_hash, d0.kv_root, 1).render()
        await rt.interest(
            Interest(name=name1, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
        )
        assert ctr.kernel_launches == 2
        await rt.aclose()


async def test_pressure_evicts_cache_not_waiting_pin() -> None:
    eng = InProcessVLLMEngine(seed=4, num_gpu_blocks=64)
    rt = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, decode_quota=16))
    await rt.start()
    handle = await rt.open("pinned victim contrast", max_tokens=16)
    rid = rt.get(handle.cont_id).state.engine_request_id
    eng.sched.admit("cold", n_tokens=8, n_blocks=8, count_prefill=False)
    eng.sched.blocks.unpin("cold")
    freed = eng.apply_memory_pressure(8)
    assert freed >= 1
    assert eng.sched.blocks.has_blocks(rid)
    assert not eng.sched.blocks.has_blocks("cold")
    await rt.aclose()


async def test_decode_zero_is_pause_not_kernel() -> None:
    eng = InProcessVLLMEngine(seed=5)
    pre = await eng.prefill(Prompt(text="x"), EngineState())
    launches = eng.kernel_launches
    out = await eng.decode(pre.state, 0)
    assert out.kernel_launched is False
    assert eng.kernel_launches == launches
    assert eng.request_status(pre.state) == "waiting"


async def test_experiment_waiting_suite_ok() -> None:
    from dart.eval.experiment import waiting_plugin_suite

    report = await waiting_plugin_suite()
    assert report["ok"] is True
    assert report["http_generate_calls"] == 2
    assert report["after_two_interests"]["admissions"] == 1


def test_factory_vllm_inprocess() -> None:
    from dart.factory import build_engine

    eng = build_engine("vllm-inprocess")
    assert isinstance(eng, InProcessVLLMEngine)


def test_factory_vllm_defaults_to_inprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    from dart.factory import build_engine

    monkeypatch.delenv("DART_VLLM_URL", raising=False)
    eng = build_engine("vllm")
    assert isinstance(eng, InProcessVLLMEngine)


def test_default_kernel_is_synthetic_for_paper_model() -> None:
    eng = InProcessVLLMEngine()
    assert type(eng._kernel).__name__ == "SyntheticEngine"
    assert eng.backend == "inprocess"


def test_use_live_runner_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    from dart.engine.vllm_runner import use_live_vllm_runner

    monkeypatch.delenv("DART_VLLM_INPROCESS", raising=False)
    assert use_live_vllm_runner("dart-synth-8b") is False
    monkeypatch.setenv("DART_VLLM_INPROCESS", "0")
    assert use_live_vllm_runner("meta-llama/Llama-3.1-8B-Instruct") is False


class _FakeVLLMEngine:
    def __init__(self) -> None:
        self.adds = 0
        self.steps = 0
        self.aborted: list[str] = []
        self._reqs: dict[str, dict[str, object]] = {}

    def add_request(self, rid: str, prompt: object, params: object) -> None:
        self.adds += 1
        cap = int(getattr(params, "max_tokens", 32) or 32)
        self._reqs[rid] = {"ids": [], "text": "", "finished": False, "max": cap, "prompt": prompt}

    def step(self) -> list[object]:
        from types import SimpleNamespace

        self.steps += 1
        out: list[object] = []
        for rid, rec in self._reqs.items():
            if rec["finished"]:
                continue
            rec["ids"].append(200 + len(rec["ids"]))  # type: ignore[union-attr]
            rec["text"] = str(rec["text"]) + "tok "
            if len(rec["ids"]) >= int(rec["max"]):  # type: ignore[arg-type]
                rec["finished"] = True
            out.append(
                SimpleNamespace(
                    request_id=rid,
                    finished=rec["finished"],
                    outputs=[SimpleNamespace(token_ids=list(rec["ids"]), text=rec["text"])],  # type: ignore[arg-type]
                )
            )
        return out

    def has_unfinished_requests(self) -> bool:
        return any(not r["finished"] for r in self._reqs.values())

    def abort_request(self, rid: str) -> None:
        self.aborted.append(rid)
        if rid in self._reqs:
            self._reqs[rid]["finished"] = True


async def test_live_vllm_kernel_keeps_unfinished_request() -> None:
    from dart.engine.vllm_runner import VLLMModelRunner

    fake = _FakeVLLMEngine()
    kernel = VLLMModelRunner("meta-llama/unit", engine=fake)
    eng = InProcessVLLMEngine(model_id="meta-llama/unit", kernel=kernel)
    rt = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, decode_quota=32, segment_size=8))
    await rt.start()
    handle = await rt.open("live pages", max_tokens=32)
    assert fake.adds == 1
    prefill_steps = fake.steps
    await asyncio.sleep(0.03)
    assert fake.steps == prefill_steps  # W=0: do not step the live request
    assert not fake.aborted
    assert eng.scheduler_snapshot()["admissions"] == 1
    assert eng.backend == "vllm-gpu"
    name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    d0 = await rt.interest(
        Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
    )
    assert d0.token_count() > 0
    after_page = fake.steps
    await asyncio.sleep(0.03)
    assert fake.steps == after_page
    name1 = CipName.tokens(handle.model_hash, d0.kv_root, 1).render()
    await rt.interest(
        Interest(name=name1, window=8, lifetime_ms=2000, lease=handle.lease), lease=handle.lease
    )
    assert fake.adds == 1  # still one vLLM request
    assert eng.scheduler_snapshot()["admissions"] == 1
    await rt.close(handle.cont_id)
    assert fake.aborted
    await rt.aclose()


def test_factory_vllm_http_when_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from dart.engine.vllm import VLLMChatEngine
    from dart.factory import build_engine

    monkeypatch.setenv("DART_VLLM_URL", "http://127.0.0.1:8000/v1")
    eng = build_engine("vllm")
    assert isinstance(eng, VLLMChatEngine)
    assert isinstance(build_engine("vllm-http"), VLLMChatEngine)
