"""Paper eval: kill-test, Andes-complete, engine kernel probe, CAS peer, grammar."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from dart.andes import run_andes, run_push
from dart.consumers import DrainPacer, JsonNeedPacer, ReadingPacer
from dart.engine.grammar import JumpForwardEngine, LogitMaskedEngine, jump_span_launches, masked_span_launches
from dart.engine.stats import stats_of
from dart.engine.synthetic import SyntheticEngine
from dart.protocol import CipName, Interest
from dart.runtime import DartRuntime
from dart.store import FileCAS
from dart.types import RuntimeConfig


Mode = Literal["push", "reading", "api", "json"]


def _cfg(mode: str, max_tokens: int, duration_s: float, step_latency_s: float) -> RuntimeConfig:
    return RuntimeConfig(
        decode_quota=max_tokens,
        w_init=16 if mode != "push" else 128,
        w_max=128 if mode != "push" else 512,
        startup_credit=16,
        poll_interval_s=0.002,
        t_decode_s=max(step_latency_s, 0.01),
        lease_ttl_s=max(60.0, duration_s + 10),
        interest_lifetime_s=5.0,
    )


async def run_idd(
    mode: Mode,
    *,
    max_tokens: int = 512,
    duration_s: float = 2.0,
    step_latency_s: float = 0.0,
    seed: int = 0,
    engine: SyntheticEngine | None = None,
) -> dict[str, Any]:
    eng = engine or SyntheticEngine(step_latency_s=step_latency_s, seed=seed)
    rt = DartRuntime(eng, _cfg(mode, max_tokens, duration_s, step_latency_s))
    await rt.start()
    t0 = time.monotonic()
    handle = await rt.open("Write a long essay about named continuations.", max_tokens=max_tokens)
    if mode == "push":
        pacer = DrainPacer(window=min(32, max_tokens))
    elif mode == "api":
        pacer = ReadingPacer(tokens_per_sec=200.0, burst=32)
    elif mode == "json":
        pacer = JsonNeedPacer(burst=64, pause_s=0.05)
    else:
        pacer = ReadingPacer(tokens_per_sec=30.0, burst=16)

    consumed = 0

    async def _consume() -> None:
        nonlocal consumed
        async for data in rt.consume(handle, pacer, max_tokens=max_tokens):
            consumed += data.token_count()
            if time.monotonic() - t0 >= duration_s:
                break

    try:
        await asyncio.wait_for(_consume(), timeout=duration_s + 2.0)
    except TimeoutError:
        pass
    await asyncio.sleep(0)
    snap = rt.metrics_snapshot()
    probe = await rt.engine_probe()
    await rt.close(handle.cont_id)
    await rt.aclose()
    totals = snap["totals"]
    generated = totals["tokens_generated"]
    st = stats_of(eng)
    return {
        "mode": f"idd-{mode}",
        "duration_s": time.monotonic() - t0,
        "tokens_generated": generated,
        "tokens_consumed": consumed,
        "tokens_displayed": consumed,
        "tokens_generated_unconsumed": max(0, generated - consumed),
        "pacer_inventory": 0,
        "pacer_inventory_hw": 0,
        "decode_kernel_launches": totals["decode_kernel_launches"],
        "decode_steps_skipped": totals["decode_steps_skipped"],
        "engine_kernel_launches": st.kernel_launches,
        "engine_tokens_predicted": st.tokens_predicted,
        "engine_probe": probe,
        "kv_bytes": snap["kv_bytes"],
        "kv_high_water": snap["kv_high_water"],
        "kv_gpu_bytes": snap["kv_gpu_bytes"],
        "generated_over_consumed": (
            generated / consumed if consumed else (float("inf") if generated else 0.0)
        ),
    }


# Back-compat names used by tests/cli.
run_workload = run_idd


async def compare(
    *,
    duration_s: float = 2.0,
    max_tokens: int = 256,
    step_latency_s: float = 0.0,
) -> dict[str, Any]:
    push = await run_idd("push", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s)
    reading = await run_idd(
        "reading", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s
    )
    api = await run_idd("api", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s)
    bursty = await run_idd(
        "json", duration_s=duration_s, max_tokens=max_tokens, step_latency_s=step_latency_s
    )

    def _cut(metric: str) -> float | None:
        a, b = push[metric], reading[metric]
        if not a:
            return None
        return (a - b) / a

    return {
        "push": push,
        "credit_reading_30tps": reading,
        "credit_api_200tps": api,
        "credit_json_bursty": bursty,
        "kill_test": {
            "unconsumed_cut_vs_push": _cut("tokens_generated_unconsumed")
            if push["tokens_generated_unconsumed"]
            else 1.0 if reading["tokens_generated_unconsumed"] == 0 else _cut("tokens_generated"),
            "kv_high_water_cut_vs_push": _cut("kv_high_water"),
            "generated_cut_vs_push": _cut("tokens_generated"),
            "engine_kernel_cut_vs_push": _cut("engine_kernel_launches"),
            "survive": (
                (reading["tokens_generated"] < push["tokens_generated"] * 0.9)
                or (reading["kv_high_water"] < push["kv_high_water"])
            )
            and reading["tokens_consumed"] > 0,
        },
    }


async def andes_complete(
    *,
    duration_s: float = 1.0,
    max_tokens: int = 96,
    consume_tps: float = 30.0,
    watermark: int = 32,
    step_latency_s: float = 0.0,
) -> dict[str, Any]:
    """Kill criterion 1: watermark pause vs refuse-to-decode."""

    push = await run_push(max_tokens=max_tokens, step_latency_s=step_latency_s)
    andes = await run_andes(
        consume_tps=consume_tps,
        watermark=watermark,
        max_tokens=max_tokens,
        duration_s=duration_s,
        step_latency_s=step_latency_s,
    )
    idd = await run_idd(
        "reading",
        max_tokens=max_tokens,
        duration_s=duration_s,
        step_latency_s=step_latency_s,
    )

    def _frac(base: int, other: int) -> float | None:
        if not base:
            return None
        return (base - other) / base

    andes_capture_of_idd = None
    push_gen = push["tokens_generated"]
    if push_gen and idd["tokens_generated"] < push_gen:
        idd_win = push_gen - idd["tokens_generated"]
        andes_win = push_gen - andes["tokens_generated"]
        andes_capture_of_idd = andes_win / idd_win if idd_win else None

    andes_complete_kill = (
        andes_capture_of_idd is not None and andes_capture_of_idd >= 0.90
        and andes["pacer_inventory_hw"] <= max(1, watermark)
    )
    return {
        "push": push,
        "andes": andes,
        "idd": idd,
        "generated_cut_andes_vs_push": _frac(push["tokens_generated"], andes["tokens_generated"]),
        "generated_cut_idd_vs_push": _frac(push["tokens_generated"], idd["tokens_generated"]),
        "generated_cut_idd_vs_andes": _frac(andes["tokens_generated"], idd["tokens_generated"]),
        "inventory_andes": andes["pacer_inventory_hw"],
        "inventory_idd": idd["pacer_inventory_hw"],
        "andes_captures_idd_win": andes_capture_of_idd,
        "andes_complete_kill": andes_complete_kill,
        "idd_beats_andes_on_inventory": idd["pacer_inventory_hw"] < andes["pacer_inventory_hw"]
        or idd["tokens_generated"] < andes["tokens_generated"],
    }


async def grammar_ablation(span: str = "tool-args") -> dict[str, Any]:
    masked = LogitMaskedEngine(seed=1)
    jump = JumpForwardEngine(seed=1)
    m = await masked_span_launches(masked, "emit json", span)
    j = await jump_span_launches(jump, "emit json", span)
    return {
        "span": span,
        "logit_masking": m,
        "jump_forward": j,
        "kernel_ratio_mask_over_jump": (
            m["kernel_launches"] / j["kernel_launches"] if j["kernel_launches"] else None
        ),
        "one_data_object": j["kernel_launches"] == 1,
        "masking_more_launches": m["kernel_launches"] > j["kernel_launches"],
    }


async def cas_peer_hit(tmp_cas: str) -> dict[str, Any]:
    """Process A decodes into FileCAS; process B (CacheOnly) satisfies without a GPU."""
    from dart.engine.cache_only import CacheOnlyEngine

    cas_a = FileCAS(tmp_cas)
    eng = SyntheticEngine(seed=3)
    rt_a = DartRuntime(eng, RuntimeConfig(poll_interval_s=0.001, decode_quota=32, cas_dir=None), cas=cas_a)
    await rt_a.start()
    handle = await rt_a.open("named peer object", max_tokens=16)
    name = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
    data = await rt_a.interest(
        Interest(name=name, window=8, lifetime_ms=2000, lease=handle.lease),
        lease=handle.lease,
    )
    launches_a = eng.kernel_launches
    await rt_a.aclose()

    cas_b = FileCAS(tmp_cas)  # second process: reload from disk
    peer = DartRuntime(CacheOnlyEngine(), RuntimeConfig(poll_interval_s=0.05), cas=cas_b)
    hit = await peer.satisfy_named(name)
    probe = await peer.engine_probe()
    await peer.aclose()
    return {
        "name": name,
        "producer_text": data.text,
        "peer_text": hit.text,
        "peer_cache_hit": hit.cache_hit,
        "producer_engine_launches": launches_a,
        "peer_engine_launches": stats_of(peer.engine).kernel_launches,
        "peer_gpu": False,
        "match": hit.text == data.text and hit.cache_hit is True and stats_of(peer.engine).kernel_launches == 0,
    }


async def paper_suite(
    *,
    duration_s: float = 1.0,
    max_tokens: int = 96,
    cas_dir: str | None = None,
) -> dict[str, Any]:
    import tempfile

    kill = await compare(duration_s=duration_s, max_tokens=max_tokens)
    andes = await andes_complete(duration_s=duration_s, max_tokens=max_tokens)
    gram = await grammar_ablation()
    with tempfile.TemporaryDirectory() as td:
        cas = await cas_peer_hit(cas_dir or td)
    mesh = await mesh_handover_suite()
    return {
        "kill_test": kill,
        "andes_complete": andes,
        "grammar": gram,
        "cas_peer": cas,
        "mesh": mesh,
        "limitations": "docs/limitations.md",
    }


async def mesh_handover_suite() -> dict[str, Any]:
    """Pin resume, CAS route, pin-holder preference, NIXL handover without prefill."""
    from dart.mesh import build_local_mesh
    from dart.protocol import CipName, Interest

    mesh = build_local_mesh(
        3,
        connector="nixl",
        seed=11,
        costs=[10.0, 5.0, 1.0],
        poll_interval_s=0.001,
        decode_quota=64,
        segment_size=8,
        w_init=16,
        interest_lifetime_s=5.0,
    )
    await mesh.start()
    try:
        handle = await mesh.open("named mesh continuation", node_id="node-0", max_tokens=48)
        prefills_after_open = [getattr(n.runtime.engine, "prefills", 0) for n in mesh.nodes]
        name0 = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
        first, d0 = await mesh.route(
            Interest(name=name0, window=8, lifetime_ms=3000, lease=handle.lease, cont_id=handle.cont_id),
            lease=handle.lease,
        )
        kernels_after_first = [getattr(n.runtime.engine, "kernel_launches", 0) for n in mesh.nodes]

        # CAS: same name, any node, no extra kernel.
        again, d_cas = await mesh.route(
            Interest(name=name0, window=8, lifetime_ms=3000, lease=handle.lease, cont_id=handle.cont_id),
            lease=handle.lease,
        )
        kernels_after_cas = [getattr(n.runtime.engine, "kernel_launches", 0) for n in mesh.nodes]

        # Pin holder (node-0, expensive) must win over cheapest (node-2).
        name1 = CipName.tokens(handle.model_hash, first.kv_root, 1).render()
        second, d_pin = await mesh.route(
            Interest(name=name1, window=8, lifetime_ms=3000, lease=handle.lease, cont_id=handle.cont_id),
            lease=handle.lease,
        )
        kernels_after_pin = [getattr(n.runtime.engine, "kernel_launches", 0) for n in mesh.nodes]

        await mesh.nodes[0].runtime.release_for_handover(handle.cont_id)
        name2 = CipName.tokens(handle.model_hash, second.kv_root, 2).render()
        third, d_ho = await mesh.route(
            Interest(name=name2, window=8, lifetime_ms=3000, lease=handle.lease, cont_id=handle.cont_id),
            lease=handle.lease,
        )
        prefills_after = [getattr(n.runtime.engine, "prefills", 0) for n in mesh.nodes]
        kernels_after_ho = [getattr(n.runtime.engine, "kernel_launches", 0) for n in mesh.nodes]

        ok = (
            d0.kind.value == "pin"
            and d_cas.kind.value == "cas"
            and d_cas.cache_hit
            and again.text == first.text
            and kernels_after_cas == kernels_after_first
            and d_pin.node_id == "node-0"
            and d_pin.kind.value == "pin"
            and kernels_after_pin[2] == kernels_after_first[2]
            and d_ho.kind.value == "handover"
            and d_ho.node_id == "node-2"
            and d_ho.prefill_skipped
            and prefills_after == prefills_after_open
            and prefills_after[2] == 0
            and kernels_after_ho[2] >= 1
            and third.token_count() > 0
        )
        return {
            "open_prefills": prefills_after_open,
            "first_route": d0.as_dict(),
            "cas_route": d_cas.as_dict(),
            "pin_route": d_pin.as_dict(),
            "handover_route": d_ho.as_dict(),
            "kernels": {
                "after_first": kernels_after_first,
                "after_cas": kernels_after_cas,
                "after_pin": kernels_after_pin,
                "after_handover": kernels_after_ho,
            },
            "prefills_after": prefills_after,
            "transfer_bytes": d_ho.transfer_bytes,
            "connector": mesh.connector.metrics(),
            "ok": ok,
        }
    finally:
        await mesh.aclose()
