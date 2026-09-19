"""CLI: serve, peer, experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dart",
        description="DART — Demand-Addressed Runtime Tokens (Interest-Driven Decode)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the CIP + OpenAI-compatible gateway")
    serve.add_argument(
        "--engine",
        default=os.environ.get("DART_ENGINE", "synthetic"),
        help="synthetic | hf | vllm | vllm-inprocess | llamacpp | cache",
    )
    serve.add_argument(
        "--model",
        default=os.environ.get("DART_MODEL", "dart-synth-8b"),
        help="Model id. For --engine hf, default is HuggingFaceTB/SmolLM2-135M-Instruct.",
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.environ.get("DART_PORT", "8090")))
    serve.add_argument("--cas-dir", default=os.environ.get("DART_CAS_DIR"))
    serve.add_argument("--reload", action="store_true")
    serve.add_argument(
        "--connector",
        default=os.environ.get("DART_KV_CONNECTOR", "memory"),
        help="KV connector: memory | file | lmcache | nixl",
    )

    peer = sub.add_parser("peer", help="CAS-only peer: satisfy Interests with no GPU")
    peer.add_argument("--cas-dir", required=True)
    peer.add_argument("--host", default="0.0.0.0")
    peer.add_argument("--port", type=int, default=8091)

    mesh = sub.add_parser("mesh", help="In-process multi-node Interest mesh")
    mesh.add_argument("--nodes", type=int, default=3)
    mesh.add_argument(
        "--connector",
        default="nixl",
        choices=["memory", "file", "lmcache", "nixl"],
    )
    mesh.add_argument("--host", default="0.0.0.0")
    mesh.add_argument("--port", type=int, default=8090)
    mesh.add_argument("--kv-dir", default=None)

    exp = sub.add_parser("experiment", help="Kill-test / Andes / grammar / CAS / paper suite")
    exp.add_argument("--seconds", type=float, default=2.0)
    exp.add_argument("--max-tokens", type=int, default=256)
    exp.add_argument("--step-latency", type=float, default=0.0)
    exp.add_argument("--json", action="store_true", dest="as_json")
    exp.add_argument(
        "--suite",
        default="kill",
        choices=["kill", "andes", "grammar", "cas", "paper", "mesh", "waiting"],
    )
    exp.add_argument("--cas-dir", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "peer":
        return _peer(args)
    if args.cmd == "mesh":
        return _mesh(args)
    if args.cmd == "experiment":
        return _experiment(args)
    parser.error("unknown command")
    return 2


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from dart.factory import build_runtime
    from dart.api.gateway import create_app

    # Console demo pages are 30-token Interests. Grant a full page in one
    # decode (w_init / segment_size 32). Paper-suite configs stay at 16.
    runtime = build_runtime(
        args.engine,
        args.model,
        cas_dir=args.cas_dir,
        connector=args.connector,
        w_init=32,
        startup_credit=32,
        segment_size=32,
        interest_lifetime_s=15.0,
    )
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


def _peer(args: argparse.Namespace) -> int:
    import uvicorn

    from dart.api.gateway import create_peer_app

    app = create_peer_app(args.cas_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _mesh(args: argparse.Namespace) -> int:
    import uvicorn

    from dart.api.gateway import create_app
    from dart.mesh import build_local_mesh

    router = build_local_mesh(
        args.nodes,
        connector=args.connector,
        kv_path=args.kv_dir,
        poll_interval_s=0.002,
    )
    app = create_app(router.nodes[0].runtime, router=router)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _experiment(args: argparse.Namespace) -> int:
    from dart.eval.experiment import (
        andes_complete,
        cas_peer_hit,
        compare,
        grammar_ablation,
        mesh_handover_suite,
        paper_suite,
        waiting_plugin_suite,
    )

    if args.suite == "kill":
        report = asyncio.run(
            compare(
                duration_s=args.seconds,
                max_tokens=args.max_tokens,
                step_latency_s=args.step_latency,
            )
        )
        if args.as_json:
            json.dump(report, sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0
        return _print_kill(report)
    if args.suite == "andes":
        report = asyncio.run(
            andes_complete(
                duration_s=args.seconds,
                max_tokens=args.max_tokens,
                step_latency_s=args.step_latency,
            )
        )
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0 if report["idd_beats_andes_on_inventory"] else 1
    if args.suite == "grammar":
        report = asyncio.run(grammar_ablation())
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if report["one_data_object"] and report["masking_more_launches"] else 1
    if args.suite == "cas":
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            report = asyncio.run(cas_peer_hit(args.cas_dir or td))
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if report["match"] else 1
    if args.suite == "mesh":
        report = asyncio.run(mesh_handover_suite())
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0 if report["ok"] else 1
    if args.suite == "waiting":
        report = asyncio.run(waiting_plugin_suite())
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0 if report["ok"] else 1
    report = asyncio.run(
        paper_suite(duration_s=args.seconds, max_tokens=args.max_tokens, cas_dir=args.cas_dir)
    )
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    ok = (
        report["kill_test"]["kill_test"]["survive"]
        and report["andes_complete"]["idd_beats_andes_on_inventory"]
        and report["grammar"]["one_data_object"]
        and report["cas_peer"]["match"]
        and report["mesh"]["ok"]
        and report["waiting_plugin"]["ok"]
    )
    return 0 if ok else 1


def _print_kill(report: dict) -> int:
    kt = report["kill_test"]
    print("DART kill-test (push vs credit-gated decode)")
    print("-" * 56)
    for key in ("push", "credit_reading_30tps", "credit_api_200tps", "credit_json_bursty"):
        row = report[key]
        print(
            f"{key:24} gen={row['tokens_generated']:4} "
            f"consumed={row['tokens_consumed']:4} "
            f"unused={row['tokens_generated_unconsumed']:4} "
            f"kernels={row['decode_kernel_launches']:4} "
            f"engine={row['engine_kernel_launches']:4} "
            f"kv_hw={row['kv_high_water']}"
        )
    print("-" * 56)
    print(f"generated cut vs push:     {kt['generated_cut_vs_push']}")
    print(f"KV high-water cut vs push: {kt['kv_high_water_cut_vs_push']}")
    print(f"engine kernel cut vs push: {kt.get('engine_kernel_cut_vs_push')}")
    print(f"survive inversion:         {kt['survive']}")
    return 0 if kt["survive"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
