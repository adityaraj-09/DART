"""CLI: `dart serve` and `dart experiment`."""

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
    serve.add_argument("--engine", default=os.environ.get("DART_ENGINE", "synthetic"))
    serve.add_argument("--model", default=os.environ.get("DART_MODEL", "dart-synth-8b"))
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.environ.get("DART_PORT", "8090")))
    serve.add_argument("--cas-dir", default=os.environ.get("DART_CAS_DIR"))
    serve.add_argument("--reload", action="store_true")

    exp = sub.add_parser("experiment", help="Push vs credit-gated kill-test")
    exp.add_argument("--seconds", type=float, default=2.0)
    exp.add_argument("--max-tokens", type=int, default=256)
    exp.add_argument("--step-latency", type=float, default=0.0)
    exp.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "experiment":
        return _experiment(args)
    parser.error("unknown command")
    return 2


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from dart.factory import build_runtime
    from dart.gateway import create_app

    runtime = build_runtime(args.engine, args.model, cas_dir=args.cas_dir)
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


def _experiment(args: argparse.Namespace) -> int:
    from dart.experiment import compare

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
            f"skipped={row['decode_steps_skipped']:4} "
            f"kv_hw={row['kv_high_water']}"
        )
    print("-" * 56)
    print(f"generated cut vs push:     {kt['generated_cut_vs_push']}")
    print(f"KV high-water cut vs push: {kt['kv_high_water_cut_vs_push']}")
    print(f"survive inversion:         {kt['survive']}")
    return 0 if kt["survive"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
