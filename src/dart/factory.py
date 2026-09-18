"""Build a runtime from CLI flags / environment."""

from __future__ import annotations

import os

from dart.engine.cache_only import CacheOnlyEngine
from dart.engine.llamacpp import LlamaCppEngine
from dart.engine.synthetic import SyntheticEngine
from dart.engine.vllm import VLLMChatEngine
from dart.engine.vllm_inprocess import InProcessVLLMEngine
from dart.kvconn import KVConnector, build_connector
from dart.pin import PinnedKVPool
from dart.runtime import DartRuntime
from dart.types import ModelConfig, RuntimeConfig


def build_engine(
    kind: str = "synthetic",
    model: str | None = None,
    *,
    step_latency_s: float = 0.0,
) -> SyntheticEngine | VLLMChatEngine | LlamaCppEngine | CacheOnlyEngine | InProcessVLLMEngine:
    kind = (kind or os.environ.get("DART_ENGINE") or "synthetic").lower()
    model = model or os.environ.get("DART_MODEL") or "dart-synth-8b"
    if kind in {"synthetic", "synth", "local"}:
        return SyntheticEngine(model, step_latency_s=step_latency_s)
    if kind in {"vllm-inprocess", "vllm_inprocess", "vllm-waiting", "waiting"}:
        return InProcessVLLMEngine(model, step_latency_s=step_latency_s)
    if kind == "vllm":
        return VLLMChatEngine(model)
    if kind in {"llamacpp", "llama.cpp", "llama"}:
        return LlamaCppEngine(model)
    if kind in {"cache", "cas", "peer"}:
        return CacheOnlyEngine(model)
    raise ValueError(
        f"unknown engine {kind!r}; use synthetic | vllm | vllm-inprocess | llamacpp | cache"
    )


def build_runtime(
    kind: str = "synthetic",
    model: str | None = None,
    *,
    cas_dir: str | None = None,
    secret: str | None = None,
    step_latency_s: float = 0.0,
    connector: str | KVConnector | None = None,
    pin_pool: PinnedKVPool | None = None,
    **cfg: object,
) -> DartRuntime:
    engine = build_engine(kind, model, step_latency_s=step_latency_s)
    config = RuntimeConfig(
        cas_dir=cas_dir or os.environ.get("DART_CAS_DIR"),
        secret=secret or os.environ.get("DART_SECRET") or "dart-dev-secret-change-me",
        producer_id=os.environ.get("DART_PRODUCER_ID") or "local-0",
        **{k: v for k, v in cfg.items() if v is not None},  # type: ignore[arg-type]
    )
    conn: KVConnector | None
    if isinstance(connector, str) or connector is None:
        kind_conn = connector or os.environ.get("DART_KV_CONNECTOR") or "memory"
        kv_path = os.environ.get("DART_KV_DIR") or cas_dir
        try:
            conn = build_connector(kind_conn, path=kv_path if kind_conn in {"file", "nixl"} else None)
        except ValueError:
            conn = build_connector("memory")
    else:
        conn = connector
    return DartRuntime(engine, config, pin_pool=pin_pool, connector=conn)


def describe_engine(engine: object) -> dict[str, str]:
    cfg = getattr(engine, "config", ModelConfig(model_id="unknown"))
    return {
        "class": type(engine).__name__,
        "model_id": getattr(engine, "model_id", cfg.model_id),
        "fingerprint": cfg.fingerprint(),
    }
