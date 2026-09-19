"""Build a runtime from CLI flags / environment."""

from __future__ import annotations

import os
from pathlib import Path

from dart.engine.cache_only import CacheOnlyEngine
from dart.engine.hf import HuggingFaceEngine
from dart.engine.llamacpp import LlamaCppEngine
from dart.engine.synthetic import SyntheticEngine
from dart.engine.vllm import VLLMChatEngine
from dart.engine.vllm_inprocess import InProcessVLLMEngine
from dart.kv.kvconn import KVConnector, build_connector
from dart.kv.pin import PinnedKVPool
from dart.core.runtime import DartRuntime
from dart.core.types import ModelConfig, RuntimeConfig


def build_engine(
    kind: str = "synthetic",
    model: str | None = None,
    *,
    step_latency_s: float = 0.0,
) -> (
    SyntheticEngine
    | VLLMChatEngine
    | LlamaCppEngine
    | CacheOnlyEngine
    | InProcessVLLMEngine
    | HuggingFaceEngine
):
    kind = (kind or os.environ.get("DART_ENGINE") or "synthetic").lower()
    model = model or os.environ.get("DART_MODEL") or "dart-synth-8b"
    if kind in {"synthetic", "synth", "local"}:
        return SyntheticEngine(model, step_latency_s=step_latency_s)
    if kind in {"vllm-inprocess", "vllm_inprocess", "vllm-waiting", "waiting"}:
        return InProcessVLLMEngine(model, step_latency_s=step_latency_s)
    if kind in {"vllm-http", "vllm_http"}:
        return VLLMChatEngine(model)
    if kind == "vllm":
        # Production default: admit once, park in waiting. Remote HTTP only
        # when an explicit vLLM server URL is configured.
        if os.environ.get("DART_VLLM_URL"):
            return VLLMChatEngine(model)
        return InProcessVLLMEngine(model, step_latency_s=step_latency_s)
    if kind in {"llamacpp", "llama.cpp", "llama"}:
        return LlamaCppEngine(model)
    if kind in {"hf", "huggingface", "transformers"}:
        device = os.environ.get("DART_HF_DEVICE") or "cpu"
        return HuggingFaceEngine(model, device=device)
    if kind in {"cache", "cas", "peer"}:
        return CacheOnlyEngine(model)
    raise ValueError(
        "unknown engine "
        f"{kind!r}; use synthetic | hf | vllm | vllm-http | vllm-inprocess | llamacpp | cache"
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
    extra = dict(cfg)
    if isinstance(engine, HuggingFaceEngine):
        extra.setdefault("interest_lifetime_s", 15.0)
        extra.setdefault("t_decode_s", 0.08)
    cas = cas_dir or os.environ.get("DART_CAS_DIR")
    pin_dir = extra.pop("pin_dir", None) or os.environ.get("DART_PIN_DIR")
    if not pin_dir and cas:
        pin_dir = str(Path(cas) / "pins")
    quota_raw = extra.pop("tenant_interest_quota", None)
    if quota_raw is None:
        quota_raw = os.environ.get("DART_TENANT_QUOTA") or 0
    prev = extra.pop("secret_previous", None) or os.environ.get("DART_SECRET_PREV") or None
    tenant = extra.pop("default_tenant", None) or os.environ.get("DART_TENANT") or "default"
    config = RuntimeConfig(
        cas_dir=cas,
        pin_dir=pin_dir,
        secret=secret or os.environ.get("DART_SECRET") or "dart-dev-secret-change-me",
        secret_previous=prev,
        tenant_interest_quota=int(quota_raw or 0),
        default_tenant=str(tenant),
        producer_id=os.environ.get("DART_PRODUCER_ID") or "local-0",
        **{k: v for k, v in extra.items() if v is not None},  # type: ignore[arg-type]
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
    if pin_pool is None and config.pin_dir:
        from dart.kv.pin import FilePinnedKVPool

        pin_pool = FilePinnedKVPool(config.pin_dir)
    return DartRuntime(engine, config, pin_pool=pin_pool, connector=conn)


def describe_engine(engine: object) -> dict[str, str]:
    cfg = getattr(engine, "config", ModelConfig(model_id="unknown"))
    return {
        "class": type(engine).__name__,
        "model_id": getattr(engine, "model_id", cfg.model_id),
        "fingerprint": cfg.fingerprint(),
    }
