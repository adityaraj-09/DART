"""DART — Demand-Addressed Runtime Tokens.

A named continuation runtime: outstanding Interests are the only thing
that may run a decode kernel.
"""

from dart.client.sdk import DartClient
from dart.eval.andes import run_andes
from dart.core.cc import CongestionController
from dart.client.consumers import (
    DrainPacer,
    JsonNeedPacer,
    ReadingPacer,
    ToolCallPacer,
    TtsPacer,
    ViewportPacer,
)
from dart.engine.base import DecodeResult, Engine, PrefillResult
from dart.engine.cache_only import CacheOnlyEngine
from dart.engine.stats import KernelStats, stats_of
from dart.engine.synthetic import SyntheticEngine
from dart.engine.vllm_inprocess import InProcessVLLMEngine
from dart.core.errors import (
    AmplificationError,
    DartError,
    EngineError,
    HandoverError,
    InterestNack,
    LeaseError,
    NameParseError,
    PinMissError,
    TenantQuotaError,
)
from dart.kv.kvconn import KVBlob, build_connector
from dart.core.lease import ContinuationLease, sign_lease, verify_lease
from dart.mesh import InterestRouter, build_local_mesh
from dart.kv.pin import FilePinnedKVPool, PinnedKVPool
from dart.cip.protocol import CipName, Data, Interest, Nack
from dart.core.runtime import ContinuationHandle, DartRuntime
from dart.core.types import RuntimeConfig
from dart.kv.store import FileCAS, KVStore, MemoryCAS

__version__ = "0.1.0"

__all__ = [
    "AmplificationError",
    "CacheOnlyEngine",
    "CipName",
    "CongestionController",
    "ContinuationHandle",
    "ContinuationLease",
    "DartClient",
    "DartError",
    "DartRuntime",
    "Data",
    "DecodeResult",
    "DrainPacer",
    "Engine",
    "EngineError",
    "FileCAS",
    "FilePinnedKVPool",
    "HandoverError",
    "InProcessVLLMEngine",
    "Interest",
    "InterestNack",
    "InterestRouter",
    "JsonNeedPacer",
    "KernelStats",
    "KVBlob",
    "KVStore",
    "LeaseError",
    "MemoryCAS",
    "Nack",
    "NameParseError",
    "PinnedKVPool",
    "PinMissError",
    "PrefillResult",
    "ReadingPacer",
    "RuntimeConfig",
    "SyntheticEngine",
    "TenantQuotaError",
    "ToolCallPacer",
    "TtsPacer",
    "ViewportPacer",
    "build_connector",
    "build_local_mesh",
    "run_andes",
    "sign_lease",
    "stats_of",
    "verify_lease",
    "__version__",
]
