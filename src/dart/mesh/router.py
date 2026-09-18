"""Multi-node Interest routing.

An Interest is not sticky to the machine that prefills. The router picks
a producer in this order:

1. CAS hit — named Data already exists; no GPU.
2. Live pin holder — that node already has kv_root in GPU/host memory.
3. Cheapest node + handover pull — NIXL/LMCache transfer, then adopt
   without re-prefill.

Changing machines is not live-migration of a request. The HTTP request is
gone; the continuation object (lease + kv_root) remains.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from dart.core.errors import HandoverError, InterestNack, LeaseError
from dart.kv.kvconn import KVBlob, KVConnector, MemoryKVConnector
from dart.core.lease import verify_lease
from dart.kv.pin import PinnedKVPool
from dart.cip.protocol import Data, Interest
from dart.core.runtime import ContinuationHandle, DartRuntime
from dart.kv.store import MemoryCAS
from dart.core.types import NackReason, Prompt, RuntimeConfig

logger = logging.getLogger("dart.mesh")


class RouteKind(str, Enum):
    CAS = "cas"
    PIN = "pin"
    HANDOVER = "handover"
    LOCAL = "local"


@dataclass
class RouteDecision:
    kind: RouteKind
    node_id: str
    cache_hit: bool = False
    handover: bool = False
    from_node: str | None = None
    transfer_bytes: int = 0
    detail: str = ""
    prefill_skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "node_id": self.node_id,
            "cache_hit": self.cache_hit,
            "handover": self.handover,
            "from_node": self.from_node,
            "transfer_bytes": self.transfer_bytes,
            "detail": self.detail,
            "prefill_skipped": self.prefill_skipped,
        }


@dataclass
class MeshNode:
    node_id: str
    runtime: DartRuntime
    cost: float = 1.0

    def load(self) -> int:
        return sum(
            1
            for c in self.runtime._conts.values()
            if not c.done and not c.closed and not c.handed_over and not c.sleeping
        )

    def score(self) -> float:
        gpu = self.runtime.kv.gpu_bytes()
        return self.cost * (1.0 + self.load()) + gpu / 1e9


@dataclass
class RouterStats:
    cas_routes: int = 0
    pin_routes: int = 0
    handover_routes: int = 0
    local_routes: int = 0
    nacks: int = 0
    transfer_bytes: int = 0


class InterestRouter:
    """Fan-in for CIP Interests across several DartRuntime producers."""

    def __init__(
        self,
        nodes: list[MeshNode],
        *,
        pin_pool: PinnedKVPool | None = None,
        connector: KVConnector | None = None,
        cas: MemoryCAS | None = None,
        secret: str | None = None,
    ) -> None:
        if not nodes:
            raise ValueError("mesh requires at least one node")
        self.nodes = list(nodes)
        self._by_id: dict[str, MeshNode] = {n.node_id: n for n in self.nodes}
        self.pin_pool = pin_pool or nodes[0].runtime.pin_pool
        self.connector: KVConnector = connector or nodes[0].runtime.connector or MemoryKVConnector()
        self.cas = cas or nodes[0].runtime.cas
        self.secret = secret or nodes[0].runtime.config.secret
        self.stats = RouterStats()
        self._root_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
        self.history: list[dict[str, Any]] = []

    def register(self, node: MeshNode) -> None:
        self.nodes.append(node)
        self._by_id[node.node_id] = node

    def unregister(self, node_id: str) -> None:
        self._by_id.pop(node_id, None)
        self.nodes = [n for n in self.nodes if n.node_id != node_id]

    async def start(self) -> None:
        for n in self.nodes:
            await n.runtime.start()

    async def aclose(self) -> None:
        for n in self.nodes:
            await n.runtime.aclose()

    def cheapest(self, *, exclude: set[str] | None = None) -> MeshNode:
        exclude = exclude or set()
        live = [n for n in self.nodes if n.node_id not in exclude]
        if not live:
            raise HandoverError("no mesh nodes available")
        return min(live, key=lambda n: n.score())

    def node(self, node_id: str) -> MeshNode:
        try:
            return self._by_id[node_id]
        except KeyError as exc:
            raise HandoverError(f"unknown mesh node {node_id}") from exc

    async def open(
        self,
        prompt: Prompt | str | list[Any],
        *,
        node_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.8,
        model: str | None = None,
    ) -> ContinuationHandle:
        node = self.node(node_id) if node_id else self.cheapest()
        return await node.runtime.open(
            prompt, max_tokens=max_tokens, temperature=temperature, model=model
        )

    async def route(self, req: Interest, *, lease: str | None = None) -> tuple[Data, RouteDecision]:
        token = req.lease or lease
        if not token:
            raise LeaseError("missing lease")
        cap = verify_lease(token, self.secret)
        cap.assert_window(req.window)

        kv_root = req.parsed.kv_root
        async with self._root_lock(kv_root):
            cached = self._cas_get(req.name)
            if cached is not None:
                self.stats.cas_routes += 1
                decision = RouteDecision(
                    kind=RouteKind.CAS,
                    node_id=cached.producer_id or "cas",
                    cache_hit=True,
                    detail="named Data already in CAS",
                )
                self._record(req, decision)
                return cached.model_copy(update={"cache_hit": True}), decision

            holder = self._live_holder(kv_root, cap.cont_id)
            if holder is not None:
                data = await holder.runtime.interest(req, lease=token)
                self.stats.pin_routes += 1
                decision = RouteDecision(
                    kind=RouteKind.PIN,
                    node_id=holder.node_id,
                    detail=f"pin holder {holder.node_id} already has {kv_root[:12]}",
                )
                self._record(req, decision)
                return data, decision

            local = self._live_continuation(cap.cont_id, kv_root)
            if local is not None:
                data = await local.runtime.interest(req, lease=token)
                self.stats.local_routes += 1
                decision = RouteDecision(
                    kind=RouteKind.LOCAL,
                    node_id=local.node_id,
                    detail="continuation still live on node",
                )
                self._record(req, decision)
                return data, decision

            dest, from_node, blob, nbytes = await self._pull_kv(kv_root, cap.model_hash)
            handle = await dest.runtime.adopt(lease=token, blob=blob, from_node=from_node or "")
            if from_node and from_node in self._by_id and from_node != dest.node_id:
                try:
                    await self._by_id[from_node].runtime.release_for_handover(handle.cont_id)
                except LeaseError:
                    pass
            data = await dest.runtime.interest(req, lease=token)
            self.stats.handover_routes += 1
            self.stats.transfer_bytes += nbytes
            decision = RouteDecision(
                kind=RouteKind.HANDOVER,
                node_id=dest.node_id,
                handover=True,
                from_node=from_node,
                transfer_bytes=nbytes,
                prefill_skipped=True,
                detail=f"handover {from_node or 'store'} → {dest.node_id}",
            )
            self._record(req, decision)
            return data, decision

    def status(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "cost": n.cost,
                    "load": n.load(),
                    "score": n.score(),
                    "continuations": len(n.runtime._conts),
                    "engine_prefills": getattr(n.runtime.engine, "prefills", None),
                    "engine_kernels": getattr(n.runtime.engine, "kernel_launches", None),
                    "kv_gpu_bytes": n.runtime.kv.gpu_bytes(),
                }
                for n in self.nodes
            ],
            "pins": self.pin_pool.metrics(),
            "connector": self.connector.metrics(),
            "cas_entries": len(self.cas.names()) if hasattr(self.cas, "names") else 0,
            "stats": {
                "cas_routes": self.stats.cas_routes,
                "pin_routes": self.stats.pin_routes,
                "handover_routes": self.stats.handover_routes,
                "local_routes": self.stats.local_routes,
                "nacks": self.stats.nacks,
                "transfer_bytes": self.stats.transfer_bytes,
            },
            "recent": self.history[-16:],
        }

    def _cas_get(self, name: str) -> Data | None:
        data = self.cas.get_data(name)
        if data is not None:
            return data
        seen: set[int] = {id(self.cas)}
        for n in self.nodes:
            if id(n.runtime.cas) in seen:
                continue
            seen.add(id(n.runtime.cas))
            data = n.runtime.cas.get_data(name)
            if data is not None:
                return data
        return None

    def _live_holder(self, kv_root: str, cont_id: str | None) -> MeshNode | None:
        rec = self.pin_pool.lookup(kv_root)
        if rec is None:
            return None
        node = self._by_id.get(rec.holder_id)
        if node is None:
            return None
        try:
            cont = node.runtime.get(rec.cont_id if rec.cont_id else (cont_id or ""))
        except LeaseError:
            return None
        if cont.handed_over or cont.closed or cont.done:
            return None
        return node

    def _live_continuation(self, cont_id: str, kv_root: str | None = None) -> MeshNode | None:
        for n in self.nodes:
            try:
                cont = n.runtime.get(cont_id)
            except LeaseError:
                continue
            if cont.handed_over or cont.closed or cont.done:
                continue
            if kv_root and cont.state.kv_root != kv_root:
                rec = self.pin_pool.lookup(kv_root)
                if rec is None or rec.holder_id != n.node_id:
                    continue
            return n
        return None

    async def _pull_kv(
        self, kv_root: str, model_hash: str
    ) -> tuple[MeshNode, str | None, KVBlob, int]:
        rec = self.pin_pool.lookup(kv_root)
        from_node = rec.holder_id if rec else None
        exclude = {from_node} if from_node and len(self.nodes) > 1 else set()
        dest = self.cheapest(exclude=exclude)
        blob: KVBlob | None = None
        try:
            blob = await self.connector.transfer(
                kv_root, src=from_node or "", dst=dest.node_id
            )
        except HandoverError:
            blob = None
        if blob is None and rec is not None:
            blob = KVBlob.from_state(
                kv_root=rec.kv_root,
                model_hash=rec.model_hash,
                cont_id=rec.cont_id,
                state=rec.clone_state(),
                holder_id=dest.node_id,
                lease=rec.lease,
                segment_index=rec.segment_index,
                output_text=rec.output_text,
                prompt_text=rec.prompt_text,
                max_tokens=rec.max_tokens,
            )
            try:
                await self.connector.put(blob)
            except Exception:
                logger.exception("connector put during pin-sourced handover failed")
        if blob is None:
            blob = await self.connector.get(kv_root)
        if blob is None:
            self.stats.nacks += 1
            raise InterestNack(NackReason.UNKNOWN_NAME.value, f"no KV for {kv_root[:16]}")
        if blob.model_hash and blob.model_hash != model_hash:
            self.stats.nacks += 1
            raise InterestNack(NackReason.NO_MODEL.value, "model fingerprint mismatch on KV blob")
        nbytes = blob.nbytes
        return dest, from_node, blob, nbytes

    def _root_lock(self, kv_root: str) -> asyncio.Lock:
        lock = self._root_locks.get(kv_root)
        if lock is None:
            lock = asyncio.Lock()
            self._root_locks[kv_root] = lock
        return lock

    def _record(self, req: Interest, decision: RouteDecision) -> None:
        self.history.append(
            {
                "name": req.name,
                "at": time.time(),
                **decision.as_dict(),
            }
        )


def build_local_mesh(
    n: int = 3,
    *,
    connector: str = "nixl",
    seed: int = 0,
    costs: list[float] | None = None,
    kv_path: str | None = None,
    **cfg: Any,
) -> InterestRouter:
    """N in-process producers sharing CAS, pin table, and a KV connector."""
    from dart.engine.synthetic import SyntheticEngine
    from dart.kv.kvconn import build_connector
    from dart.kv.store import MemoryCAS

    if n < 1:
        raise ValueError("need at least one mesh node")
    cas = MemoryCAS()
    pool = PinnedKVPool()
    conn = build_connector(connector, path=kv_path)
    secret = str(cfg.pop("secret", None) or "dart-dev-secret-change-me")
    nodes: list[MeshNode] = []
    for i in range(n):
        engine = SyntheticEngine(model_id="dart-synth-8b", seed=seed)
        config = RuntimeConfig(
            producer_id=f"node-{i}",
            secret=secret,
            **{k: v for k, v in cfg.items() if v is not None},
        )
        rt = DartRuntime(engine, config, cas=cas, pin_pool=pool, connector=conn)
        cost = costs[i] if costs and i < len(costs) else 1.0 + i
        nodes.append(MeshNode(node_id=f"node-{i}", runtime=rt, cost=cost))
    return InterestRouter(nodes, pin_pool=pool, connector=conn, cas=cas, secret=secret)
