"""Credit-gated vLLM-shaped scheduler: waiting + pinned KV blocks.

vLLM's default scheduler will keep generating a running request, and prefix
cache may evict KV under pressure. This plugin is the opposite for IDD:

- Admit once (prefill). The request then sits in ``waiting``.
- Blocks are **pinned** while the continuation is live. Memory pressure may
  evict unpinned prefix-cache blocks, never a waiting pin.
- ``schedule_decode(n)`` moves waiting → running → waiting. No second admit.
- ``W=0`` (DART does not call decode, or ``pause(mode=keep)``) does not free
  blocks and does not finish the request.

The residual-stream kernel is pluggable. Tests use SyntheticEngine. A real
vLLM model runner can sit in the same slots.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class KVBlock:
    block_id: int
    request_id: str
    digest: str
    nbytes: int
    pinned: bool = True
    on_gpu: bool = True


@dataclass
class SchedulerRequest:
    request_id: str
    status: RequestStatus = RequestStatus.WAITING
    token_ids: list[int] = field(default_factory=list)
    num_computed: int = 0
    credit: int = 0
    pinned: bool = True
    admitted_at: float = field(default_factory=time.time)
    decode_steps: int = 0
    block_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "num_computed": self.num_computed,
            "credit": self.credit,
            "pinned": self.pinned,
            "blocks": len(self.block_ids),
            "decode_steps": self.decode_steps,
        }


class BlockPool:
    """GPU block allocator. Pinned blocks are not eviction candidates."""

    def __init__(self, *, num_gpu_blocks: int = 2048, block_nbytes: int = 4096) -> None:
        self.num_gpu_blocks = num_gpu_blocks
        self.block_nbytes = block_nbytes
        self._blocks: dict[int, KVBlock] = {}
        self._by_req: dict[str, list[int]] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self.evictions = 0
        self.allocs = 0

    def allocate(self, request_id: str, n_blocks: int, *, digest: str = "", pinned: bool = True) -> list[int]:
        n_blocks = max(1, n_blocks)
        with self._lock:
            free = self.num_gpu_blocks - len(self._blocks)
            if n_blocks > free:
                self._evict_unlocked(need=n_blocks - free)
            free = self.num_gpu_blocks - len(self._blocks)
            if n_blocks > free:
                raise MemoryError(
                    f"need {n_blocks} GPU blocks, {free} free after evicting unpinned"
                )
            ids: list[int] = []
            for _ in range(n_blocks):
                bid = self._next_id
                self._next_id += 1
                self._blocks[bid] = KVBlock(
                    block_id=bid,
                    request_id=request_id,
                    digest=digest,
                    nbytes=self.block_nbytes,
                    pinned=pinned,
                    on_gpu=True,
                )
                ids.append(bid)
                self.allocs += 1
            self._by_req.setdefault(request_id, []).extend(ids)
            return ids

    def pin(self, request_id: str) -> None:
        with self._lock:
            for bid in self._by_req.get(request_id, []):
                blk = self._blocks.get(bid)
                if blk is None:
                    continue
                blk.pinned = True
                blk.on_gpu = True

    def unpin(self, request_id: str) -> None:
        """Prefix-cache eligible: still on GPU until pressure."""
        with self._lock:
            for bid in self._by_req.get(request_id, []):
                blk = self._blocks.get(bid)
                if blk is not None:
                    blk.pinned = False

    def free(self, request_id: str) -> int:
        with self._lock:
            return self._free_unlocked(request_id)

    def evict_unpinned(self, need_blocks: int) -> int:
        with self._lock:
            return self._evict_unlocked(need=need_blocks)

    def blocks_for(self, request_id: str) -> list[KVBlock]:
        with self._lock:
            return [self._blocks[i] for i in self._by_req.get(request_id, []) if i in self._blocks]

    def has_blocks(self, request_id: str) -> bool:
        with self._lock:
            return any(i in self._blocks for i in self._by_req.get(request_id, []))

    def gpu_bytes(self, *, pinned_only: bool = False) -> int:
        with self._lock:
            return sum(
                b.nbytes
                for b in self._blocks.values()
                if b.on_gpu and (b.pinned if pinned_only else True)
            )

    def pinned_blocks(self) -> int:
        with self._lock:
            return sum(1 for b in self._blocks.values() if b.pinned)

    def live_blocks(self) -> int:
        with self._lock:
            return len(self._blocks)

    def _free_unlocked(self, request_id: str) -> int:
        ids = self._by_req.pop(request_id, [])
        n = 0
        for bid in ids:
            blk = self._blocks.pop(bid, None)
            if blk is not None:
                n += 1
        return n

    def _evict_unlocked(self, need: int) -> int:
        if need <= 0:
            return 0
        # Group unpinned by request; drop whole sequences.
        victims: dict[str, list[int]] = {}
        for bid, blk in self._blocks.items():
            if blk.pinned:
                continue
            victims.setdefault(blk.request_id, []).append(bid)
        freed = 0
        for rid, ids in list(victims.items()):
            if freed >= need:
                break
            for bid in ids:
                self._blocks.pop(bid, None)
                freed += 1
                self.evictions += 1
            remaining = [i for i in self._by_req.get(rid, []) if i in self._blocks]
            if remaining:
                self._by_req[rid] = remaining
            else:
                self._by_req.pop(rid, None)
        return freed


class CreditGatedScheduler:
    """vLLM waiting-queue plugin: decode only with credit; pin otherwise."""

    def __init__(self, *, num_gpu_blocks: int = 2048, block_nbytes: int = 4096) -> None:
        self.blocks = BlockPool(num_gpu_blocks=num_gpu_blocks, block_nbytes=block_nbytes)
        self._reqs: dict[str, SchedulerRequest] = {}
        self._lock = threading.Lock()
        self.admissions = 0
        self.admission_reentries = 0
        self.decode_forwards = 0
        self.prefill_forwards = 0
        self.pause_keeps = 0
        self.aborts = 0

    def has(self, request_id: str) -> bool:
        return request_id in self._reqs

    def get(self, request_id: str) -> SchedulerRequest | None:
        return self._reqs.get(request_id)

    def admit(
        self,
        request_id: str,
        *,
        n_tokens: int,
        n_blocks: int,
        token_ids: list[int] | None = None,
        digest: str = "",
        count_prefill: bool = True,
    ) -> SchedulerRequest:
        with self._lock:
            existing = self._reqs.get(request_id)
            if existing is not None and existing.status is not RequestStatus.FINISHED:
                self.admission_reentries += 1
                return existing
            ids = self.blocks.allocate(request_id, n_blocks, digest=digest, pinned=True)
            req = SchedulerRequest(
                request_id=request_id,
                status=RequestStatus.WAITING,
                token_ids=list(token_ids or []),
                num_computed=n_tokens,
                pinned=True,
                block_ids=ids,
            )
            self._reqs[request_id] = req
            self.admissions += 1
            if count_prefill:
                self.prefill_forwards += 1
            return req

    def pause(self, request_id: str, *, mode: str = "keep") -> SchedulerRequest | None:
        with self._lock:
            req = self._reqs.get(request_id)
            if req is None or req.status is RequestStatus.FINISHED:
                return req
            if mode == "keep":
                req.status = RequestStatus.WAITING
                req.pinned = True
                req.credit = 0
                self.blocks.pin(request_id)
                self.pause_keeps += 1
            else:
                req.status = RequestStatus.PREEMPTED
                req.pinned = False
                self.blocks.unpin(request_id)
            return req

    def resume(self, request_id: str, credit: int) -> SchedulerRequest:
        with self._lock:
            req = self._reqs.get(request_id)
            if req is None or req.status is RequestStatus.FINISHED:
                raise KeyError(request_id)
            if not self.blocks.has_blocks(request_id):
                raise MemoryError(f"KV blocks missing for {request_id}; would re-prefill")
            req.credit = max(0, credit)
            req.status = RequestStatus.RUNNING
            req.pinned = True
            self.blocks.pin(request_id)
            return req

    def after_decode(self, request_id: str, new_tokens: int, token_ids: list[int] | None = None) -> None:
        with self._lock:
            req = self._reqs.get(request_id)
            if req is None:
                return
            req.num_computed += max(0, new_tokens)
            req.decode_steps += 1
            if token_ids:
                req.token_ids = list(token_ids)
            self.decode_forwards += 1
            req.credit = max(0, req.credit - new_tokens)
            req.status = RequestStatus.WAITING

    def abort(self, request_id: str) -> None:
        with self._lock:
            req = self._reqs.get(request_id)
            if req is None:
                return
            req.status = RequestStatus.FINISHED
            req.pinned = False
            self.blocks.free(request_id)
            self.aborts += 1

    def waiting(self) -> list[SchedulerRequest]:
        return [r for r in self._reqs.values() if r.status is RequestStatus.WAITING]

    def running(self) -> list[SchedulerRequest]:
        return [r for r in self._reqs.values() if r.status is RequestStatus.RUNNING]

    def snapshot(self) -> dict[str, Any]:
        return {
            "admissions": self.admissions,
            "admission_reentries": self.admission_reentries,
            "prefill_forwards": self.prefill_forwards,
            "decode_forwards": self.decode_forwards,
            "pause_keeps": self.pause_keeps,
            "aborts": self.aborts,
            "waiting": len(self.waiting()),
            "running": len(self.running()),
            "live_requests": sum(
                1 for r in self._reqs.values() if r.status is not RequestStatus.FINISHED
            ),
            "pinned_blocks": self.blocks.pinned_blocks(),
            "live_blocks": self.blocks.live_blocks(),
            "pinned_bytes": self.blocks.gpu_bytes(pinned_only=True),
            "gpu_bytes": self.blocks.gpu_bytes(),
            "evictions": self.blocks.evictions,
            "requests": [r.as_dict() for r in self._reqs.values()],
        }
