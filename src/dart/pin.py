"""In-process pinned KV keyed by Merkle kv_root.

Sleeping a continuation must not forget identity. A compatible producer
that already holds the pin resumes decode without re-prefill: the engine
state (ids, sampler, extents, pos) is the continuation, not a vLLM
request id. Historical roots stay pinned until the lease expires so a
later Interest can still name that prefix.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dart.errors import PinMissError
from dart.types import EngineState, KVExtent


@dataclass
class PinRecord:
    kv_root: str
    model_hash: str
    state: EngineState
    holder_id: str
    pinned_until: float
    cont_id: str
    lease: str = ""
    segment_index: int = 0
    output_text: str = ""
    prompt_text: str = ""
    max_tokens: int = 0
    on_gpu: bool = True
    nbytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def clone_state(self) -> EngineState:
        return self.state.model_copy(deep=True)

    def snapshot(self) -> "PinRecord":
        return PinRecord(
            kv_root=self.kv_root,
            model_hash=self.model_hash,
            state=self.clone_state(),
            holder_id=self.holder_id,
            pinned_until=self.pinned_until,
            cont_id=self.cont_id,
            lease=self.lease,
            segment_index=self.segment_index,
            output_text=self.output_text,
            prompt_text=self.prompt_text,
            max_tokens=self.max_tokens,
            on_gpu=self.on_gpu,
            nbytes=self.nbytes,
            created_at=self.created_at,
            last_access=self.last_access,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kv_root": self.kv_root,
            "model_hash": self.model_hash,
            "holder_id": self.holder_id,
            "pinned_until": self.pinned_until,
            "cont_id": self.cont_id,
            "segment_index": self.segment_index,
            "on_gpu": self.on_gpu,
            "nbytes": self.nbytes,
            "pos": self.state.pos,
            "stopped": self.state.stopped,
        }


class PinnedKVPool:
    """Process-local (or shared-mesh) pin table indexed by kv_root."""

    def __init__(self, *, max_pins: int = 4096) -> None:
        self.max_pins = max_pins
        self._recs: dict[str, PinRecord] = {}
        self._lock = threading.Lock()
        self.pins = 0
        self.hits = 0
        self.misses = 0
        self.adopts = 0
        self.drops = 0
        self._high_water = 0

    def pin(
        self,
        kv_root: str,
        state: EngineState,
        *,
        holder_id: str,
        until: float,
        cont_id: str,
        model_hash: str,
        lease: str = "",
        segment_index: int = 0,
        output_text: str = "",
        prompt_text: str = "",
        max_tokens: int = 0,
        on_gpu: bool = True,
    ) -> PinRecord:
        if not kv_root:
            raise ValueError("kv_root is required")
        extents = [e.model_copy() for e in state.kv_extents]
        nbytes = sum(e.nbytes for e in extents)
        snap = state.model_copy(deep=True)
        snap.kv_extents = extents
        now = time.time()
        with self._lock:
            rec = self._recs.get(kv_root)
            if rec is None:
                rec = PinRecord(
                    kv_root=kv_root,
                    model_hash=model_hash,
                    state=snap,
                    holder_id=holder_id,
                    pinned_until=until,
                    cont_id=cont_id,
                    lease=lease,
                    segment_index=segment_index,
                    output_text=output_text,
                    prompt_text=prompt_text,
                    max_tokens=max_tokens,
                    on_gpu=on_gpu,
                    nbytes=nbytes,
                    created_at=now,
                    last_access=now,
                )
                self._recs[kv_root] = rec
                self.pins += 1
            else:
                rec.state = snap
                rec.holder_id = holder_id
                rec.pinned_until = max(rec.pinned_until, until)
                rec.cont_id = cont_id
                rec.model_hash = model_hash
                rec.lease = lease or rec.lease
                rec.segment_index = segment_index
                rec.output_text = output_text
                rec.prompt_text = prompt_text
                rec.max_tokens = max_tokens or rec.max_tokens
                rec.on_gpu = on_gpu
                rec.nbytes = nbytes
                rec.last_access = now
            self._high_water = max(self._high_water, len(self._recs))
            self._evict_unlocked(now)
            return rec.snapshot()

    def lookup(self, kv_root: str, *, now: float | None = None) -> PinRecord | None:
        now = time.time() if now is None else now
        with self._lock:
            rec = self._recs.get(kv_root)
            if rec is None:
                self.misses += 1
                return None
            if rec.pinned_until <= now:
                self.misses += 1
                return None
            rec.last_access = now
            self.hits += 1
            return rec.snapshot()

    def has(self, kv_root: str, *, now: float | None = None) -> bool:
        return self.lookup(kv_root, now=now) is not None

    def adopt(self, kv_root: str, new_holder: str, *, now: float | None = None) -> PinRecord:
        """Transfer the pin to a new holder. Returns a cloned snapshot."""
        now = time.time() if now is None else now
        with self._lock:
            rec = self._recs.get(kv_root)
            if rec is None or rec.pinned_until <= now:
                self.misses += 1
                raise PinMissError(f"no live pin for kv_root {kv_root[:16]}")
            rec.holder_id = new_holder
            rec.on_gpu = True
            rec.last_access = now
            self.adopts += 1
            self.hits += 1
            return rec.snapshot()

    def sleep(self, kv_root: str) -> None:
        with self._lock:
            rec = self._recs.get(kv_root)
            if rec is None:
                return
            rec.on_gpu = False
            for e in rec.state.kv_extents:
                e.on_gpu = False

    def wake(self, kv_root: str) -> None:
        with self._lock:
            rec = self._recs.get(kv_root)
            if rec is None:
                return
            rec.on_gpu = True
            for e in rec.state.kv_extents:
                e.on_gpu = True

    def drop_if_unpinned(self, kv_root: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            rec = self._recs.get(kv_root)
            if rec is None:
                return False
            if rec.pinned_until > now:
                return False
            del self._recs[kv_root]
            self.drops += 1
            return True

    def drop_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            return self._drop_expired_unlocked(now)

    def holder_of(self, kv_root: str) -> str | None:
        rec = self.lookup(kv_root)
        return rec.holder_id if rec else None

    def pins_for_holder(self, holder_id: str) -> list[PinRecord]:
        with self._lock:
            return [r.snapshot() for r in self._recs.values() if r.holder_id == holder_id]

    def holders(self) -> dict[str, str]:
        with self._lock:
            return {k: r.holder_id for k, r in self._recs.items()}

    def gpu_bytes(self) -> int:
        with self._lock:
            return sum(r.nbytes for r in self._recs.values() if r.on_gpu)

    def current_bytes(self) -> int:
        with self._lock:
            return sum(r.nbytes for r in self._recs.values())

    def __len__(self) -> int:
        return len(self._recs)

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pins": self.pins,
                "live": len(self._recs),
                "hits": self.hits,
                "misses": self.misses,
                "adopts": self.adopts,
                "drops": self.drops,
                "gpu_bytes": sum(r.nbytes for r in self._recs.values() if r.on_gpu),
                "bytes": sum(r.nbytes for r in self._recs.values()),
                "high_water": self._high_water,
            }

    def _drop_expired_unlocked(self, now: float) -> int:
        dead = [k for k, r in self._recs.items() if r.pinned_until <= now]
        for k in dead:
            del self._recs[k]
            self.drops += 1
        return len(dead)

    def _evict_unlocked(self, now: float) -> None:
        self._drop_expired_unlocked(now)
        if len(self._recs) <= self.max_pins:
            return
        # Prefer sleeping, then oldest last_access. Never evict the newest pin.
        ranked = sorted(
            self._recs.values(),
            key=lambda r: (r.on_gpu, r.last_access),
        )
        while len(self._recs) > self.max_pins and ranked:
            victim = ranked.pop(0)
            self._recs.pop(victim.kv_root, None)
            self.drops += 1
