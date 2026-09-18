"""Content-addressed token segments and pinned KV extents.

Peers that already hold `/cip/.../tokens/seg/i` for a kv_root answer
without touching a GPU. KV extents can sleep (off GPU, still named) when
a continuation has zero Interests.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, unquote

from dart.protocol import Data
from dart.types import KVExtent


class ContentStore(Protocol):
    def put_data(self, name: str, data: Data) -> None: ...
    def get_data(self, name: str) -> Data | None: ...
    def has(self, name: str) -> bool: ...


class MemoryCAS:
    """In-process named Data cache. Safe for tests and single-node v1."""

    def __init__(self) -> None:
        self._data: dict[str, Data] = {}
        self._lock = threading.Lock()
        self.puts = 0
        self.hits = 0
        self.misses = 0

    def put_data(self, name: str, data: Data) -> None:
        with self._lock:
            self._data[name] = data
            self.puts += 1

    def get_data(self, name: str) -> Data | None:
        with self._lock:
            item = self._data.get(name)
            if item is None:
                self.misses += 1
            else:
                self.hits += 1
            return item

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._data

    def names(self) -> list[str]:
        with self._lock:
            return list(self._data)


class FileCAS(MemoryCAS):
    """Directory-backed CAS so a second process can Interest by reading files."""

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "names").mkdir(exist_ok=True)
        self._load()

    def _path(self, name: str) -> Path:
        return self.root / "names" / quote(name, safe="")

    def _load(self) -> None:
        for p in (self.root / "names").glob("*"):
            try:
                payload = json.loads(p.read_text())
                data = Data.model_validate(payload)
                self._data[unquote(p.name)] = data
            except (OSError, json.JSONDecodeError, ValueError):
                continue

    def put_data(self, name: str, data: Data) -> None:
        super().put_data(name, data)
        path = self._path(name)
        path.write_text(data.model_dump_json())


class KVRecord:
    __slots__ = ("extents", "pinned_until", "on_gpu", "nbytes")

    def __init__(self) -> None:
        self.extents: list[KVExtent] = []
        self.pinned_until: float = 0.0
        self.on_gpu: bool = True
        self.nbytes: int = 0


class KVStore:
    """Named KV extents with pin leases. Sleep does not forget identity."""

    def __init__(self) -> None:
        self._recs: dict[str, KVRecord] = {}
        self._lock = threading.Lock()
        self._high_water = 0
        self._current = 0

    def replace(self, cont_id: str, extents: list[KVExtent]) -> None:
        with self._lock:
            rec = self._recs.get(cont_id) or KVRecord()
            old = rec.nbytes
            rec.extents = list(extents)
            rec.nbytes = sum(e.nbytes for e in extents)
            rec.on_gpu = True
            self._recs[cont_id] = rec
            self._current += rec.nbytes - old
            self._high_water = max(self._high_water, self._current)

    def pin(self, cont_id: str, until: float) -> None:
        with self._lock:
            rec = self._recs.setdefault(cont_id, KVRecord())
            rec.pinned_until = max(rec.pinned_until, until)

    def sleep(self, cont_id: str) -> None:
        """Move off GPU but keep named extents (LMCache-shaped)."""
        with self._lock:
            rec = self._recs.get(cont_id)
            if rec is None:
                return
            rec.on_gpu = False
            for e in rec.extents:
                e.on_gpu = False

    def wake(self, cont_id: str) -> None:
        with self._lock:
            rec = self._recs.get(cont_id)
            if rec is None:
                return
            rec.on_gpu = True
            for e in rec.extents:
                e.on_gpu = True

    def drop_if_unpinned(self, cont_id: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            rec = self._recs.get(cont_id)
            if rec is None:
                return False
            if rec.pinned_until > now:
                return False
            self._current -= rec.nbytes
            del self._recs[cont_id]
            return True

    def bytes_for(self, cont_id: str) -> int:
        rec = self._recs.get(cont_id)
        return rec.nbytes if rec else 0

    def gpu_bytes(self) -> int:
        with self._lock:
            return sum(r.nbytes for r in self._recs.values() if r.on_gpu)

    @property
    def high_water(self) -> int:
        return self._high_water

    @property
    def current_bytes(self) -> int:
        return self._current

    def extents(self, cont_id: str) -> list[KVExtent]:
        rec = self._recs.get(cont_id)
        return list(rec.extents) if rec else []
