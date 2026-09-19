"""KV handover connectors: LMCache GPU pages and NIXL RDMA.

Handover moves *pages* keyed by ``kv_root``. The destination adopts that
root and never calls ``engine.prefill``. Control-plane metadata (pos,
sampler, lease) rides along; the data plane is ``KVPage`` payloads.

When ``lmcache`` / ``nixl`` are installed, put/get/transfer go through
those libraries. Without them, DART still transfers page buffers — not a
deepcopy of the continuation object.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from dart.core.errors import HandoverError
from dart.core.types import EngineState
from dart.kv.pages import (
    KVPage,
    PagePool,
    pages_from_state,
    try_lmcache_retrieve,
    try_lmcache_store,
    try_nixl_rdma,
)


class KVBlob(BaseModel):
    """Named KV payload a producer can install without re-prefill."""

    kv_root: str
    model_hash: str
    cont_id: str
    state: EngineState
    nbytes: int = 0
    holder_id: str = ""
    lease: str = ""
    segment_index: int = 0
    output_text: str = ""
    prompt_text: str = ""
    max_tokens: int = 0
    created_at: float = Field(default_factory=time.time)
    page_count: int = 0
    transport: str = ""

    @classmethod
    def from_state(
        cls,
        *,
        kv_root: str,
        model_hash: str,
        cont_id: str,
        state: EngineState,
        holder_id: str,
        lease: str = "",
        segment_index: int = 0,
        output_text: str = "",
        prompt_text: str = "",
        max_tokens: int = 0,
    ) -> KVBlob:
        snap = state.model_copy(deep=True)
        pages = pages_from_state(snap)
        nbytes = sum(p.nbytes for p in pages) or sum(e.nbytes for e in snap.kv_extents)
        return cls(
            kv_root=kv_root,
            model_hash=model_hash,
            cont_id=cont_id,
            state=snap,
            nbytes=nbytes,
            holder_id=holder_id,
            lease=lease,
            segment_index=segment_index,
            output_text=output_text,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            page_count=len(pages),
        )


class TransferRecord(BaseModel):
    kv_root: str
    src: str
    dst: str
    nbytes: int
    transport: str
    at: float = Field(default_factory=time.time)


@runtime_checkable
class KVConnector(Protocol):
    name: str

    async def put(self, blob: KVBlob) -> None: ...

    async def get(self, kv_root: str) -> KVBlob | None: ...

    async def has(self, kv_root: str) -> bool: ...

    async def transfer(self, kv_root: str, *, src: str, dst: str) -> KVBlob: ...

    def metrics(self) -> dict[str, Any]: ...


def _try_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


class MemoryKVConnector:
    """In-process page table. Default path for tests and a single DartRuntime."""

    name = "memory"

    def __init__(self) -> None:
        self._blobs: dict[str, KVBlob] = {}
        self.pages = PagePool()
        self._lock = threading.Lock()
        self.puts = 0
        self.gets = 0
        self.hits = 0
        self.misses = 0
        self.transfers = 0
        self.bytes_moved = 0
        self.errors = 0
        self.history: list[TransferRecord] = []

    def _pages_for(self, blob: KVBlob) -> list[KVPage]:
        return pages_from_state(blob.state)

    async def put(self, blob: KVBlob) -> None:
        pages = self._pages_for(blob)
        blob = blob.model_copy(update={"page_count": len(pages), "nbytes": blob.nbytes or sum(p.nbytes for p in pages)})
        self.pages.put(blob.kv_root, pages)
        with self._lock:
            self._blobs[blob.kv_root] = blob.model_copy(deep=True)
            self.puts += 1

    async def get(self, kv_root: str) -> KVBlob | None:
        with self._lock:
            self.gets += 1
            item = self._blobs.get(kv_root)
            if item is None:
                self.misses += 1
                return None
            self.hits += 1
            return item.model_copy(deep=True)

    async def has(self, kv_root: str) -> bool:
        with self._lock:
            return kv_root in self._blobs

    async def transfer(self, kv_root: str, *, src: str, dst: str) -> KVBlob:
        blob = await self.get(kv_root)
        if blob is None:
            self.errors += 1
            raise HandoverError(f"no KV blob for {kv_root[:16]}")
        moved_pages = self.pages.transfer(kv_root)
        if not moved_pages:
            moved_pages = self._pages_for(blob)
            self.pages.put(kv_root, moved_pages)
        moved = blob.model_copy(update={"holder_id": dst, "page_count": len(moved_pages), "transport": self.name})
        await self.put(moved)
        with self._lock:
            self.transfers += 1
            self.bytes_moved += moved.nbytes
            self.history.append(
                TransferRecord(
                    kv_root=kv_root,
                    src=src,
                    dst=dst,
                    nbytes=moved.nbytes,
                    transport=self.name,
                )
            )
        return moved

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "entries": len(self._blobs),
                "puts": self.puts,
                "gets": self.gets,
                "hits": self.hits,
                "misses": self.misses,
                "transfers": self.transfers,
                "bytes_moved": self.bytes_moved,
                "errors": self.errors,
                "rdma_available": False,
                "pages": self.pages.metrics(),
            }


class FileKVConnector(MemoryKVConnector):
    """Directory-backed blobs so a second process can pull KV by kv_root."""

    name = "file"

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._load()

    def _path(self, kv_root: str) -> Path:
        return self.root / f"{kv_root}.json"

    def _load(self) -> None:
        for p in self.root.glob("*.json"):
            try:
                blob = KVBlob.model_validate_json(p.read_text())
                self._blobs[blob.kv_root] = blob
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    async def put(self, blob: KVBlob) -> None:
        await super().put(blob)
        path = self._path(blob.kv_root)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(blob.model_dump_json())
        tmp.replace(path)

    async def get(self, kv_root: str) -> KVBlob | None:
        got = await super().get(kv_root)
        if got is not None:
            return got
        path = self._path(kv_root)
        if not path.exists():
            return None
        try:
            blob = KVBlob.model_validate_json(path.read_text())
        except (OSError, ValueError):
            self.errors += 1
            return None
        with self._lock:
            self._blobs[kv_root] = blob
            self.hits += 1
        return blob.model_copy(deep=True)


class LMCacheConnector(MemoryKVConnector):
    """LMCache GPU pages keyed by kv_root.

    When ``lmcache`` is installed, put/get store page payloads in that
    cache. Adopt still installs ``kv_root`` with zero ``engine.prefill``.
    """

    name = "lmcache"

    def __init__(self) -> None:
        super().__init__()
        self.lmcache_available = _try_import("lmcache")
        self.gpu_page_puts = 0
        self.gpu_page_hits = 0

    async def put(self, blob: KVBlob) -> None:
        pages = self._pages_for(blob)
        if try_lmcache_store(blob.kv_root, pages):
            self.gpu_page_puts += 1
        await super().put(blob)

    async def get(self, kv_root: str) -> KVBlob | None:
        gpu = try_lmcache_retrieve(kv_root)
        if gpu:
            self.gpu_page_hits += 1
            self.pages.put(kv_root, gpu)
        return await super().get(kv_root)

    def metrics(self) -> dict[str, Any]:
        m = super().metrics()
        m["lmcache_available"] = self.lmcache_available
        m["gpu_page_puts"] = self.gpu_page_puts
        m["gpu_page_hits"] = self.gpu_page_hits
        m["transport"] = "lmcache-gpu" if self.lmcache_available else "lmcache-pages"
        return m


class NixlConnector:
    """NIXL RDMA handover of named KV pages.

    Real NIXL posts page payloads over RDMA. Without the library, transfer
    still moves ``KVPage`` buffers (not a continuation memcpy) through
    ``backend``. Adopt on the destination is zero ``engine.prefill``.
    """

    name = "nixl"

    def __init__(self, *, backend: KVConnector | None = None) -> None:
        self.backend: KVConnector = backend or MemoryKVConnector()
        self.rdma_available = _try_import("nixl")
        self.rdma_transfers = 0
        self.puts = 0
        self.gets = 0
        self.hits = 0
        self.misses = 0
        self.transfers = 0
        self.bytes_moved = 0
        self.errors = 0
        self.history: list[TransferRecord] = []
        self._lock = threading.Lock()

    def _transport(self, rdma: bool) -> str:
        if rdma:
            return "nixl-rdma"
        return "nixl-pages"

    async def put(self, blob: KVBlob) -> None:
        await self.backend.put(blob)
        with self._lock:
            self.puts += 1

    async def get(self, kv_root: str) -> KVBlob | None:
        blob = await self.backend.get(kv_root)
        with self._lock:
            self.gets += 1
            if blob is None:
                self.misses += 1
            else:
                self.hits += 1
        return blob

    async def has(self, kv_root: str) -> bool:
        return await self.backend.has(kv_root)

    async def transfer(self, kv_root: str, *, src: str, dst: str) -> KVBlob:
        try:
            blob = await self.backend.transfer(kv_root, src=src, dst=dst)
        except HandoverError:
            with self._lock:
                self.errors += 1
            raise
        pages = pages_from_state(blob.state)
        rdma = False
        if self.rdma_available:
            rdma = try_nixl_rdma(pages, src=src, dst=dst)
        transport = self._transport(rdma)
        rec = TransferRecord(
            kv_root=kv_root,
            src=src,
            dst=dst,
            nbytes=blob.nbytes,
            transport=transport,
        )
        with self._lock:
            self.transfers += 1
            if rdma:
                self.rdma_transfers += 1
            self.bytes_moved += blob.nbytes
            self.history.append(rec)
        return blob.model_copy(update={"holder_id": dst, "page_count": len(pages), "transport": transport})

    def metrics(self) -> dict[str, Any]:
        inner = self.backend.metrics() if hasattr(self.backend, "metrics") else {}
        with self._lock:
            return {
                "name": self.name,
                "rdma_available": self.rdma_available,
                "transport": self._transport(self.rdma_available and self.rdma_transfers > 0),
                "puts": self.puts,
                "gets": self.gets,
                "hits": self.hits,
                "misses": self.misses,
                "transfers": self.transfers,
                "rdma_transfers": self.rdma_transfers,
                "bytes_moved": self.bytes_moved,
                "errors": self.errors,
                "backend": inner,
            }


def build_connector(
    kind: str | None = "memory",
    *,
    path: str | Path | None = None,
) -> KVConnector:
    kind = (kind or "memory").lower()
    if kind in {"memory", "mem", "local"}:
        return MemoryKVConnector()
    if kind in {"file", "cas", "disk"}:
        if not path:
            raise ValueError("file KV connector requires path")
        return FileKVConnector(path)
    if kind in {"lmcache", "lmc"}:
        return LMCacheConnector()
    if kind in {"nixl", "rdma", "handover"}:
        backend: KVConnector
        if path:
            backend = FileKVConnector(path)
        else:
            backend = MemoryKVConnector()
        return NixlConnector(backend=backend)
    raise ValueError(f"unknown KV connector {kind!r}; use memory | file | lmcache | nixl")
