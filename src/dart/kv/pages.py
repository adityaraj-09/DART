"""Named KV as GPU-shaped pages — the handover data plane.

Control-plane metadata (EngineState, lease, pos) is small. The object that
moves between machines is a list of pages keyed by ``kv_root``. Adopt
installs those pages; it does not call ``engine.prefill``.

When NIXL or LMCache is importable, pages go through those libraries
(RDMA / GPU cache). Otherwise DART still transfers *pages* (byte buffers
with Merkle digests), not a deepcopy of the continuation object.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from dart.core.types import EngineState, KVExtent


def _hex_bytes(text: str) -> bytes:
    try:
        return bytes.fromhex(text)
    except ValueError:
        return text.encode()


@dataclass
class KVPage:
    digest: str
    nbytes: int
    payload: bytes
    layer: int = 0
    block_id: int = 0
    on_gpu: bool = True

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "digest": self.digest,
            "nbytes": self.nbytes,
            "layer": self.layer,
            "block_id": self.block_id,
            "on_gpu": self.on_gpu,
            "payload_len": len(self.payload),
        }


def pages_from_state(state: EngineState) -> list[KVPage]:
    """Build transfer pages from extents. Payload is the page, not EngineState."""
    pages: list[KVPage] = []
    for ext in state.kv_extents:
        pages.append(page_from_extent(ext))
    if not pages:
        raw = _hex_bytes(state.kv_root)
        pages.append(
            KVPage(
                digest=state.kv_root,
                nbytes=max(64, len(raw)),
                payload=raw[:64].ljust(64, b"\0"),
                on_gpu=True,
            )
        )
    return pages


def page_from_extent(ext: KVExtent) -> KVPage:
    raw = _hex_bytes(ext.digest)
    # Cap in-process payload so synthetic accounting (nbytes) does not
    # allocate full GPU tensors. Real LMCache/NIXL replace this buffer.
    cap = min(max(ext.nbytes, len(raw)), 4096)
    payload = raw[:cap].ljust(cap, b"\0")
    return KVPage(
        digest=ext.digest,
        nbytes=ext.nbytes,
        payload=payload,
        layer=ext.layer,
        block_id=ext.block_id,
        on_gpu=ext.on_gpu,
    )


@dataclass
class PagePool:
    """In-process page table keyed by kv_root."""

    _pages: dict[str, list[KVPage]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    puts: int = 0
    gets: int = 0
    hits: int = 0
    misses: int = 0
    transfers: int = 0
    page_bytes_moved: int = 0

    def put(self, kv_root: str, pages: list[KVPage]) -> None:
        with self._lock:
            self._pages[kv_root] = [KVPage(**{**p.__dict__}) for p in pages]
            self.puts += 1

    def get(self, kv_root: str) -> list[KVPage] | None:
        with self._lock:
            self.gets += 1
            found = self._pages.get(kv_root)
            if found is None:
                self.misses += 1
                return None
            self.hits += 1
            return [KVPage(**{**p.__dict__}) for p in found]

    def transfer(self, kv_root: str) -> list[KVPage]:
        pages = self.get(kv_root)
        if pages is None:
            return []
        with self._lock:
            self.transfers += 1
            self.page_bytes_moved += sum(p.nbytes for p in pages)
        return pages

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._pages),
                "puts": self.puts,
                "gets": self.gets,
                "hits": self.hits,
                "misses": self.misses,
                "transfers": self.transfers,
                "page_bytes_moved": self.page_bytes_moved,
                "pages": sum(len(v) for v in self._pages.values()),
            }


def try_lmcache_store(kv_root: str, pages: list[KVPage]) -> bool:
    """Put pages into a real LMCache GPU store. False if unavailable."""
    try:
        import lmcache  # noqa: F401
    except ImportError:
        return False
    store = getattr(lmcache, "store", None)
    if callable(store):
        try:
            store(kv_root, [p.payload for p in pages])
            return True
        except Exception:
            return False
    builder = getattr(lmcache, "LMCacheEngineBuilder", None)
    if builder is None:
        try:
            from lmcache.experimental.cache_engine import LMCacheEngineBuilder as builder
        except ImportError:
            builder = None
    if builder is None:
        return False
    try:
        engine = builder.get("dart") if hasattr(builder, "get") else None
        if engine is None:
            return False
        engine.store(kv_root, [p.payload for p in pages])
        return True
    except Exception:
        return False


def try_lmcache_retrieve(kv_root: str) -> list[KVPage] | None:
    try:
        import lmcache  # noqa: F401
    except ImportError:
        return None
    retrieve = getattr(lmcache, "retrieve", None)
    if not callable(retrieve):
        return None
    try:
        payloads = retrieve(kv_root)
    except Exception:
        return None
    if not payloads:
        return None
    return [
        KVPage(digest=f"{kv_root}:{i}", nbytes=len(buf), payload=bytes(buf))
        for i, buf in enumerate(payloads)
    ]


def try_nixl_rdma(pages: list[KVPage], *, src: str, dst: str) -> bool:
    """Move page payloads through NIXL. False if the library cannot transfer."""
    try:
        import nixl  # noqa: F401
    except ImportError:
        return False
    xfer = getattr(nixl, "transfer", None) or getattr(nixl, "rdma_transfer", None)
    if callable(xfer):
        try:
            xfer([p.payload for p in pages], src=src, dst=dst)
            return True
        except Exception:
            return False
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError:
        return False
    try:
        agent = nixl_agent(f"dart-{src or 'src'}-{dst or 'dst'}", nixl_agent_config())
        post = getattr(agent, "transfer", None) or getattr(agent, "initialize_xfer", None)
        if not callable(post):
            return False
        post([p.payload for p in pages])
        return True
    except Exception:
        return False
