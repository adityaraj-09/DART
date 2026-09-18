"""Merkle identity over KV extents.

This is identity, not a ZK proof: any compatible producer that can fetch
the named extents can resume. The root changes every committed segment
without rewriting historical Data names (those stay bound to the old root).
"""

from __future__ import annotations

import hashlib

from dart.types import KVExtent


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return _h(b"empty").hex()
    layer = [_h(x) if len(x) != 32 else x for x in leaves]
    while len(layer) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(layer), 2):
            a = layer[i]
            b = layer[i + 1] if i + 1 < len(layer) else a
            nxt.append(_h(a + b))
        layer = nxt
    return layer[0].hex()


def extent_digest(layer: int, block_id: int, pos: int, token_ids: list[int]) -> str:
    h = hashlib.sha256()
    h.update(layer.to_bytes(4, "little"))
    h.update(block_id.to_bytes(4, "little"))
    h.update(pos.to_bytes(8, "little"))
    h.update(bytes(str(token_ids).encode()))
    return h.hexdigest()


def root_from_extents(extents: list[KVExtent]) -> str:
    leaves = [
        bytes.fromhex(e.digest) if len(e.digest) == 64 else _h(e.digest.encode())
        for e in sorted(extents, key=lambda x: (x.layer, x.block_id))
    ]
    return merkle_root(leaves)


def bytes_per_extent(n_kv_heads: int, head_dim: int, block_size: int, dtype_nbytes: int = 2) -> int:
    # K and V
    return 2 * n_kv_heads * head_dim * block_size * dtype_nbytes
