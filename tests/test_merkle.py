from __future__ import annotations

from dart.cip.merkle import extent_digest, merkle_root, root_from_extents
from dart.core.types import KVExtent


def test_merkle_empty_and_pair() -> None:
    a = merkle_root([])
    b = merkle_root([b"x"])
    c = merkle_root([b"x", b"y"])
    assert a != b != c
    assert len(a) == 64


def test_root_changes_when_extent_changes() -> None:
    e1 = KVExtent(layer=0, block_id=0, digest=extent_digest(0, 0, 1, [1]), nbytes=8)
    e2 = KVExtent(layer=0, block_id=0, digest=extent_digest(0, 0, 2, [1, 2]), nbytes=8)
    assert root_from_extents([e1]) != root_from_extents([e2])
