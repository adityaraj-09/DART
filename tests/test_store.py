from __future__ import annotations

from pathlib import Path

from dart.protocol import CipName, Data
from dart.store import FileCAS, KVStore, MemoryCAS
from dart.types import KVExtent


def test_memory_cas_hit_miss() -> None:
    cas = MemoryCAS()
    name = CipName.tokens("ab" * 8, "cd" * 16, 0).render()
    data = Data(name=name, text="hi", kv_root="ef" * 16, kv_root_prev="cd" * 16, pos=3)
    assert cas.get_data(name) is None
    cas.put_data(name, data)
    got = cas.get_data(name)
    assert got is not None and got.text == "hi"
    assert cas.hits == 1 and cas.misses == 1


def test_file_cas_survives_reload(tmp_path: Path) -> None:
    name = CipName.tokens("ab" * 8, "cd" * 16, 1).render()
    data = Data(name=name, text="seg", kv_root="ef" * 16, kv_root_prev="cd" * 16, pos=4)
    a = FileCAS(tmp_path)
    a.put_data(name, data)
    b = FileCAS(tmp_path)
    got = b.get_data(name)
    assert got is not None and got.text == "seg"


def test_kv_pin_sleep_high_water() -> None:
    kv = KVStore()
    extents = [KVExtent(layer=0, block_id=0, digest="aa" * 32, nbytes=100)]
    kv.replace("c", extents)
    assert kv.high_water == 100
    kv.sleep("c")
    assert kv.gpu_bytes() == 0
    kv.wake("c")
    assert kv.gpu_bytes() == 100
    kv.pin("c", until=1.0)
    assert kv.drop_if_unpinned("c", now=0) is False
    assert kv.drop_if_unpinned("c", now=10) is True
    assert kv.bytes_for("c") == 0
