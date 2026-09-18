from __future__ import annotations

import pytest

from dart.core.errors import NameParseError
from dart.cip.protocol import CipName, Interest
from dart.core.types import InterestKind, ModelConfig


def test_fingerprint_stable() -> None:
    a = ModelConfig(model_id="m", tokenizer_hash="t")
    b = ModelConfig(model_id="m", tokenizer_hash="t")
    c = ModelConfig(model_id="m", tokenizer_hash="other")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert len(a.fingerprint()) == 16


def test_name_roundtrip_tokens() -> None:
    n = CipName.tokens("ab" * 8, "cd" * 16, 3)
    s = n.render()
    p = CipName.parse(s)
    assert p.kind is InterestKind.TOKENS
    assert p.segment == 3
    assert p.render() == s


def test_name_grammar_and_kv() -> None:
    g = CipName.grammar("ab" * 8, "cd" * 16, "next-value")
    assert "grammar/span/next-value" in g.render()
    kv = CipName.parse(f"/cip/{'ab'*8}/{'cd'*16}/kv/layer/2/blk/9")
    assert kv.layer == 2 and kv.block == 9


def test_bad_name() -> None:
    with pytest.raises(NameParseError):
        CipName.parse("/not-cip/x")
    with pytest.raises(NameParseError):
        CipName.parse("/cip/zz/yy/nope")


def test_interest_infers_grammar_kind() -> None:
    name = CipName.grammar("ab" * 8, "cd" * 16, "tool-args").render()
    i = Interest(name=name, window=8)
    assert i.kind is InterestKind.GRAMMAR
    assert i.grammar_span == "tool-args"
