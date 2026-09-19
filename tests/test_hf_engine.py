from __future__ import annotations

import asyncio

from dart.client.consumers import DrainPacer
from dart.core.runtime import DartRuntime
from dart.core.types import Prompt, RuntimeConfig
from dart.engine.hf import DEFAULT_HF_MODEL, HuggingFaceEngine, resolve_hf_model_id
from dart.factory import build_engine


def test_resolve_default_model() -> None:
    assert resolve_hf_model_id(None) == DEFAULT_HF_MODEL
    assert resolve_hf_model_id("dart-synth-8b") == DEFAULT_HF_MODEL
    assert resolve_hf_model_id("org/other") == "org/other"


def test_factory_hf_kind() -> None:
    eng = build_engine("hf")
    assert isinstance(eng, HuggingFaceEngine)
    assert eng.model_id == DEFAULT_HF_MODEL


def test_hf_runtime_uses_long_interest_lifetime() -> None:
    from dart.factory import build_runtime

    rt = build_runtime("hf")
    assert rt.config.interest_lifetime_s >= 15.0
    assert isinstance(rt.engine, HuggingFaceEngine)


async def test_hf_stub_credit_gate() -> None:
    launches = {"n": 0}

    def tokenize(_prompt: Prompt) -> list[int]:
        return [7, 8, 9]

    def generate(_ids: list[int], n: int) -> tuple[list[int], str, bool]:
        launches["n"] += 1
        return list(range(n)), "A named continuation is a pull-based decode window.", False

    eng = HuggingFaceEngine(
        DEFAULT_HF_MODEL,
        tokenize_fn=tokenize,
        generate_fn=generate,
    )
    rt = DartRuntime(
        eng,
        RuntimeConfig(poll_interval_s=0.001, decode_quota=32, interest_lifetime_s=3),
    )
    await rt.start()
    try:
        handle = await rt.open("what is a continuation?", max_tokens=16)
        await asyncio.sleep(0.03)
        assert launches["n"] == 0
        texts: list[str] = []
        async for seg in rt.consume(handle, DrainPacer(window=8), max_tokens=16):
            texts.append(seg.text)
            if sum(len(t.split()) for t in texts) >= 4:
                break
        blob = " ".join(texts).lower()
        assert launches["n"] >= 1
        assert "continuation" in blob
        # Prefill encoded the prefix once; each page encodes only the new seed token.
        assert eng.last_encode_len == 1
        assert eng.tokens_encoded == len([7, 8, 9]) + launches["n"]
    finally:
        await rt.aclose()


async def test_hf_second_page_does_not_reencode_prefix() -> None:
    encoded: list[int] = []

    def tokenize(_prompt: Prompt) -> list[int]:
        return [1, 2, 3, 4, 5]

    def generate(ids: list[int], n: int) -> tuple[list[int], str, bool]:
        encoded.append(len(ids))
        return list(range(n)), "word " * n, False

    eng = HuggingFaceEngine(DEFAULT_HF_MODEL, tokenize_fn=tokenize, generate_fn=generate)
    rt = DartRuntime(
        eng,
        RuntimeConfig(poll_interval_s=0.001, decode_quota=24, segment_size=8, interest_lifetime_s=3),
    )
    await rt.start()
    try:
        handle = await rt.open("prefix stays in past_key_values", max_tokens=16)
        from dart.cip.protocol import CipName, Interest

        n0 = CipName.tokens(handle.model_hash, handle.kv_root, 0).render()
        d0 = await rt.interest(Interest(name=n0, window=8, lifetime_ms=2000, lease=handle.lease))
        n1 = CipName.tokens(handle.model_hash, d0.kv_root, 1).render()
        await rt.interest(Interest(name=n1, window=8, lifetime_ms=2000, lease=handle.lease))
        assert encoded == [1, 1]
        assert eng.stats.prefix_cache_hits >= 2
    finally:
        await rt.aclose()
