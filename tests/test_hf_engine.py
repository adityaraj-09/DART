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
    finally:
        await rt.aclose()
