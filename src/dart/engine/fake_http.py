"""In-process fake vLLM / llama.cpp HTTP engines.

Used to measure *their* kernel launches (each generate/completion POST)
without a GPU. A real vLLM/llama.cpp process is a drop-in at the same paths.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from dart.engine.synthetic import _WORDS

_LOCK = threading.Lock()


class FakeCounters:
    def __init__(self) -> None:
        self.kernel_launches = 0
        self.tokens_predicted = 0
        self.prompt_tokens = 0
        self.prefix_cache_hits = 0
        self.seen_prefixes: set[str] = set()

    def snapshot(self) -> dict[str, int]:
        return {
            "kernel_launches": self.kernel_launches,
            "tokens_predicted": self.tokens_predicted,
            "prompt_tokens": self.prompt_tokens,
            "prefix_cache_hits": self.prefix_cache_hits,
        }


def _next_text(prompt: str, n: int, seed: int = 0) -> str:
    n = max(1, n)
    h = hashlib.sha256(f"{seed}:{prompt}".encode()).digest()
    start = int.from_bytes(h[:4], "little")
    words = [_WORDS[(start + i) % len(_WORDS)] for i in range(n)]
    return " ".join(words)


class VLLMCompletionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str = "fake"
    prompt: str | list[str] | None = None
    messages: list[dict[str, str]] | None = None
    max_tokens: int = 16
    temperature: float = 0.8
    extra_body: dict[str, Any] | None = None
    structured_outputs: dict[str, Any] | None = None


class LlamaCompletionRequest(BaseModel):
    prompt: str = ""
    n_predict: int = 16
    cache_prompt: bool = True
    temperature: float = 0.8
    json_schema: dict[str, Any] | None = None
    grammar: str | None = None


def create_fake_vllm_app(*, seed: int = 0, counters: FakeCounters | None = None) -> FastAPI:
    """OpenAI-compatible vLLM stand-in. Each POST with max_tokens>0 is a kernel launch."""

    app = FastAPI(title="fake-vllm")
    ctr = counters or FakeCounters()
    app.state.counters = ctr

    def _run(prompt: str, max_tokens: int, json_mode: bool) -> tuple[str, int, int]:
        prompt_n = max(1, len(prompt.split()) + 4)
        with _LOCK:
            if prompt in ctr.seen_prefixes:
                ctr.prefix_cache_hits += 1
            ctr.seen_prefixes.add(prompt)
            ctr.kernel_launches += 1
            ctr.prompt_tokens += prompt_n
        if json_mode:
            text = '{"status":"ok","id":7}'
            pred = 8
        else:
            text = _next_text(prompt, max_tokens, seed=seed)
            pred = max(1, len(text.split()))
        with _LOCK:
            ctr.tokens_predicted += pred
        return text, prompt_n, pred

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, **ctr.snapshot()}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        s = ctr.snapshot()
        return (
            "# HELP vllm_engine_forward_calls_total Decode/prefill forwards in the engine process.\n"
            "# TYPE vllm_engine_forward_calls_total counter\n"
            f"vllm_engine_forward_calls_total {s['kernel_launches']}\n"
            "# HELP vllm_generation_tokens_total Tokens emitted by the engine.\n"
            "# TYPE vllm_generation_tokens_total counter\n"
            f"vllm_generation_tokens_total {s['tokens_predicted']}\n"
            "# HELP vllm_prefix_cache_hits_total Prefix-cache hits.\n"
            "# TYPE vllm_prefix_cache_hits_total counter\n"
            f"vllm_prefix_cache_hits_total {s['prefix_cache_hits']}\n"
        )

    @app.post("/v1/completions")
    async def completions(req: VLLMCompletionRequest) -> dict[str, Any]:
        prompt = req.prompt if isinstance(req.prompt, str) else " ".join(req.prompt or [])
        json_mode = bool(req.structured_outputs)
        text, prompt_n, pred = _run(prompt, req.max_tokens, json_mode)
        return {
            "id": "cmpl-fake",
            "object": "text_completion",
            "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": prompt_n,
                "completion_tokens": pred,
                "total_tokens": prompt_n + pred,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat(req: VLLMCompletionRequest, raw: Request) -> dict[str, Any]:
        body = await raw.json()
        messages = body.get("messages") or req.messages or []
        prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        json_mode = bool(body.get("structured_outputs") or req.structured_outputs)
        extra = body.get("extra_body") or {}
        if extra.get("structured_outputs"):
            json_mode = True
        text, prompt_n, pred = _run(prompt, int(body.get("max_tokens", req.max_tokens)), json_mode)
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_n,
                "completion_tokens": pred,
                "total_tokens": prompt_n + pred,
            },
        }

    @app.get("/v1/engine/counters")
    def counters() -> dict[str, int]:
        return ctr.snapshot()

    return app


def create_fake_llamacpp_app(*, seed: int = 0, counters: FakeCounters | None = None) -> FastAPI:
    """llama.cpp /completion stand-in. Each POST is a kernel launch."""

    app = FastAPI(title="fake-llamacpp")
    ctr = counters or FakeCounters()
    app.state.counters = ctr

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, **ctr.snapshot()}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        s = ctr.snapshot()
        return (
            "# HELP llamacpp_decode_calls_total /completion forwards.\n"
            "# TYPE llamacpp_decode_calls_total counter\n"
            f"llamacpp_decode_calls_total {s['kernel_launches']}\n"
            f"llamacpp_tokens_predicted {s['tokens_predicted']}\n"
        )

    @app.post("/completion")
    def completion(req: LlamaCompletionRequest) -> dict[str, Any]:
        prompt_n = max(1, len(req.prompt.split()) + 4)
        with _LOCK:
            if req.cache_prompt and req.prompt in ctr.seen_prefixes:
                ctr.prefix_cache_hits += 1
            ctr.seen_prefixes.add(req.prompt)
            ctr.kernel_launches += 1
            ctr.prompt_tokens += prompt_n
        json_mode = req.json_schema is not None or req.grammar is not None
        if json_mode:
            text = '{"status":"ok","id":7}'
            pred = 8
        else:
            text = _next_text(req.prompt, req.n_predict, seed=seed)
            pred = max(1, len(text.split()))
        with _LOCK:
            ctr.tokens_predicted += pred
        return {
            "content": text,
            "stop": False,
            "tokens_evaluated": prompt_n,
            "tokens_predicted": pred,
            "timings": {"predicted_n": pred, "prompt_n": prompt_n},
        }

    @app.get("/engine/counters")
    def counters() -> dict[str, int]:
        return ctr.snapshot()

    return app


# Satisfy type checkers for unused Field import in some pydantic versions.
_ = Field
