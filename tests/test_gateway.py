from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from dart.engine.synthetic import SyntheticEngine
from dart.api.gateway import create_app
from dart.core.runtime import DartRuntime
from dart.core.types import RuntimeConfig


@pytest.fixture
async def client() -> AsyncClient:
    rt = DartRuntime(
        SyntheticEngine(seed=3),
        RuntimeConfig(poll_interval_s=0.001, decode_quota=64, interest_lifetime_s=3),
    )
    await rt.start()
    app = create_app(rt)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await rt.aclose()


async def test_health_and_index(client: AsyncClient) -> None:
    h = await client.get("/health")
    assert h.status_code == 200 and h.json()["ok"] is True
    page = await client.get("/")
    assert page.status_code == 200
    assert "Interest-Driven Decode" in page.text


async def test_openai_nonstream(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dart-synth-8b",
            "stream": False,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"x-dart-window": "8"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] > 0


async def test_openai_stream_and_metrics(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dart-synth-8b",
            "stream": True,
            "max_tokens": 12,
            "messages": [{"role": "user", "content": "stream"}],
        },
        headers={"x-dart-pace": "200", "x-dart-window": "8"},
    )
    assert r.status_code == 200
    text = r.text
    assert "data:" in text
    metrics = await client.get("/metrics")
    assert "dart_tokens_generated" in metrics.text
    js = await client.get("/v1/metrics")
    assert js.status_code == 200


async def test_cip_http_interest(client: AsyncClient) -> None:
    opened = await client.post(
        "/v1/continuations",
        json={"prompt": "hello continuation", "max_tokens": 24},
    )
    assert opened.status_code == 200, opened.text
    body = opened.json()
    interest = await client.post(
        f"/v1/continuations/{body['cont_id']}/interest",
        json={"window": 8, "lease": body["lease"]},
        headers={"X-Dart-Lease": body["lease"]},
    )
    assert interest.status_code == 200, interest.text
    data = interest.json()
    assert data["text"]
    assert data["kv_root"]
    closed = await client.delete(f"/v1/continuations/{body['cont_id']}")
    assert closed.status_code == 200
