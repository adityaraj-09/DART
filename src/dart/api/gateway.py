"""HTTP/WebSocket surface: CIP + OpenAI-compatible facade.

The OpenAI path is a translation layer. Disconnect and `X-Dart-Pace`
become Interests; they actually stop the decode kernel.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from dart.core.errors import AmplificationError, DartError, HandoverError, InterestNack, LeaseError, PinMissError
from dart.cip.protocol import CipMessage, CipName, Interest
from dart.core.runtime import ContinuationHandle, DartRuntime
from dart.core.types import ChatMessage, InterestKind, Prompt

STATIC = Path(__file__).parent / "static"


class OpenRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str | None = None
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 256
    temperature: float = 0.8


class InterestBody(BaseModel):
    name: str | None = None
    window: int = 16
    lifetime_ms: int = 2000
    locator: str = "default"
    kind: InterestKind = InterestKind.TOKENS
    grammar_span: str | None = None
    lease: str | None = None
    segment: int | None = None
    stream: bool = False


class AckBody(BaseModel):
    consumed: int
    lease: str | None = None


class HandoverBody(BaseModel):
    lease: str
    kv_root: str | None = None
    from_node: str = ""
    cont_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str = "dart-synth-8b"
    messages: list[dict[str, str]]
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.8


def create_app(runtime: DartRuntime, *, router: Any | None = None) -> FastAPI:
    app = FastAPI(
        title="DART",
        description="Demand-Addressed Runtime Tokens — Interest-Driven Decode",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = runtime
    app.state.router = router

    if STATIC.exists():
        app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        index = STATIC / "index.html"
        if index.exists():
            return index.read_text()
        return "<h1>DART</h1><p>Runtime is up. POST /v1/chat/completions</p>"

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "engine": runtime.engine.model_id,
            "engine_class": type(runtime.engine).__name__,
            "continuations": len(runtime._conts),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return runtime.prometheus()

    @app.get("/v1/metrics")
    async def metrics_json() -> dict[str, Any]:
        return runtime.metrics_snapshot()

    @app.get("/v1/engine")
    async def engine_probe() -> dict[str, Any]:
        return await runtime.engine_probe()

    @app.get("/v1/mesh")
    async def mesh_status() -> dict[str, Any]:
        if router is None:
            return {
                "nodes": [
                    {
                        "node_id": runtime.config.producer_id,
                        "continuations": len(runtime._conts),
                        "pins": runtime.pin_pool.metrics(),
                    }
                ],
                "single_node": True,
            }
        return router.status()

    @app.post("/v1/mesh/interest")
    async def mesh_interest(body: InterestBody) -> dict[str, Any]:
        if not body.name:
            raise HTTPException(400, "name required")
        if not body.lease:
            raise HTTPException(400, "lease required")
        req = Interest(
            name=body.name,
            window=body.window,
            lifetime_ms=body.lifetime_ms,
            locator=body.locator,
            kind=body.kind,
            grammar_span=body.grammar_span,
            lease=body.lease,
        )
        try:
            if router is not None:
                data, decision = await router.route(req, lease=body.lease)
            else:
                data = await runtime.interest(req, lease=body.lease)
                decision = None
        except (LeaseError, AmplificationError, InterestNack, PinMissError, HandoverError) as exc:
            raise HTTPException(400, str(exc)) from exc
        payload = json.loads(data.model_dump_json())
        payload["token_count"] = data.token_count()
        if decision is not None:
            payload["route"] = decision.as_dict()
        return payload

    @app.post("/v1/handover")
    async def handover(body: HandoverBody) -> dict[str, Any]:
        try:
            handle = await runtime.adopt(
                lease=body.lease, kv_root=body.kv_root, from_node=body.from_node
            )
        except (LeaseError, PinMissError, InterestNack, HandoverError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {**handle.dump(), "adopted": True, "prefill_skipped": True}

    @app.get("/v1/kv")
    async def kv_lookup(root: str) -> dict[str, Any]:
        rec = runtime.pin_pool.lookup(root)
        if rec is not None:
            return {"source": "pin", **rec.as_dict()}
        if runtime.connector is not None:
            blob = await runtime.connector.get(root)
            if blob is not None:
                return {
                    "source": "connector",
                    "kv_root": blob.kv_root,
                    "holder_id": blob.holder_id,
                    "nbytes": blob.nbytes,
                    "cont_id": blob.cont_id,
                    "pos": blob.state.pos,
                }
        raise HTTPException(404, f"no KV for {root}")

    @app.get("/v1/cas")
    async def cas_get(name: str) -> dict[str, Any]:
        data = runtime.get_named(name)
        if data is None:
            raise HTTPException(404, f"CAS miss: {name}")
        return json.loads(data.model_dump_json())

    @app.post("/v1/peer/interest")
    async def peer_interest(body: InterestBody) -> dict[str, Any]:
        if not body.name:
            raise HTTPException(400, "name required")
        try:
            data = await runtime.satisfy_named(body.name)
        except InterestNack as exc:
            raise HTTPException(404, str(exc)) from exc
        return json.loads(data.model_dump_json())

    @app.post("/v1/continuations")
    async def open_cont(body: OpenRequest) -> dict[str, Any]:
        prompt: Prompt | list[ChatMessage] | str
        if body.messages:
            prompt = body.messages
        elif body.prompt:
            prompt = body.prompt
        else:
            raise HTTPException(400, "prompt or messages required")
        opener = router.open if router is not None else runtime.open
        handle = await opener(
            prompt, max_tokens=body.max_tokens, temperature=body.temperature, model=body.model
        )
        return handle.dump()

    @app.post("/v1/continuations/{cont_id}/interest")
    async def post_interest(
        cont_id: str,
        body: InterestBody,
        x_dart_lease: str | None = Header(default=None, alias="X-Dart-Lease"),
    ) -> dict[str, Any]:
        cont = _cont_or_404(runtime, cont_id)
        lease = body.lease or x_dart_lease or cont.signed
        name = body.name
        if not name:
            if body.kind is InterestKind.GRAMMAR:
                span = body.grammar_span or "next-value"
                name = CipName.grammar(cont.model_hash, cont.state.kv_root, span).render()
            else:
                seg = body.segment if body.segment is not None else cont.segment_index
                name = CipName.tokens(cont.model_hash, cont.state.kv_root, seg).render()
        req = Interest(
            name=name,
            window=body.window,
            lifetime_ms=body.lifetime_ms,
            locator=body.locator,
            kind=body.kind,
            grammar_span=body.grammar_span,
            cont_id=cont_id,
            lease=lease,
        )

        async def _issue() -> Any:
            if router is not None:
                data, _decision = await router.route(req, lease=lease)
                return data
            return await runtime.interest(req, lease=lease)

        if body.stream:
            async def events() -> AsyncIterator[bytes]:
                tap: asyncio.Queue[str] = asyncio.Queue()
                cont.token_listeners.append(tap)
                task = asyncio.create_task(_issue())
                try:
                    while not task.done():
                        try:
                            piece = await asyncio.wait_for(tap.get(), timeout=0.05)
                        except TimeoutError:
                            continue
                        yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n".encode()
                    while not tap.empty():
                        piece = tap.get_nowait()
                        yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n".encode()
                    data = await task
                    payload = json.loads(data.model_dump_json())
                    payload["type"] = "done"
                    payload["token_count"] = data.token_count()
                    yield f"data: {json.dumps(payload)}\n\n".encode()
                except (LeaseError, AmplificationError, InterestNack, PinMissError, HandoverError) as exc:
                    yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n".encode()
                except DartError as exc:
                    yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n".encode()
                finally:
                    if tap in cont.token_listeners:
                        cont.token_listeners.remove(tap)

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            data = await _issue()
        except (LeaseError, AmplificationError, InterestNack, PinMissError, HandoverError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except DartError as exc:
            raise HTTPException(500, str(exc)) from exc
        payload = json.loads(data.model_dump_json())
        payload["token_count"] = data.token_count()
        return payload

    @app.post("/v1/continuations/{cont_id}/ack")
    async def post_ack(cont_id: str, body: AckBody) -> dict[str, str]:
        try:
            await runtime.ack(cont_id, body.consumed, lease=body.lease)
        except LeaseError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok"}

    @app.delete("/v1/continuations/{cont_id}")
    async def delete_cont(cont_id: str) -> dict[str, str]:
        try:
            await runtime.close(cont_id)
        except LeaseError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "closed"}

    @app.get("/v1/continuations/{cont_id}")
    async def get_cont(cont_id: str) -> dict[str, Any]:
        cont = _cont_or_404(runtime, cont_id)
        snap = runtime.metrics_snapshot()
        item = next((i for i in snap["items"] if i["cont_id"] == cont_id), None)
        return {
            "cont_id": cont_id,
            "kv_root": cont.state.kv_root,
            "pos": cont.state.pos,
            "done": cont.done,
            "sleeping": cont.sleeping,
            "output_text": cont.output_text,
            "metrics": item,
        }

    @app.websocket("/v1/cip")
    async def cip_ws(ws: WebSocket) -> None:
        await ws.accept()
        handle: ContinuationHandle | None = None
        try:
            while True:
                raw = await ws.receive_text()
                msg = CipMessage.model_validate_json(raw)
                if msg.type == "open":
                    handle = await runtime.open(Prompt(text=""), max_tokens=runtime.config.decode_quota)
                    await ws.send_json({"type": "open", **handle.dump()})
                    continue
                if msg.type == "close" and handle:
                    await runtime.close(handle.cont_id)
                    await ws.send_json({"type": "close", "cont_id": handle.cont_id})
                    continue
                if msg.type == "ack" and handle and msg.consumed:
                    await runtime.ack(handle.cont_id, msg.consumed, lease=handle.lease)
                    continue
                if msg.type == "interest" and msg.interest is not None:
                    lease = msg.interest.lease or msg.lease or (handle.lease if handle else None)
                    try:
                        data = await runtime.interest(msg.interest, lease=lease)
                        await ws.send_json({"type": "data", "data": json.loads(data.model_dump_json())})
                    except InterestNack as exc:
                        await ws.send_json(
                            {
                                "type": "nack",
                                "nack": {"name": msg.interest.name, "reason": exc.reason, "detail": exc.detail},
                            }
                        )
                else:
                    await ws.send_json({"type": "nack", "nack": {"reason": "unknown_name", "detail": "bad frame"}})
        except WebSocketDisconnect:
            if handle:
                await runtime.close(handle.cont_id)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
        pace = request.headers.get("x-dart-pace")
        window = int(request.headers.get("x-dart-window", "16"))
        handle = await runtime.open(
            [ChatMessage(role=m.get("role", "user"), content=m.get("content", "")) for m in body.messages],
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            model=body.model,
        )
        if not body.stream:
            text = await _drain(runtime, handle, request, window, pace, body.max_tokens)
            return {
                "id": f"chatcmpl-{handle.cont_id[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(runtime, handle),
            }

        async def events() -> AsyncIterator[bytes]:
            cid = f"chatcmpl-{handle.cont_id[:12]}"
            try:
                async for piece in _stream_segments(runtime, handle, request, window, pace, body.max_tokens):
                    chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                done = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                await runtime.close(handle.cont_id)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.on_event("startup")
    async def _up() -> None:
        if router is not None:
            await router.start()
        else:
            await runtime.start()

    @app.on_event("shutdown")
    async def _down() -> None:
        if router is not None:
            await router.aclose()
        else:
            await runtime.aclose()

    return app


def _cont_or_404(runtime: DartRuntime, cont_id: str) -> Any:
    try:
        return runtime.get(cont_id)
    except LeaseError as exc:
        raise HTTPException(404, str(exc)) from exc


async def _stream_segments(
    runtime: DartRuntime,
    handle: ContinuationHandle,
    request: Request,
    window: int,
    pace: str | None,
    max_tokens: int,
) -> AsyncIterator[str]:
    from dart.client.consumers import DrainPacer, ReadingPacer

    pacer: DrainPacer | ReadingPacer
    if pace:
        pacer = ReadingPacer(tokens_per_sec=float(pace), burst=window)
    else:
        pacer = DrainPacer(window=window)
    async for data in runtime.consume(handle, pacer, max_tokens=max_tokens):
        if await request.is_disconnected():
            break
        if data.text:
            yield data.text


async def _drain(
    runtime: DartRuntime,
    handle: ContinuationHandle,
    request: Request,
    window: int,
    pace: str | None,
    max_tokens: int,
) -> str:
    parts: list[str] = []
    async for piece in _stream_segments(runtime, handle, request, window, pace, max_tokens):
        parts.append(piece)
    return "".join(parts)


def _usage(runtime: DartRuntime, handle: ContinuationHandle) -> dict[str, int]:
    cont = runtime.get(handle.cont_id)
    return {
        "prompt_tokens": cont.metrics.prefill_tokens,
        "completion_tokens": cont.metrics.tokens_generated,
        "total_tokens": cont.metrics.prefill_tokens + cont.metrics.tokens_generated,
    }


def create_peer_app(cas_dir: str):
    """Second process: FileCAS only. No decode kernel."""
    from dart.engine.cache_only import CacheOnlyEngine
    from dart.kv.store import FileCAS
    from dart.core.types import RuntimeConfig

    cas = FileCAS(cas_dir)
    runtime = DartRuntime(CacheOnlyEngine(), RuntimeConfig(cas_dir=cas_dir), cas=cas)
    return create_app(runtime)
