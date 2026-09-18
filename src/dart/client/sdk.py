"""Python SDK: pacers issue Interests; callers never speak CIP unless they want to."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from dart.client.consumers import DrainPacer, JsonNeedPacer, Pacer, ReadingPacer, TtsPacer
from dart.cip.protocol import Data
from dart.core.runtime import ContinuationHandle, DartRuntime
from dart.core.types import ChatMessage


class DartClient:
    """In-process or HTTP client. Prefer in-process when embedding the runtime."""

    def __init__(
        self,
        runtime: DartRuntime | None = None,
        *,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        if runtime is None and not base_url:
            raise ValueError("runtime or base_url required")
        self.runtime = runtime
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout_s = timeout_s

    async def stream(
        self,
        messages: list[ChatMessage] | list[dict[str, str]] | str,
        *,
        consumer: Pacer | None = None,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.8,
    ) -> AsyncIterator[Data]:
        pacer: Pacer = consumer or DrainPacer()
        if self.runtime is not None:
            handle = await self.runtime.open(
                messages, max_tokens=max_tokens, temperature=temperature, model=model
            )
            async for data in self.runtime.consume(handle, pacer, max_tokens=max_tokens):
                yield data
            return
        assert self.base_url
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            body: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature, "model": model}
            if isinstance(messages, str):
                body["prompt"] = messages
            else:
                body["messages"] = [
                    m if isinstance(m, dict) else m.model_dump() for m in messages  # type: ignore[union-attr]
                ]
            opened = (await client.post(f"{self.base_url}/v1/continuations", json=body)).json()
            handle = ContinuationHandle.from_dump(opened)
            produced = 0
            seg = 0
            kv_root = handle.kv_root
            while produced < max_tokens:
                window = await pacer.next_window()
                kind = "tokens"
                payload: dict[str, Any] = {
                    "window": window,
                    "lease": handle.lease,
                    "segment": seg,
                    "kind": kind,
                }
                if getattr(pacer, "name", "") == "json":
                    payload["kind"] = "grammar"
                    payload["grammar_span"] = "next-value"
                resp = await client.post(
                    f"{self.base_url}/v1/continuations/{handle.cont_id}/interest",
                    json=payload,
                    headers={"X-Dart-Lease": handle.lease},
                )
                if resp.status_code >= 400:
                    break
                data = Data.model_validate(resp.json())
                produced += data.token_count()
                kv_root = data.kv_root
                handle.kv_root = kv_root
                seg += 1
                yield data
                if data.stopped:
                    break
            await client.delete(f"{self.base_url}/v1/continuations/{handle.cont_id}")


__all__ = [
    "DartClient",
    "DrainPacer",
    "JsonNeedPacer",
    "ReadingPacer",
    "TtsPacer",
]
