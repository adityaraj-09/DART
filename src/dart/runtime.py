"""Credit-gated continuation runtime.

Outstanding Interests are the only thing that may run a decode kernel.
The scheduler is a congestion controller over named continuations.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from dart.cc import CongestionController
from dart.engine.base import Engine
from dart.errors import AmplificationError, InterestNack, LeaseError
from dart.lease import ContinuationLease, sign_lease, verify_lease
from dart.protocol import CipName, Data, Interest, Nack, TokenSegment
from dart.store import FileCAS, KVStore, MemoryCAS
from dart.types import (
    ChatMessage,
    ContinuationMetrics,
    EngineState,
    InterestKind,
    NackReason,
    Prompt,
    RuntimeConfig,
)

logger = logging.getLogger("dart.runtime")


@dataclass
class Continuation:
    id: str
    lease: ContinuationLease
    signed: str
    cc: CongestionController
    state: EngineState
    model_hash: str
    metrics: ContinuationMetrics
    lock: asyncio.Lock
    pending: dict[str, asyncio.Future[Data | Nack]]
    wakeup: asyncio.Event
    max_tokens: int
    segment_index: int = 0
    output_text: str = ""
    done: bool = False
    sleeping: bool = False
    producer_id: str = "local-0"
    last_lifetime_s: float = 2.0
    created_at: float = field(default_factory=time.monotonic)
    last_interest_at: float = 0.0
    last_locator: str = "default"
    draft_ids: list[int] = field(default_factory=list)
    draft_text: str = ""
    snapshot: EngineState | None = None
    grammar_span: str | None = None
    prompt_text: str = ""
    tokens_since_ack: int = 0
    closed: bool = False
    last_skip_at: float = 0.0


class ContinuationHandle:
    def __init__(self, cont: Continuation) -> None:
        self.cont_id = cont.id
        self.lease = cont.signed
        self.model_hash = cont.model_hash
        self.kv_root = cont.state.kv_root
        self.pos = cont.state.pos
        self.model_id = cont.lease.model_id

    @classmethod
    def from_dump(cls, data: dict[str, Any]) -> ContinuationHandle:
        handle = cls.__new__(cls)
        handle.cont_id = data["cont_id"]
        handle.lease = data["lease"]
        handle.model_hash = data["model_hash"]
        handle.kv_root = data["kv_root"]
        handle.pos = int(data["pos"])
        handle.model_id = data.get("model_id", "")
        return handle

    def token_name(self, segment: int, kv_root: str | None = None) -> str:
        return CipName.tokens(self.model_hash, kv_root or self.kv_root, segment).render()

    def dump(self) -> dict[str, Any]:
        return {
            "cont_id": self.cont_id,
            "lease": self.lease,
            "model_hash": self.model_hash,
            "kv_root": self.kv_root,
            "pos": self.pos,
            "model_id": self.model_id,
        }


class DartRuntime:
    def __init__(
        self,
        engine: Engine,
        config: RuntimeConfig | None = None,
        *,
        cas: MemoryCAS | FileCAS | None = None,
        kv: KVStore | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or RuntimeConfig()
        self.cas: MemoryCAS = cas or (
            FileCAS(self.config.cas_dir) if self.config.cas_dir else MemoryCAS()
        )
        self.kv = kv or KVStore()
        self._conts: dict[str, Continuation] = {}
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._wakeup = asyncio.Event()
        self.started_at = time.monotonic()

    async def start(self) -> None:
        if self._task is None:
            self._stopped.clear()
            self._task = asyncio.create_task(self._scheduler_loop(), name="dart-scheduler")

    async def aclose(self) -> None:
        self._stopped.set()
        self._wakeup.set()
        if self._task is not None:
            await asyncio.wait_for(self._task, timeout=5)
            self._task = None
        for c in list(self._conts.values()):
            c.closed = True
            c.done = True
            self._fail_pending(c, NackReason.EXPIRED, "runtime closed")

    async def open(
        self,
        prompt: Prompt | str | list[ChatMessage] | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.8,
        model: str | None = None,
    ) -> ContinuationHandle:
        await self.start()
        p = _as_prompt(prompt)
        cfg = self.config
        state = EngineState()
        state.sampler.temperature = temperature
        prefill = await self.engine.prefill(p, state)
        state = prefill.state
        model_hash = self.engine.config.fingerprint()
        cont_id = uuid.uuid4().hex
        now = time.time()
        lease = ContinuationLease(
            cont_id=cont_id,
            model_hash=model_hash,
            model_id=model or self.engine.model_id,
            kv_root=state.kv_root,
            pos=state.pos,
            w_max=cfg.w_max,
            decode_quota=max_tokens or cfg.decode_quota,
            expiry_unix=now + cfg.lease_ttl_s,
            issued_at=now,
            producer_hint=cfg.producer_id,
            startup_credit=cfg.startup_credit,
        )
        signed = sign_lease(lease, cfg.secret)
        metrics = ContinuationMetrics(prefill_tokens=len(state.prompt_ids))
        metrics.kv_bytes_high_water = sum(e.nbytes for e in state.kv_extents)
        cont = Continuation(
            id=cont_id,
            lease=lease,
            signed=signed,
            cc=CongestionController(
                w_init=cfg.w_init,
                w_max=cfg.w_max,
                k_max=cfg.k_max,
                t_decode_s=cfg.t_decode_s,
                rtt_s=cfg.rtt_init_s,
            ),
            state=state,
            model_hash=model_hash,
            metrics=metrics,
            lock=asyncio.Lock(),
            pending={},
            wakeup=asyncio.Event(),
            max_tokens=max_tokens or cfg.decode_quota,
            producer_id=cfg.producer_id,
            prompt_text=p.as_text(),
        )
        self.kv.replace(cont_id, state.kv_extents)
        self.kv.pin(cont_id, lease.expiry_unix)
        self._conts[cont_id] = cont
        logger.info("opened continuation %s kv_root=%s pos=%s", cont_id[:8], state.kv_root[:12], state.pos)
        return ContinuationHandle(cont)

    def get(self, cont_id: str) -> Continuation:
        try:
            return self._conts[cont_id]
        except KeyError as exc:
            raise LeaseError(f"unknown continuation {cont_id}") from exc

    async def interest(self, req: Interest, *, lease: str | None = None) -> Data:
        token = req.lease or lease
        if not token:
            raise LeaseError("missing lease")
        cap = verify_lease(token, self.config.secret)
        cap.assert_window(req.window)
        cont = self.get(cap.cont_id)
        if cont.closed or cont.done:
            cached = self.cas.get_data(req.name)
            if cached:
                cached = cached.model_copy(update={"cache_hit": True})
                return cached
            raise InterestNack(NackReason.DONE.value, "continuation finished")

        cached = self.cas.get_data(req.name)
        if cached is not None:
            cont.metrics.cache_hits += 1
            cont.metrics.interests += 1
            n = cached.token_count()
            cont.cc.on_ack(n)
            cont.metrics.tokens_consumed += n
            return cached.model_copy(update={"cache_hit": True})

        parsed = req.parsed
        if parsed.model_hash != cont.model_hash:
            raise InterestNack(NackReason.NO_MODEL.value, "model fingerprint mismatch")

        async with cont.lock:
            cont.metrics.interests += 1
            cont.last_interest_at = time.monotonic()
            cont.last_lifetime_s = req.lifetime_ms / 1000.0
            if req.locator and req.locator != cont.last_locator:
                # Interest-triggered handover: a new locator may answer.
                logger.info(
                    "handover %s locator %s -> %s",
                    cont.id[:8],
                    cont.last_locator,
                    req.locator,
                )
                cont.last_locator = req.locator
                cont.producer_id = req.locator
            if parsed.kind is InterestKind.GRAMMAR:
                cont.grammar_span = req.grammar_span or parsed.span
            if cont.tokens_since_ack:
                cont.cc.on_ack(cont.tokens_since_ack)
                cont.metrics.tokens_consumed += cont.tokens_since_ack
                cont.tokens_since_ack = 0
            granted = cont.cc.on_interest(req.window, lease_w_max=cont.lease.w_max)
            if granted <= 0 and parsed.kind is InterestKind.TOKENS and not cont.grammar_span:
                if cont.cc.in_flight <= 0 and not cont.pending:
                    raise AmplificationError("no credit remaining under congestion window")

            fut: asyncio.Future[Data | Nack] = asyncio.get_running_loop().create_future()
            # Aggregate duplicate Interests for the same name.
            existing = cont.pending.get(req.name)
            if existing is not None and not existing.done():
                fut = existing
            else:
                cont.pending[req.name] = fut
            cont.sleeping = False
            cont.wakeup.set()
            self._wakeup.set()

        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=req.lifetime_ms / 1000.0)
        except TimeoutError:
            async with cont.lock:
                leftover = cont.cc.on_timeout()
                discarded = _discard_draft(cont)
                cont.metrics.tokens_draft_discarded += discarded
                if leftover:
                    logger.debug("timeout dropped %s credits on %s", leftover, cont.id[:8])
            raise InterestNack(NackReason.EXPIRED.value, "InterestLifetime expired") from None
        if isinstance(result, Nack):
            raise InterestNack(result.reason.value, result.detail)
        return result

    async def ack(self, cont_id: str, consumed: int, lease: str | None = None) -> None:
        if lease:
            verify_lease(lease, self.config.secret)
        cont = self.get(cont_id)
        async with cont.lock:
            n = max(0, consumed)
            take = min(n, cont.tokens_since_ack)
            cont.tokens_since_ack -= take
            cont.cc.on_ack(n)
            cont.metrics.tokens_consumed += n

    async def close(self, cont_id: str) -> None:
        cont = self.get(cont_id)
        async with cont.lock:
            cont.closed = True
            cont.done = True
            cont.cc.credits = 0
            self.kv.sleep(cont_id)
            self._fail_pending(cont, NackReason.EXPIRED, "closed")
        self._wakeup.set()

    async def consume(
        self,
        handle: ContinuationHandle,
        pacer: Any,
        *,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Data]:
        """SDK helper: pacer issues Interests until stop or quota."""
        quota = max_tokens if max_tokens is not None else self.get(handle.cont_id).max_tokens
        produced = 0
        kv_root = handle.kv_root
        seg = 0
        while produced < quota:
            cont = self.get(handle.cont_id)
            if cont.done or cont.closed or cont.state.stopped:
                break
            window = await pacer.next_window()
            window = max(1, min(window, quota - produced, self.config.w_max))
            name = CipName.tokens(handle.model_hash, kv_root, seg).render()
            kind = InterestKind.TOKENS
            grammar_span = None
            if getattr(pacer, "name", "") == "json":
                kind = InterestKind.GRAMMAR
                grammar_span = "next-value"
                name = CipName.grammar(handle.model_hash, kv_root, grammar_span).render()
            req = Interest(
                name=name,
                window=window,
                lifetime_ms=int(self.config.interest_lifetime_s * 1000),
                kind=kind,
                grammar_span=grammar_span,
                cont_id=handle.cont_id,
                lease=handle.lease,
            )
            try:
                data = await self.interest(req, lease=handle.lease)
            except InterestNack as exc:
                if exc.reason in {NackReason.DONE.value, NackReason.EXPIRED.value}:
                    break
                raise
            produced += data.token_count()
            kv_root = data.kv_root
            handle.kv_root = kv_root
            handle.pos = data.pos
            seg += 1
            yield data
            if data.stopped:
                break

    def metrics_snapshot(self) -> dict[str, Any]:
        totals = ContinuationMetrics()
        conts = []
        for c in self._conts.values():
            m = c.metrics
            totals.tokens_generated += m.tokens_generated
            totals.tokens_consumed += m.tokens_consumed
            totals.tokens_drafted += m.tokens_drafted
            totals.tokens_draft_discarded += m.tokens_draft_discarded
            totals.decode_kernel_launches += m.decode_kernel_launches
            totals.decode_steps_skipped += m.decode_steps_skipped
            totals.cache_hits += m.cache_hits
            totals.interests += m.interests
            totals.nacks += m.nacks
            totals.prefill_tokens += m.prefill_tokens
            totals.grammar_spans += m.grammar_spans
            totals.kv_bytes_high_water = max(totals.kv_bytes_high_water, m.kv_bytes_high_water)
            conts.append(
                {
                    "cont_id": c.id,
                    "sleeping": c.sleeping,
                    "done": c.done,
                    "credits": c.cc.credits,
                    "cwnd": c.cc.cwnd,
                    "k": c.cc.speculative_k(),
                    "kv_root": c.state.kv_root,
                    "pos": c.state.pos,
                    "metrics": m.snapshot(),
                }
            )
        return {
            "uptime_s": time.monotonic() - self.started_at,
            "continuations": len(self._conts),
            "cas_entries": len(self.cas.names()) if hasattr(self.cas, "names") else 0,
            "cas_hits": getattr(self.cas, "hits", 0),
            "cas_misses": getattr(self.cas, "misses", 0),
            "kv_bytes": self.kv.current_bytes,
            "kv_gpu_bytes": self.kv.gpu_bytes(),
            "kv_high_water": self.kv.high_water,
            "totals": totals.snapshot(),
            "items": conts,
        }

    def prometheus(self) -> str:
        s = self.metrics_snapshot()
        t = s["totals"]
        lines = [
            "# HELP dart_tokens_generated Tokens produced by a decode kernel.",
            "# TYPE dart_tokens_generated counter",
            f"dart_tokens_generated {t['tokens_generated']}",
            "# HELP dart_tokens_consumed Tokens the consumer ACKed.",
            "# TYPE dart_tokens_consumed counter",
            f"dart_tokens_consumed {t['tokens_consumed']}",
            "# HELP dart_decode_kernel_launches Decode kernel invocations.",
            "# TYPE dart_decode_kernel_launches counter",
            f"dart_decode_kernel_launches {t['decode_kernel_launches']}",
            "# HELP dart_decode_steps_skipped Scheduler ticks with W=0.",
            "# TYPE dart_decode_steps_skipped counter",
            f"dart_decode_steps_skipped {t['decode_steps_skipped']}",
            "# HELP dart_kv_bytes Named KV occupancy.",
            "# TYPE dart_kv_bytes gauge",
            f"dart_kv_bytes {s['kv_bytes']}",
            "# HELP dart_kv_high_water_bytes Peak KV occupancy.",
            "# TYPE dart_kv_high_water_bytes gauge",
            f"dart_kv_high_water_bytes {s['kv_high_water']}",
            "# HELP dart_cache_hits Peer/CAS Interest satisfies.",
            "# TYPE dart_cache_hits counter",
            f"dart_cache_hits {t['cache_hits']}",
            "# HELP dart_continuations Live continuation objects.",
            "# TYPE dart_continuations gauge",
            f"dart_continuations {s['continuations']}",
        ]
        return "\n".join(lines) + "\n"

    async def _scheduler_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("scheduler tick failed")
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self.config.poll_interval_s)
            except TimeoutError:
                pass
            self._wakeup.clear()

    async def _tick(self) -> None:
        ready: list[Continuation] = []
        sleeping = 0
        for cont in list(self._conts.values()):
            if cont.done or cont.closed:
                continue
            if cont.state.stopped or len(cont.state.output_ids) >= cont.max_tokens:
                await self._finish(cont)
                continue
            if cont.cc.expired(cont.last_lifetime_s) and cont.cc.credits > 0 and not cont.pending:
                async with cont.lock:
                    cont.cc.on_timeout()
                    _discard_draft(cont)
            if cont.cc.credits > 0 or cont.grammar_span or cont.pending:
                if cont.sleeping:
                    self.kv.wake(cont.id)
                    cont.sleeping = False
                ready.append(cont)
            else:
                sleeping += 1
                now = time.monotonic()
                if now - cont.last_skip_at >= self.config.t_decode_s:
                    cont.metrics.decode_steps_skipped += 1
                    cont.last_skip_at = now
                if not cont.sleeping:
                    cont.sleeping = True
                    self.kv.sleep(cont.id)
                    self.kv.pin(cont.id, time.time() + cont.lease.remaining_ttl())
        if not ready:
            return
        batch = ready[: self.config.max_num_seqs]
        await asyncio.gather(*(self._decode_one(c) for c in batch))

    async def _decode_one(self, cont: Continuation) -> None:
        async with cont.lock:
            if cont.done or (cont.cc.credits <= 0 and not cont.grammar_span):
                return
            remaining = cont.max_tokens - len(cont.state.output_ids)
            if remaining <= 0:
                await self._finish(cont)
                return
            n, extra = cont.cc.take_decode(min(self.config.segment_size, remaining))
            grammar = cont.grammar_span
            if grammar:
                n = max(n, 1)
            if n <= 0 and not grammar:
                return
            # Speculative K is a CC signal. Do not run uncredited residual-stream
            # steps: a token with no consumer credit is wasted energy and KV.
            if extra:
                cont.metrics.tokens_drafted += extra
            want = n

        result = await self.engine.decode(cont.state, want, grammar_span=grammar)

        async with cont.lock:
            if cont.closed:
                return
            if not result.kernel_launched:
                return
            cont.metrics.decode_kernel_launches += 1
            ids = result.token_ids
            text = result.text
            committed_n = min(len(ids), n) if not grammar else len(ids)
            committed_ids = ids[:committed_n]
            committed_text = text if grammar else _prefix_text(text, committed_n, len(ids))
            draft_ids = ids[committed_n:]
            if draft_ids:
                cont.draft_ids = draft_ids
                cont.draft_text = text[len(committed_text) :]
                cont.metrics.tokens_drafted += len(draft_ids)
            cont.output_text += committed_text
            cont.metrics.tokens_generated += len(committed_ids)
            if grammar:
                cont.metrics.grammar_spans += 1
                cont.grammar_span = None
            self.kv.replace(cont.id, result.extents or cont.state.kv_extents)
            hw = self.kv.bytes_for(cont.id)
            cont.metrics.kv_bytes_high_water = max(cont.metrics.kv_bytes_high_water, hw)

            kv_root_prev = cont.lease.kv_root
            for name, fut in list(cont.pending.items()):
                if fut.done():
                    cont.pending.pop(name, None)
            waiters = [(name, fut) for name, fut in cont.pending.items() if not fut.done()]
            data = Data(
                name=waiters[0][0] if waiters else CipName.tokens(
                    cont.model_hash, kv_root_prev, cont.segment_index
                ).render(),
                tokens=TokenSegment(
                    index=cont.segment_index,
                    token_ids=committed_ids,
                    text=committed_text,
                    pos_begin=cont.state.pos - len(result.token_ids),
                    pos_end=cont.state.pos - len(draft_ids),
                ),
                text=committed_text,
                token_ids=committed_ids,
                kv_root=cont.state.kv_root,
                kv_root_prev=kv_root_prev,
                pos=cont.state.pos - len(draft_ids),
                sampler_hash=cont.state.sampler.hash,
                producer_id=cont.producer_id,
                stopped=result.stopped or len(cont.state.output_ids) >= cont.max_tokens,
                kind=InterestKind.GRAMMAR if grammar else InterestKind.TOKENS,
            )
            self.cas.put_data(data.name, data)
            cont.lease.kv_root = cont.state.kv_root
            cont.lease.pos = data.pos
            cont.segment_index += 1
            cont.tokens_since_ack += len(committed_ids)
            delivered = False
            for name, fut in waiters:
                if fut.done():
                    continue
                # Grammar or matching token name, or the only waiter.
                if name == data.name or grammar or len(waiters) == 1:
                    payload = data.model_copy(update={"name": name})
                    if name != data.name:
                        self.cas.put_data(name, payload)
                    fut.set_result(payload)
                    delivered = True
                    cont.pending.pop(name, None)
            if not delivered and waiters:
                # Interest was for a future/past segment we cannot satisfy this tick.
                pass
            if data.stopped:
                cont.state.stopped = True

    async def _finish(self, cont: Continuation) -> None:
        cont.done = True
        cont.sleeping = True
        self.kv.sleep(cont.id)
        self._fail_pending(cont, NackReason.DONE, "stopped")

    def _fail_pending(self, cont: Continuation, reason: NackReason, detail: str) -> None:
        for name, fut in list(cont.pending.items()):
            if not fut.done():
                fut.set_result(Nack(name=name, reason=reason, detail=detail))
                cont.metrics.nacks += 1
        cont.pending.clear()


def _as_prompt(prompt: Prompt | str | list[ChatMessage] | list[dict[str, str]]) -> Prompt:
    if isinstance(prompt, Prompt):
        return prompt
    if isinstance(prompt, str):
        return Prompt(text=prompt)
    msgs: list[ChatMessage] = []
    for m in prompt:
        if isinstance(m, ChatMessage):
            msgs.append(m)
        else:
            msgs.append(ChatMessage(role=str(m.get("role", "user")), content=str(m.get("content", ""))))
    return Prompt(messages=msgs)


def _discard_draft(cont: Continuation) -> int:
    n = len(cont.draft_ids)
    if n and cont.snapshot is not None:
        cont.state = cont.snapshot
        cont.snapshot = None
    cont.draft_ids.clear()
    cont.draft_text = ""
    return n


def _prefix_text(text: str, committed: int, total: int) -> str:
    if total <= 0 or committed >= total:
        return text
    # Synthetic engine concatenates word+space per token; split on spaces keeping pace.
    parts = text.split(" ")
    # last may be empty due to trailing space
    keep = committed
    out: list[str] = []
    i = 0
    while keep > 0 and i < len(parts):
        out.append(parts[i])
        keep -= 1
        i += 1
    body = " ".join(out)
    if text.endswith(" ") and not body.endswith(" "):
        body += " "
    return body
