"""Deterministic steppable producer for tests, CPU benches, and the kill-test.

This is not a toy protocol shim: it still pays a per-token KV extent and
only advances when the runtime asks. Swap for VLLMEngine in production.
"""

from __future__ import annotations

import asyncio
import hashlib
import struct

from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.stats import KernelStats
from dart.merkle import bytes_per_extent, extent_digest, root_from_extents
from dart.types import EngineState, KVExtent, ModelConfig, Prompt, SamplerState

_WORDS = (
    "the of and to a in is it you that he was for on are as with his they "
    "at be this from I have or by one had not but what all were when we "
    "there can an your which their said if do will each about how up out "
    "them then she many some so these would other into has more her two "
    "like him see time could no make than first been its who now people "
    "my made over did down only way find use may water long little very "
    "after words called just where most know get through back much before "
    "go good new write our used me man too any day same right look think "
    "also around another came come work three word must because does part "
    "even place well such here take why things help put years different "
    "away again off went old number great tell men say small every found "
    "still between name should Mr home big give air line set own under "
    "read last never us left end along while might next sound below saw "
    "something thought both few those always showed large often together "
    "asked house don't world going want school important until form food "
    "keep children feet land side without boy once animals life enough "
    "took four head above kind began almost live page got earth need far "
    "hand high year mother light parts country father let night following "
    "answer found picture study learn change answer room sea against box"
).split()

_JSON_SPANS = {
    "next-value": '{"status":"ok","id":7}',
    "json_value": '{"ok":true}',
    "tool-args": '{"name":"search","query":"dart idd"}',
    "string": '"alpha"',
}


def _tokenize(text: str) -> list[int]:
    if not text:
        return []
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isspace():
            if buf:
                parts.append("".join(buf))
                buf.clear()
            parts.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    ids: list[int] = []
    for p in parts:
        ids.append(int(hashlib.sha256(p.encode()).hexdigest()[:8], 16) % 32000)
    return ids


def _detok(ids: list[int]) -> str:
    words = []
    for i, tid in enumerate(ids):
        w = _WORDS[tid % len(_WORDS)]
        words.append((" " if i else "") + w)
    return "".join(words)


def _next_id(seed: int, ids: list[int]) -> int:
    h = hashlib.sha256()
    h.update(struct.pack("<I", seed))
    h.update(struct.pack("<I", len(ids)))
    for t in ids[-48:]:
        h.update(struct.pack("<I", t & 0xFFFFFFFF))
    return int(h.hexdigest()[:8], 16) % 32000


class SyntheticEngine:
    """Hash-based autoregressive producer with real KV-extent accounting."""

    def __init__(
        self,
        model_id: str = "dart-synth-8b",
        *,
        step_latency_s: float = 0.0,
        seed: int = 0,
        config: ModelConfig | None = None,
        script: list[str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.step_latency_s = step_latency_s
        self.seed = seed
        self.config = config or ModelConfig(model_id=model_id)
        self.script = script
        self._script_i = 0
        self.stats = KernelStats()
        self.kernel_launches = 0
        self.decode_tokens = 0
        self.supports_rollback = True

    def _extents(self, ids: list[int], pos: int) -> list[KVExtent]:
        cfg = self.config
        nbytes = bytes_per_extent(cfg.n_kv_heads, cfg.head_dim, cfg.block_size)
        n_blocks = max(1, (max(pos, 1) + cfg.block_size - 1) // cfg.block_size)
        extents: list[KVExtent] = []
        for layer in range(cfg.n_layers):
            for b in range(n_blocks):
                digest = extent_digest(layer, b, pos, ids)
                extents.append(
                    KVExtent(layer=layer, block_id=b, digest=digest, nbytes=nbytes, on_gpu=True)
                )
        return extents

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        text = prompt.as_text()
        ids = prompt.token_ids or _tokenize(text)
        if not ids:
            ids = _tokenize("prompt")
        extents = self._extents(ids, len(ids))
        root = root_from_extents(extents)
        new_state = EngineState(
            prompt_ids=ids,
            output_ids=[],
            sampler=state.sampler if state.sampler.hash != "init" else SamplerState(seed=self.seed),
            grammar=state.grammar,
            kv_extents=extents,
            kv_root=root,
            pos=len(ids),
        )
        return PrefillResult(state=new_state, text=text, extents=extents)

    async def decode(
        self,
        state: EngineState,
        n: int,
        *,
        grammar_span: str | None = None,
    ) -> DecodeResult:
        if n <= 0 and not grammar_span:
            return DecodeResult(
                token_ids=[],
                text="",
                extents=state.kv_extents,
                kv_root=state.kv_root,
                kernel_launched=False,
            )
        if self.step_latency_s > 0:
            await asyncio.sleep(self.step_latency_s * max(n, 1))
        self.kernel_launches += 1
        self.stats.kernel_launches = self.kernel_launches
        if grammar_span:
            text = _JSON_SPANS.get(grammar_span, f'{{"span":"{grammar_span}"}}')
            ids = _tokenize(text)
            state.output_ids.extend(ids)
            state.pos += len(ids)
            state.kv_extents = self._extents(state.all_ids, state.pos)
            state.kv_root = root_from_extents(state.kv_extents)
            state.sampler = state.sampler.advance(ids[-1] if ids else 0)
            self.decode_tokens += len(ids)
            self.stats.tokens_predicted = self.decode_tokens
            return DecodeResult(
                token_ids=ids,
                text=text,
                extents=state.kv_extents,
                kv_root=state.kv_root,
                grammar_span=True,
            )
        ids: list[int] = []
        chunks: list[str] = []
        had_output = bool(state.output_ids)
        for _ in range(n):
            if self.script is not None:
                if self._script_i >= len(self.script):
                    state.stopped = True
                    break
                piece = self.script[self._script_i]
                self._script_i += 1
                tid = _tokenize(piece)[0] if piece else 0
                ids.append(tid)
                chunks.append(piece if piece.endswith(" ") else piece + " ")
            else:
                tid = _next_id(self.seed, state.all_ids)
                ids.append(tid)
            state.output_ids.append(tid)
            state.pos += 1
            state.sampler = state.sampler.advance(tid)
        text = "".join(chunks) if self.script is not None else _detok(ids)
        if had_output and text and not text.startswith((" ", "\n")):
            text = " " + text
        state.kv_extents = self._extents(state.all_ids, state.pos)
        state.kv_root = root_from_extents(state.kv_extents)
        self.decode_tokens += len(ids)
        self.stats.tokens_predicted = self.decode_tokens
        return DecodeResult(
            token_ids=ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=state.stopped,
        )
