"""In-process vLLM engine: one admission, then waiting with pinned blocks.

``dart serve --engine vllm`` is this engine unless ``DART_VLLM_URL`` is set
(HTTP adapter). Production path: admit once, park in ``waiting`` with pinned
KV when ``W=0``. ``--engine vllm-http`` is the re-admission POST path.

The Interest loop never HTTP POSTs. A live ``vllm`` model runner is
``_kernel`` when the package is present and the model is a real checkpoint
(or ``DART_VLLM_INPROCESS=1``). Then waiting/pinned blocks are vLLM's GPU
pages (unfinished ``add_request``). Tests and ``dart-synth-8b`` keep
``SyntheticEngine`` as the residual stream.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.stats import KernelStats
from dart.engine.synthetic import SyntheticEngine
from dart.engine.vllm_sched import CreditGatedScheduler, RequestStatus
from dart.core.errors import EngineError
from dart.cip.merkle import bytes_per_extent
from dart.core.types import EngineState, ModelConfig, Prompt


def _vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


class InProcessVLLMEngine:
    """Producer whose scheduler leaves the request in waiting with pinned KV."""

    def __init__(
        self,
        model_id: str = "dart-synth-8b",
        *,
        seed: int = 0,
        config: ModelConfig | None = None,
        step_latency_s: float = 0.0,
        num_gpu_blocks: int = 2048,
        kernel: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.config = config or ModelConfig(model_id=model_id, tokenizer_hash="vllm-inprocess")
        self._kernel = kernel if kernel is not None else _default_kernel(
            model_id, seed=seed, config=self.config, step_latency_s=step_latency_s
        )
        nbytes = bytes_per_extent(
            self.config.n_kv_heads, self.config.head_dim, self.config.block_size
        )
        self.sched = CreditGatedScheduler(num_gpu_blocks=num_gpu_blocks, block_nbytes=nbytes)
        self.stats = KernelStats()
        self.kernel_launches = 0
        self.decode_tokens = 0
        self.prefills = 0
        self.supports_rollback = True
        self.vllm_installed = _vllm_importable()
        live = type(self._kernel).__name__ == "VLLMModelRunner"
        self.backend = "vllm-gpu" if live else (
            "vllm" if self.vllm_installed and os.environ.get("DART_VLLM_INPROCESS") else "inprocess"
        )
        self.pause_keeps = 0

    def scheduler_snapshot(self) -> dict[str, Any]:
        snap = self.sched.snapshot()
        snap["backend"] = self.backend
        snap["vllm_installed"] = self.vllm_installed
        snap["kernel"] = type(self._kernel).__name__
        return snap

    async def scrape_engine_metrics(self) -> dict[str, int]:
        s = self.sched.snapshot()
        return {
            "kernel_launches": self.stats.kernel_launches,
            "tokens_predicted": self.stats.tokens_predicted,
            "admissions": s["admissions"],
            "admission_reentries": s["admission_reentries"],
            "decode_forwards": s["decode_forwards"],
            "prefill_forwards": s["prefill_forwards"],
            "waiting": s["waiting"],
            "pinned_blocks": s["pinned_blocks"],
            "pause_keeps": s["pause_keeps"],
        }

    def _blocks_needed(self, pos: int) -> int:
        bs = max(1, self.config.block_size)
        return max(1, (max(pos, 1) + bs - 1) // bs) * max(1, self.config.n_layers)

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        result = await self._kernel.prefill(prompt, state)
        st = result.state
        rid = st.engine_request_id or uuid.uuid4().hex
        st.engine_request_id = rid
        if self.sched.has(rid):
            self.sched.admission_reentries += 1
        else:
            self.sched.admit(
                rid,
                n_tokens=st.pos,
                n_blocks=self._blocks_needed(st.pos),
                token_ids=st.all_ids,
                digest=st.kv_root,
            )
            # admit() already counted prefill_forwards; kernel prefill is that forward.
            self.prefills += 1
            self.stats.kernel_launches += 1
            self.kernel_launches = self.stats.kernel_launches
        self.sched.pause(rid, mode="keep")
        self.stats.admission_reentries = self.sched.admission_reentries
        for e in st.kv_extents:
            e.on_gpu = True
        return PrefillResult(state=st, text=result.text, extents=st.kv_extents)

    async def decode(
        self,
        state: EngineState,
        n: int,
        *,
        grammar_span: str | None = None,
    ) -> DecodeResult:
        if n <= 0 and not grammar_span:
            await self.pause_generation(state, mode="keep")
            return DecodeResult(
                token_ids=[],
                text="",
                extents=state.kv_extents,
                kv_root=state.kv_root,
                kernel_launched=False,
            )
        rid = await self._bind(state)
        try:
            self.sched.resume(rid, credit=n)
        except (KeyError, MemoryError) as exc:
            raise EngineError(f"in-process vLLM cannot resume {rid}: {exc}") from exc
        result = await self._kernel.decode(state, n, grammar_span=grammar_span)
        if result.kernel_launched:
            self.sched.after_decode(rid, len(result.token_ids), token_ids=state.all_ids)
            self.stats.kernel_launches += 1
            self.stats.tokens_predicted += len(result.token_ids)
            self.kernel_launches = self.stats.kernel_launches
            self.decode_tokens = self.stats.tokens_predicted
        self.sched.pause(rid, mode="keep")
        self.stats.admission_reentries = self.sched.admission_reentries
        for e in state.kv_extents:
            e.on_gpu = True
        result.extents = state.kv_extents
        return result

    async def pause_generation(self, state: EngineState, mode: str = "keep") -> None:
        rid = state.engine_request_id
        if not rid or not self.sched.has(rid):
            return
        self.sched.pause(rid, mode=mode)
        self.pause_keeps += 1
        if mode == "keep":
            for e in state.kv_extents:
                e.on_gpu = True
        else:
            for e in state.kv_extents:
                e.on_gpu = False

    async def resume_generation(self, state: EngineState) -> None:
        rid = state.engine_request_id
        if not rid or not self.sched.has(rid):
            return
        req = self.sched.get(rid)
        if req is None:
            return
        # Stay waiting until decode() actually runs; just ensure pin.
        self.sched.pause(rid, mode="keep")
        for e in state.kv_extents:
            e.on_gpu = True

    async def abort_generation(self, state: EngineState) -> None:
        rid = state.engine_request_id
        if not rid:
            return
        self.sched.abort(rid)
        fn = getattr(self._kernel, "abort_generation", None)
        if callable(fn):
            await fn(state)

    def apply_memory_pressure(self, need_blocks: int) -> int:
        return self.sched.blocks.evict_unpinned(need_blocks)

    def request_status(self, state: EngineState) -> str | None:
        rid = state.engine_request_id
        req = self.sched.get(rid) if rid else None
        return req.status.value if req else None

    async def _bind(self, state: EngineState) -> str:
        rid = state.engine_request_id
        if rid and self.sched.has(rid):
            return rid
        # Adopt / copied state: install KV without a second prefill kernel.
        rid = rid or uuid.uuid4().hex
        state.engine_request_id = rid
        if not self.sched.has(rid):
            self.sched.admit(
                rid,
                n_tokens=state.pos,
                n_blocks=self._blocks_needed(state.pos),
                token_ids=state.all_ids,
                digest=state.kv_root,
                count_prefill=False,
            )
        return rid


def _default_kernel(
    model_id: str,
    *,
    seed: int,
    config: ModelConfig,
    step_latency_s: float,
) -> Any:
    from dart.engine.vllm_runner import use_live_vllm_runner

    if use_live_vllm_runner(model_id):
        try:
            from dart.engine.vllm_runner import VLLMModelRunner

            return VLLMModelRunner(model_id, config=config)
        except Exception:
            pass
    return SyntheticEngine(model_id, seed=seed, config=config, step_latency_s=step_latency_s)
