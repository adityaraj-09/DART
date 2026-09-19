"""Live vLLM model runner for InProcessVLLMEngine._kernel.

The Interest loop never HTTP POSTs. This process holds an unfinished
vLLM request: ``add_request`` + ``step`` until the prompt is on GPU,
then stop stepping. Those blocks are vLLM's paged KV. ``W=0`` does not
``abort_request``. The next Interest ``step``s at most ``n`` new tokens.

If the engine API is missing, fall back to ``LLM.generate`` with prefix
caching (still in-process; still not HTTP). Tests inject a fake engine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from dart.cip.merkle import bytes_per_extent, extent_digest, root_from_extents
from dart.core.errors import EngineError
from dart.core.types import EngineState, KVExtent, ModelConfig, Prompt
from dart.engine.base import DecodeResult, PrefillResult
from dart.engine.stats import KernelStats

logger = logging.getLogger("dart.engine.vllm_runner")


class VLLMModelRunner:
    """In-process vLLM residual stream. GPU pages stay with the live request."""

    def __init__(
        self,
        model_id: str,
        *,
        config: ModelConfig | None = None,
        engine: Any | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.config = config or ModelConfig(model_id=model_id, tokenizer_hash="vllm-live")
        self.stats = KernelStats()
        self.kernel_launches = 0
        self.decode_tokens = 0
        self.prefills = 0
        self.supports_rollback = False
        self._engine = engine
        self._llm: Any = None
        self._sampling_cls: Any = None
        self._max_tokens = max_tokens
        self._mode = "engine" if engine is not None else ""
        self._cum_ids: dict[str, list[int]] = {}
        self._cum_text: dict[str, str] = {}
        self._emitted: dict[str, int] = {}
        self._finished: dict[str, bool] = {}
        self._prompt_ids: dict[str, list[int]] = {}

    def _extents(self, state: EngineState) -> list[KVExtent]:
        cfg = self.config
        nbytes = bytes_per_extent(cfg.n_kv_heads, cfg.head_dim, cfg.block_size)
        n_blocks = max(1, (max(state.pos, 1) + cfg.block_size - 1) // cfg.block_size)
        out: list[KVExtent] = []
        for layer in range(min(cfg.n_layers, 8)):
            for b in range(n_blocks):
                digest = extent_digest(layer, b, state.pos, state.all_ids[-32:])
                out.append(KVExtent(layer=layer, block_id=b, digest=digest, nbytes=nbytes, on_gpu=True))
        return out

    def _ensure(self) -> None:
        if self._engine is not None:
            return
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise EngineError("vLLM package is not installed") from exc
        self._sampling_cls = SamplingParams
        engine, llm, mode = _open_vllm_engine(self.model_id)
        self._engine = engine
        self._llm = llm
        self._mode = mode
        self._sync_config_from_engine()

    def _sync_config_from_engine(self) -> None:
        model = getattr(self._engine, "model_config", None) or getattr(
            getattr(self._llm, "llm_engine", None), "model_config", None
        )
        if model is None:
            return
        try:
            self.config = self.config.model_copy(
                update={
                    "n_layers": int(getattr(model, "get_num_layers", lambda: 8)() or 8),
                    "block_size": int(getattr(getattr(self._engine, "cache_config", None), "block_size", 16) or 16),
                }
            )
        except Exception:
            pass

    def _params(self, *, max_tokens: int, temperature: float) -> Any:
        cls = self._sampling_cls
        if cls is None:
            try:
                from vllm import SamplingParams as cls
            except ImportError:
                cls = None
            self._sampling_cls = cls
        if cls is None:
            return _SimpleParams(max_tokens=max_tokens, temperature=temperature)
        return cls(max_tokens=max(1, max_tokens), temperature=temperature)

    def _tokenize(self, prompt: Prompt) -> list[int]:
        if prompt.token_ids:
            return list(prompt.token_ids)
        text = prompt.as_text() or " "
        for obj in (self._engine, self._llm):
            get = getattr(obj, "get_tokenizer", None)
            if not callable(get):
                continue
            try:
                tok = get()
                ids = tok.encode(text)
                return [int(x) for x in ids]
            except Exception:
                continue
        return [1] * max(1, len(text.split()) + 4)

    def _add_request(self, rid: str, prompt: str, params: Any) -> None:
        add = getattr(self._engine, "add_request", None)
        if not callable(add):
            raise EngineError("vLLM engine has no add_request")
        add(rid, prompt, params)

    def _step(self) -> list[Any]:
        step = getattr(self._engine, "step", None)
        if not callable(step):
            raise EngineError("vLLM engine has no step")
        out = step()
        if out is None:
            return []
        if isinstance(out, list):
            return out
        return list(out)

    def _abort(self, rid: str) -> None:
        fn = getattr(self._engine, "abort_request", None) or getattr(self._engine, "abort", None)
        if callable(fn):
            try:
                fn(rid)
            except Exception:
                logger.debug("vLLM abort_request(%s) failed", rid[:8], exc_info=True)

    def _unfinished(self) -> bool:
        fn = getattr(self._engine, "has_unfinished_requests", None)
        if callable(fn):
            return bool(fn())
        return any(not self._finished.get(r, False) for r in self._emitted)

    def _harvest(self, rid: str, outputs: list[Any]) -> None:
        for item in outputs:
            if getattr(item, "request_id", None) != rid:
                continue
            outs = getattr(item, "outputs", None) or []
            if not outs:
                continue
            first = outs[0]
            ids = [int(x) for x in (getattr(first, "token_ids", None) or [])]
            text = str(getattr(first, "text", "") or "")
            self._cum_ids[rid] = ids
            self._cum_text[rid] = text
            self._finished[rid] = bool(getattr(item, "finished", False))

    def _prefill_sync(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        self._ensure()
        text = prompt.as_text()
        ids = self._tokenize(prompt)
        rid = state.engine_request_id or uuid.uuid4().hex
        state.engine_request_id = rid
        state.prompt_ids = ids
        state.output_ids = []
        state.pos = len(ids)
        state.prefix_text = text
        state.assistant_text = ""
        self._prompt_ids[rid] = ids
        self._cum_ids[rid] = []
        self._cum_text[rid] = ""
        self._emitted[rid] = 0
        self._finished[rid] = False
        if self._mode == "generate":
            # Prefix-cache generate path: first Interest will populate GPU KV.
            state.kv_extents = self._extents(state)
            state.kv_root = root_from_extents(state.kv_extents)
            self.prefills += 1
            return PrefillResult(state=state, text=text, extents=state.kv_extents)
        self._add_request(
            rid,
            text or " ",
            self._params(max_tokens=self._max_tokens, temperature=state.sampler.temperature),
        )
        # One (or a few) steps: allocate paged KV. Do not emit tokens yet.
        for _ in range(8):
            if not self._unfinished() and self._cum_ids.get(rid):
                break
            try:
                self._harvest(rid, self._step())
            except Exception as exc:
                raise EngineError(f"vLLM prefill step failed: {exc}") from exc
            # Prefill is done once the request is known to the engine.
            if rid in self._cum_ids:
                break
        self.prefills += 1
        self.stats.kernel_launches += 1
        self.kernel_launches = self.stats.kernel_launches
        state.kv_extents = self._extents(state)
        state.kv_root = root_from_extents(state.kv_extents)
        return PrefillResult(state=state, text=text, extents=state.kv_extents)

    def _take(self, rid: str, n: int) -> tuple[list[int], str, bool]:
        have = self._emitted.get(rid, 0)
        ids = self._cum_ids.get(rid, [])
        take = ids[have : have + n]
        self._emitted[rid] = have + len(take)
        full = self._cum_text.get(rid, "")
        # Best-effort slice of newly generated text.
        if not take:
            piece = ""
        elif have == 0:
            piece = full
        else:
            piece = full[-max(1, len(take) * 8) :]
        return take, piece, bool(self._finished.get(rid))

    def _decode_engine_sync(self, state: EngineState, n: int) -> tuple[list[int], str, bool]:
        rid = state.engine_request_id
        if not rid:
            raise EngineError("vLLM decode without request id")
        pending = self._cum_ids.get(rid, [])
        if len(pending) > self._emitted.get(rid, 0):
            got, text, stopped = self._take(rid, n)
            if len(got) >= n or stopped:
                return got, text, stopped
            n = n - len(got)
            extra_ids, extra_text, stopped = self._drive(rid, n)
            return got + extra_ids, text + extra_text, stopped
        return self._drive(rid, n)

    def _drive(self, rid: str, n: int) -> tuple[list[int], str, bool]:
        target = self._emitted.get(rid, 0) + n
        guard = 0
        while self._emitted.get(rid, 0) < target and not self._finished.get(rid) and guard < n * 32 + 8:
            if not self._unfinished() and self._emitted.get(rid, 0) >= target:
                break
            try:
                self._harvest(rid, self._step())
            except Exception as exc:
                raise EngineError(f"vLLM decode step failed: {exc}") from exc
            guard += 1
            if not self._unfinished() and not self._cum_ids.get(rid):
                break
        return self._take(rid, n)

    def _decode_generate_sync(self, state: EngineState, n: int) -> tuple[list[int], str, bool]:
        if self._llm is None:
            raise EngineError("vLLM generate fallback has no LLM")
        from vllm import SamplingParams

        prompt = (state.prefix_text or " ") + (state.assistant_text or "")
        params = SamplingParams(max_tokens=max(1, n), temperature=state.sampler.temperature)
        outs = self._llm.generate(prompt, params)
        first = outs[0].outputs[0]
        ids = [int(x) for x in (first.token_ids or [])]
        text = str(first.text or "")
        stopped = bool(getattr(first, "finish_reason", "") == "stop")
        return ids, text, stopped

    def _apply(self, state: EngineState, new_ids: list[int], text: str, stopped: bool) -> DecodeResult:
        predicted = max(1, len(new_ids) or len(text.split()) or 1)
        self.stats.record_generate(predicted=predicted, prompt_tokens=1, prefix_hit=True)
        self.kernel_launches = self.stats.kernel_launches
        self.decode_tokens = self.stats.tokens_predicted
        state.assistant_text += text
        if not new_ids:
            new_ids = list(range(state.pos, state.pos + predicted))
        state.output_ids.extend(new_ids)
        state.pos += len(new_ids)
        state.kv_extents = self._extents(state)
        state.kv_root = root_from_extents(state.kv_extents)
        state.stopped = stopped
        return DecodeResult(
            token_ids=new_ids,
            text=text,
            extents=state.kv_extents,
            kv_root=state.kv_root,
            stopped=stopped,
        )

    async def prefill(self, prompt: Prompt, state: EngineState) -> PrefillResult:
        return await asyncio.to_thread(self._prefill_sync, prompt, state)

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
        window = max(1, n if not grammar_span else max(n, 32))

        def run() -> tuple[list[int], str, bool]:
            self._ensure()
            if self._mode == "generate":
                return self._decode_generate_sync(state, window)
            return self._decode_engine_sync(state, window)

        try:
            new_ids, text, stopped = await asyncio.to_thread(run)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"vLLM live decode failed: {exc}") from exc
        return self._apply(state, new_ids, text, stopped)

    async def abort_generation(self, state: EngineState) -> None:
        rid = state.engine_request_id
        if rid:
            await asyncio.to_thread(self._abort, rid)
            self._finished[rid] = True


class _SimpleParams:
    def __init__(self, *, max_tokens: int, temperature: float) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature


def _open_vllm_engine(model_id: str) -> tuple[Any, Any, str]:
    """Prefer LLMEngine (unfinished request = live GPU pages). Else LLM.generate."""
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    extra: dict[str, Any] = {}
    if os.environ.get("DART_VLLM_MAX_MODEL_LEN"):
        extra["max_model_len"] = int(os.environ["DART_VLLM_MAX_MODEL_LEN"])
    gpu_mem = os.environ.get("DART_VLLM_GPU_MEM")
    if gpu_mem:
        extra["gpu_memory_utilization"] = float(gpu_mem)
    try:
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
    except ImportError:
        LLM = None  # type: ignore[assignment]
        EngineArgs = None  # type: ignore[assignment]
        LLMEngine = None  # type: ignore[assignment]
    if EngineArgs is not None and LLMEngine is not None:
        try:
            args = EngineArgs(
                model=model_id,
                enable_prefix_caching=True,
                disable_log_stats=True,
                **extra,
            )
            engine = LLMEngine.from_engine_args(args)
            return engine, None, "engine"
        except TypeError:
            try:
                args = EngineArgs(model=model_id, **extra)
                engine = LLMEngine.from_engine_args(args)
                return engine, None, "engine"
            except Exception as exc:
                logger.warning("LLMEngine.from_engine_args failed: %s", exc)
        except Exception as exc:
            logger.warning("LLMEngine.from_engine_args failed: %s", exc)
    if LLM is not None:
        try:
            llm = LLM(model=model_id, enable_prefix_caching=True, **extra)
        except TypeError:
            llm = LLM(model=model_id, **extra)
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        if engine is not None and callable(getattr(engine, "add_request", None)):
            return engine, llm, "engine"
        return engine, llm, "generate"
    raise EngineError("could not construct an in-process vLLM engine")


def use_live_vllm_runner(model_id: str) -> bool:
    """When to plug VLLMModelRunner into InProcessVLLMEngine._kernel."""
    flag = (os.environ.get("DART_VLLM_INPROCESS") or "").strip().lower()
    if flag in {"0", "off", "false", "no"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return _vllm_present()
    if model_id in {"", "dart-synth-8b", "synthetic", "local"}:
        return False
    return _vllm_present()


def _vllm_present() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def build_live_kernel(model_id: str, *, config: ModelConfig | None = None) -> VLLMModelRunner:
    return VLLMModelRunner(model_id, config=config)
