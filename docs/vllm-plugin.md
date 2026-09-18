# In-process vLLM waiting-queue plugin

HTTP `--engine vllm` still hangs up after every Interest (`chat.completions` with `max_tokens=W`). Prefix cache *may* keep KV. Admission *will* run again.

`--engine vllm-inprocess` is the scheduler plugin:

1. **Admit once** on `prefill` / `open()`.
2. Request sits in **`waiting`** with **pinned GPU blocks**.
3. `W=0` (no Interest) → `pause_generation(mode="keep")`. Kitchen does not cook; the table stays yours. Memory pressure may evict *unpinned* prefix-cache blocks, never this pin.
4. Next Interest → `waiting` → `running` → decode `n` tokens → back to `waiting`. **No second admit. No re-prefill.**

That is why it exists: not to invent credit-gated decode (DART already refuses `generate` when `W=0`), but to stop vLLM from forgetting the KV or bouncing the live continuation off the admission door.

## What runs in tests / CPU

`CreditGatedScheduler` (`src/dart/engine/vllm_sched.py`) plus `InProcessVLLMEngine` (`src/dart/engine/vllm_inprocess.py`). The residual stream is `SyntheticEngine` unless `vllm` is installed **and** `DART_VLLM_INPROCESS=1`.

```bash
dart serve --engine vllm-inprocess
dart experiment --suite waiting
```

## What is still HTTP

```bash
dart serve --engine vllm   # OpenAI HTTP, prefix cache, re-enters admission
```

If the real vLLM V1 `Scheduler` class is importable, `dart.engine.vllm_plugin.dart_scheduler_class()` subclasses it so zero-credit running requests are parked in `waiting` instead of staying in the decode batch. The entry point `vllm.general_plugins` / `dart_idd` is a safe no-op when vLLM is absent.

## Counters

| Path | `open()` | Two Interests | Idle `W=0` |
|---|---|---|---|
| HTTP `VLLMChatEngine` | 0 forwards | 2 POSTs (admission twice) | 0 POSTs; KV may evict |
| In-process plugin | 1 prefill forward, 1 admit | 2 decode forwards, **still 1 admit** | waiting + pinned blocks |
