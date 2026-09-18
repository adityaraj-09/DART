# Engine adapters

DART never owns the residual stream. It owns **whether** the residual stream may run.

All producers implement:

```python
async def prefill(prompt, state) -> PrefillResult
async def decode(state, n, *, grammar_span=None) -> DecodeResult
```

`DecodeResult.kernel_launched` is the truth the kill-test measures. Returning empty with `kernel_launched=False` is required when `n <= 0`.

## SyntheticEngine (default)

Hash-based autoregression with **real KV-extent accounting** (layers × blocks × K/V bytes). Deterministic, no GPU, supports rollback for speculative K.

Use it for:

- unit tests  
- `dart experiment` (the inversion test)  
- local demo at `GET /`

This is not a protocol toy: the scheduler, CAS, leases, and CC are the same objects vLLM will sit behind.

## VLLMChatEngine (production GPU)

OpenAI-compatible HTTP to a vLLM server (`DART_VLLM_URL`, default `http://127.0.0.1:8000/v1`).

Each Interest becomes `chat.completions` with `max_tokens=W`. Enable `--enable-prefix-caching` on the server so this is not a re-prefill.

Limits of the HTTP path (honest):

- Token ids and real KV bytes are not returned; extents are opaque handles sized like the config.  
- Cross-machine handover needs LMCache / NixlConnector, not this adapter.  
- Per-request `waiting` with pinned blocks is a **vLLM scheduler plugin** we did not fork into existence. The invariant (no generate without credit) still holds because DART never calls the API when `W=0`.  
- The HTTP path **re-enters admission** and **prefix cache may evict** (implicit re-prefill). See [`limitations.md`](./limitations.md). `GET /v1/engine` compares DART’s local POST count with the engine process `/metrics` forward counter.

In-process `vllm.AsyncLLM` can replace HTTP later without changing CIP.

```bash
export DART_ENGINE=vllm
export DART_MODEL=meta-llama/Llama-3.1-8B-Instruct
export DART_VLLM_URL=http://127.0.0.1:8000/v1
dart serve --engine vllm --model "$DART_MODEL"
```

## LlamaCppEngine (bench / edge)

`POST /completion` with `n_predict=W` and `cache_prompt=true`. Matches the “afternoon kill-test on llama.cpp” slice, but already wired through CIP so you do not throw the loop away.

```bash
export DART_ENGINE=llamacpp
export DART_LLAMACPP_URL=http://127.0.0.1:8080
dart serve --engine llamacpp
```

## Compatibility fingerprint

`ModelConfig.fingerprint()` is part of every CIP name. A phone answering an Interest for datacenter KV must match tokenizer, RoPE, dtype, block size, GQA layout. Mismatch → `Nack no_model`. That is the thaw/KDN problem, now on the protocol path, and it is *refused* rather than silently decoded.

## Adding a producer

1. Implement `Engine`.  
2. Set `supports_rollback=True` only if `decode` can restore `EngineState` after discarded draft.  
3. Register in `dart.factory.build_engine`.  
4. Do not generate in a background thread “to stay warm.” Warmth is other paid work, not unread tokens.
