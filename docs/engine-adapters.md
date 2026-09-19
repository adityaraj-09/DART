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

## InProcessVLLMEngine (production GPU)

`--engine vllm` (and `vllm-inprocess`) admits once, then parks the request in `waiting` with pinned KV when `W=0`. The next Interest resumes; it does not POST `max_tokens=W` and does not re-enter admission.

`_kernel` is a live `VLLMModelRunner` when the `vllm` package can load the checkpoint (or `DART_VLLM_INPROCESS=1`). Waiting/pinned blocks are that engine’s GPU pages: one unfinished `add_request`, no HTTP. Tests and `dart-synth-8b` keep `SyntheticEngine`. CIP does not change.

```bash
dart serve --engine vllm --model meta-llama/Llama-3.1-8B-Instruct
```

## VLLMChatEngine (HTTP, re-admits)

`--engine vllm-http`, or `--engine vllm` when `DART_VLLM_URL` is set. Each Interest becomes `chat.completions` with `max_tokens=W`. Enable `--enable-prefix-caching` on the server so this is not a re-prefill.

Limits of the HTTP path (honest):

- Token ids and real KV bytes are not returned; extents are opaque handles sized like the config.  
- A live continuation can 503 or re-prefill if the remote prefix cache evicts.  
- Cross-machine handover uses `KVConnector` (`--connector nixl` / `lmcache`) and adopts `kv_root` without `engine.prefill`. See [`mesh.md`](./mesh.md).  
- The HTTP path **re-enters admission**. See [`limitations.md`](./limitations.md). `GET /v1/engine` compares DART’s local POST count with the engine process `/metrics` forward counter.

```bash
export DART_VLLM_URL=http://127.0.0.1:8000/v1
dart serve --engine vllm-http --model "$DART_MODEL"
```

## HuggingFaceEngine (small real model)

In-process `transformers` causal LM. Each Interest is `max_new_tokens=W`. Default demo weights: **HuggingFaceTB/SmolLM2-135M-Instruct** (135M, CPU).

```bash
pip install -e ".[hf]"
dart serve --engine hf --model HuggingFaceTB/SmolLM2-135M-Instruct --port 8090
```

The console header shows the live `model_id`. This is a real decoder, not the synthetic word list.

Prefill stores Hugging Face `past_key_values` on the request. Later pages pass only the new seed token plus that cache — they do **not** re-encode the prefix. Set `DART_HF_DEVICE=cuda` for a GPU box.

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
