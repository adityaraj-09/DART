# DART

**Demand-Addressed Runtime Tokens** — a named continuation runtime. Outstanding **Interests** are the only thing that may run a decode kernel.

Generation is filling named holes in a continuation address space:

```text
/cip/<model-hash>/<kv-root>/tokens/seg/<i>
/cip/<model-hash>/<kv-root>/grammar/span/<s>
```

A consumer (terminal, compositor, TTS, JSON parser, tool runtime) pulls what it can absorb. A producer decodes only to satisfy that window. Peers that already hold the object answer without a GPU. Speculative depth `K`, grammar jump-forward, and KV sleep are congestion-control variables.

This is not Andes (push then pace), not MOQT (delivery of already-generated tokens), not LMCache (store). It closes the loop: **Interest authorizes decode**.

Architecture: [`docs/architecture.md`](docs/architecture.md) · Protocol: [`docs/protocol.md`](docs/protocol.md) · CC: [`docs/congestion-control.md`](docs/congestion-control.md) · Engines: [`docs/engine-adapters.md`](docs/engine-adapters.md)

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
# CPU demo producer (deterministic, real KV accounting)
dart serve --engine synthetic --port 8090

# Kill-test: always-push vs credit-gated decode
dart experiment --seconds 3 --max-tokens 256
```

Open http://127.0.0.1:8090 — set pace to 30 tok/s and stream. Metrics on the right are kernel launches, skipped steps, and KV high-water.

Production GPU:

```bash
export DART_ENGINE=vllm
export DART_MODEL=meta-llama/Llama-3.1-8B-Instruct
export DART_VLLM_URL=http://127.0.0.1:8000/v1
export DART_SECRET=replace-me
dart serve --engine vllm --model "$DART_MODEL"
```

Enable `--enable-prefix-caching` on vLLM. DART will not call `generate` unless the consumer has credit.

## SDK

```python
from dart.sdk import DartClient, ReadingPacer, TtsPacer

async for seg in DartClient(runtime=rt).stream(
    "Explain DASH for tokens.",
    consumer=ReadingPacer(tokens_per_sec=30),
    max_tokens=128,
):
    print(seg.text, end="", flush=True)
```

OpenAI-compatible clients keep working:

```bash
curl -N http://127.0.0.1:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-dart-pace: 30' \
  -d '{"model":"dart-synth-8b","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

Disconnect stops the kernel. `X-Dart-Pace` is the receive window.

## Tests

```bash
pytest -q
```
