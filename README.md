# DART

**Demand-Addressed Runtime Tokens**

DART is a continuation runtime for autoregressive decode. Outstanding **Interests** are the only thing that may run a decode kernel. Consumers pull tokens they can absorb; producers generate only that window; any peer that already holds the named object can answer without a GPU.

```text
/cip/<model-hash>/<kv-root>/tokens/seg/<i>
```

DART sits in front of an existing kernel (vLLM, llama.cpp, or the in-repo synthetic producer). It does not replace a GPU fleet.

---

## Quick start

```bash
pip install -e ".[dev]"
dart serve --engine synthetic --port 8090
```

Open [http://127.0.0.1:8090](http://127.0.0.1:8090). Stream at 30 tok/s and watch kernel launches, skipped steps, and KV high-water on the right. Disconnect stops decode.

```bash
curl -N http://127.0.0.1:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-dart-pace: 30' \
  -d '{"model":"dart-synth-8b","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

```python
from dart import DartRuntime, SyntheticEngine, ReadingPacer
from dart.client import DartClient

rt = DartRuntime(SyntheticEngine())
async for seg in DartClient(runtime=rt).stream(
    "Explain named continuations.",
    consumer=ReadingPacer(tokens_per_sec=30),
    max_tokens=128,
):
    print(seg.text, end="", flush=True)
```

---

## Why it exists

Most serving stacks generate as fast as the GPU allows, then try to deliver. A token with no consumer credit is a wasted residual-stream step: energy, KV pages, and batch slots.

DART inverts that. An Interest is a compute capability. Zero Interests means zero decode. Cache hits never touch the kernel.

| Approach | Behavior |
|---|---|
| Push / “max occupancy” | Generate, then buffer |
| Andes | Generate, then pause the watermark |
| DART | Do not generate until credited |

---

## How it works

```text
Consumer                         DART                              Engine
   │  open(prompt)                  │  prefill (once)                 │
   │───────────────────────────────►│────────────────────────────────►│
   │  lease + kv_root               │                                 │
   │◄───────────────────────────────│                                 │
   │  Interest(name, window W)      │  CAS hit → Data, no GPU         │
   │───────────────────────────────►│  else credit += W, decode ≤ W   │
   │  Data + new kv_root            │────────────────────────────────►│
   │◄───────────────────────────────│◄────────────────────────────────│
```

1. **Named continuations** — live generation is an address space, not an HTTP request.
2. **Credit window** — `W` is tokens, not bytes. Congestion control is the scheduler.
3. **CAS then pin then decode** — named Data is free; pinned KV resumes without re-prefill; otherwise the cheapest producer runs.
4. **Handover** — changing machines is answering the next Interest from a new locator, not live-migrating a request.

---

## Engines

| Flag | Role |
|---|---|
| `--engine synthetic` | Deterministic CPU producer with real KV-extent accounting. Default for demo and tests. |
| `--engine vllm` | HTTP adapter to a live vLLM server (`DART_VLLM_URL`). Each Interest is `max_tokens=W`. Enable `--enable-prefix-caching`. |
| `--engine vllm-inprocess` | Waiting-queue plugin: admit once, park in `waiting` with pinned KV when `W=0`. |
| `--engine llamacpp` | HTTP adapter to llama.cpp (`DART_LLAMACPP_URL`). |
| `--engine cache` | CAS-only peer. Never decodes. |

```bash
export DART_ENGINE=vllm
export DART_MODEL=meta-llama/Llama-3.1-8B-Instruct
export DART_VLLM_URL=http://127.0.0.1:8000/v1
export DART_SECRET=replace-me
dart serve --engine vllm --model "$DART_MODEL"
```

---

## Operations

```bash
dart serve --engine synthetic --port 8090          # CIP + OpenAI facade
dart mesh --nodes 3 --connector nixl --port 8090   # in-process multi-node router
dart peer --cas-dir /var/dart/cas --port 8091      # FileCAS peer, no GPU
dart experiment --suite paper                      # kill-test, Andes, grammar, CAS
dart experiment --suite mesh
dart experiment --suite waiting
```

| Surface | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible facade. `X-Dart-Pace`, `X-Dart-Window`. Disconnect closes the continuation. |
| `POST /v1/continuations` + `.../interest` | CIP HTTP |
| `GET /v1/mesh`, `POST /v1/handover`, `GET /v1/kv` | Pin, route, adopt |
| `WS /v1/cip` | Framed Interest / Data / Nack |
| `GET /metrics`, `GET /v1/metrics` | Prometheus and JSON |

Environment: `DART_ENGINE`, `DART_MODEL`, `DART_SECRET`, `DART_CAS_DIR`, `DART_KV_CONNECTOR`, `DART_PRODUCER_ID`, `DART_PORT`.

---

## Repository layout

```text
src/dart/
  cli.py factory.py          Entry points
  core/                      Runtime, leases, congestion control, types
  cip/                       CIP names and Merkle kv_root
  kv/                        Token CAS, pinned KV, NIXL/LMCache connectors
  mesh/                      Interest router
  engine/                    Synthetic, vLLM HTTP, vLLM in-process, llama.cpp
  api/                       FastAPI gateway and demo UI
  client/                    SDK and pacers
  eval/                      Kill-test and paper suites
docs/                        Architecture, protocol, mesh, adapters
tests/                       pytest
```

Public imports stay stable:

```python
from dart import DartRuntime, Interest, ReadingPacer
from dart.client import DartClient
```

---

## Documentation

| Guide | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Control loop, process model, kill criteria |
| [Protocol](docs/protocol.md) | CIP names, Interest / Data / Nack, leases |
| [Congestion control](docs/congestion-control.md) | Window, AIMD, speculative K |
| [Engine adapters](docs/engine-adapters.md) | Synthetic, vLLM, llama.cpp |
| [Mesh](docs/mesh.md) | Pin, handover, routing order |
| [Waiting plugin](docs/vllm-plugin.md) | In-process `waiting` + pinned blocks |
| [Limitations](docs/limitations.md) | HTTP admission, prefix-cache eviction, scope |

```bash
pytest -q
```

---

## Status

Apache-2.0. DART is a control plane. vLLM and llama.cpp remain the kernels. The HTTP vLLM path re-enters admission; `--engine vllm-inprocess` keeps the request in `waiting` with pinned blocks. Real NIXL RDMA and LMCache GPU pages are optional; tests use in-process memcpy.
