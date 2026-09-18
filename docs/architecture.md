# DART architecture

**DART** (Demand-Addressed Runtime Tokens) is a named continuation runtime for autoregressive decode. **IDD** (Interest-Driven Decode) is the primitive:

> Outstanding Interests are the only thing that may run a decode kernel.

A live generation is not an HTTP request. It is an address space of objects. Consumers (compositor, TTS, JSON parser, tool runtime, another model) issue Interests for the next objects they can absorb. Producers decode only to satisfy those Interests. Any peer that already holds the named object can answer. Changing machines is not live-migration of a request; it is a different producer answering the next Interest.

This document is the system that is implemented in this repository. Paper-facing eval and the HTTP/vLLM honesty clause: [`limitations.md`](./limitations.md).

DART is a continuation runtime, not a GPU fleet that replaces vLLM. Live vLLM/llama.cpp behind CIP is already the adapter path; the unfinished slice is in-process pin + mesh routing.

**Paper suites:** `dart experiment --suite paper` (kill-test, Andes-complete, grammar jump vs mask, CAS peer).

---

## 1. Why this exists

Today: generate as fast as the GPU can, then try to deliver. The scheduler optimizes occupancy. The network is a pipe. The client is a sink.

A token that has no consumer credit is a wasted residual-stream step — wasted energy, wasted KV pages, wasted batch slot. Autoregression is the expensive encoder. **You should not encode segments nobody has credit to consume.**

Andes noticed humans read slower than GPUs generate, then still generated and buffered. DART refuses to generate.

---

## 2. Layer cake (what we own vs. what we reuse)

```text
┌─────────────────────────────────────────────────────────────┐
│ Clients: compositor · TTS · JSON parser · tool runtime      │
│ SDK pacers estimate receive window → Interests              │
└──────────────────────────────┬──────────────────────────────┘
                               │ CIP  (JSON/HTTP/WS v1)
┌──────────────────────────────▼──────────────────────────────┐
│ DART control plane                                          │
│  HMAC leases · anti-amplification · producer selection      │
│  Interest aggregation · congestion controller               │
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
   ┌──────────▼──────────┐         ┌──────────▼──────────┐
   │ Token CAS           │         │ Named KV extents    │
   │ Memory / FileCAS    │         │ pin · sleep · wake  │
   └──────────┬──────────┘         └──────────┬──────────┘
              │ miss                          │
   ┌──────────▼───────────────────────────────▼──────────┐
   │ Credit-gated scheduler                              │
   │ decode iff credits > 0; else skip kernel, pin KV    │
   └──────────────────────┬──────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   SyntheticEngine    VLLMChatEngine   LlamaCppEngine
   (CPU, tests,       (prod GPU,       (bench / edge)
    kill-test)         prefix-cached
                       max_tokens=W)
```

| Layer | Reuse | Invent |
|---|---|---|
| Decode kernel | vLLM V1, llama.cpp | Credit gate: `W=0` ⇒ unscheduled, KV pinned |
| Grammar jump-forward | vLLM xgrammar / llama.cpp GBNF | Typed Interest `grammar/span/<s>` as one named Data |
| KV move | NIXL / LMCache / Mooncake (ops) | Handover = next Interest from a new locator |
| Delivery | HTTP/2, WHATWG streams, SSE | Token window, not a byte window |
| Occupancy fill | Prefill, batch jobs | Never fill with unread tokens |

We **do not** ship a new CUDA kernel, NDN stack, or IETF draft. MOQT is a future *delivery* profile, not the decode authorizer.

---

## 3. Continuation object

After prefill, the runtime publishes a **capability**, not a vLLM process snapshot:

| Field | Role |
|---|---|
| `model_id` + fingerprint | tokenizer / RoPE / dtype / block size / GQA layout |
| `kv_root` | Merkle root over layer-block hashes (identity, not a ZK proof) |
| `seq_pos` | sampler hash + grammar stack |
| `W_max`, `decode_quota`, expiry | anti-amplification lease |
| `startup_credit` | slow-start so TTFT does not die |

Any compatible producer that can fetch the named KV extents may resume. The signed lease is the only credential an Interest needs.

---

## 4. Control loop

```text
Consumer                    Runtime                         Engine
   │                           │                               │
   │  open(prompt)             │  prefill                      │
   │──────────────────────────►│──────────────────────────────►│
   │  capability + kv_root     │                               │
   │◄──────────────────────────│                               │
   │                           │                               │
   │  Interest(seg/i, W, T)    │  if CAS hit: Data, no GPU     │
   │──────────────────────────►│                               │
   │                           │  credits += min(W, cwnd)      │
   │                           │  schedule iff credits > 0     │
   │                           │  decode(n ≤ credits)          │
   │                           │──────────────────────────────►│
   │                           │  Data + new kv_root + pos     │
   │◄──────────────────────────│◄──────────────────────────────│
   │                           │  CAS.put(name)                │
   │  next Interest (ACK)      │  on_ack → grow cwnd           │
```

Zero Interests: the scheduler increments `decode_steps_skipped`, **does not** call `engine.decode`, and sleeps KV extents off GPU while the pin lease holds.

Disconnect (SSE cancel, WS close, `DELETE /v1/continuations/{id}`): credits = 0, pending Interests NACK, KV sleeps. That is how “SSE backpressure” actually stops the kernel.

---

## 5. Congestion controller = scheduler

See [congestion-control.md](./congestion-control.md). Mapping:

| Signal | Knob |
|---|---|
| outstanding Interests / RTT | speculative `K` |
| repeated Interest for a grammar nonterminal | jump-forward span as one Data |
| `InterestLifetime` expiry | stop decode, pin KV, discard draft |
| duplicate Interests from a new locator | handover (Interest-triggered) |
| zero Interests | zero decode |
| next Interest | ACK + AIMD / slow-start |

---

## 6. Process model (production)

**Single node v1 (this repo):** one `DartRuntime` asyncio scheduler, one producer, in-process CAS. Enough to be a real primitive and to serve an OpenAI-compatible facade.

**Cluster v2 (ops, not a rewrite):**

1. Keep this runtime as the credit gate in front of each decode worker.  
2. Put token Data in FileCAS / object storage keyed by CIP name.  
3. Put KV extents in LMCache; `kv_root` is the identity NIXL already pulls.  
4. A new locator answering an Interest is a decode worker that `get_num_new_matched_tokens` hits on that root.  
5. Do **not** migrate a “request.” The HTTP request is gone; the continuation object remains.

**vLLM long-term:** a scheduler plugin that keeps the request in `waiting` with pinned blocks when `W=0`. The HTTP adapter here uses prefix-cached `max_tokens=W` so we do not fork vLLM to ship. The *invariant* is the same: no generate() without credit.

---

## 7. Security and billing

An Interest is a compute capability.

- HMAC-SHA256 lease (`src/dart/lease.py`), secret `DART_SECRET`.  
- `window ≤ w_max` (default 128). Cache hits are free; decode spends `decode_quota`.  
- Duplicate Interests for the same name coalesce (no extra credit).  
- Billing unit: joules and KV-bytes per **consumed** token, plus a prefill fee.

---

## 8. Public surfaces

| Surface | Path | Who uses it |
|---|---|---|
| Demo compositor | `GET /` | humans, QA |
| CIP HTTP | `POST /v1/continuations`, `.../interest` | mesh, tools |
| CIP WebSocket | `/v1/cip` | low-latency consumers |
| OpenAI facade | `POST /v1/chat/completions` | existing apps; `X-Dart-Pace`, `X-Dart-Window` |
| Metrics | `GET /metrics` Prometheus, `GET /v1/metrics` JSON | SRE |
| SDK | `dart.sdk.DartClient` + pacers | product code |
| CLI | `dart serve`, `dart experiment` | operators, kill-test |

Pacers (`src/dart/consumers.py`):

- `DrainPacer` — API drainer; credit as fast as the caller pulls.  
- `ReadingPacer(tokens_per_sec=30)` — compositor.  
- `TtsPacer(realtime_factor=1)` — mouth as credit source.  
- `JsonNeedPacer` — burst then pause (tool args / JSON value).

---

## 9. Module map

| Module | Responsibility |
|---|---|
| `dart.protocol` | CIP names, Interest / Data / Nack |
| `dart.lease` | signed continuation capability |
| `dart.cc` | window, AIMD, speculative K |
| `dart.store` | token CAS, KV pin/sleep |
| `dart.merkle` | `kv_root` identity |
| `dart.runtime` | scheduler + continuation table |
| `dart.engine.*` | Synthetic / vLLM / llama.cpp |
| `dart.gateway` | FastAPI |
| `dart.sdk` | product client |
| `dart.experiment` | push vs credit kill-test |

---

## 10. Kill criteria (still in force)

1. **Andes-complete:** if coupling a watermark pause to vLLM captures ≥90% of the joule/KV win, ship that patch and stop treating CIP as the product. Run `dart experiment` first.  
2. **Batch collapse:** if credit gating drops GPU util so far that consumed tok/J gets worse, enable fill-from-other-demand (prefill/batch) before declaring death.  
3. **Name tax:** keep segments 8–32 tokens. Per-token names are forbidden.  
4. **Prior art hit:** a system that already gates the **decode kernel** on named consumer Interests.

---

## 11. What “production” means here

- The decode kernel is gated on a token credit window, not on HTTP socket buffering that never reaches the engine.  
- Objects are named and cacheable; a second process can satisfy an Interest from FileCAS.  
- Leases prevent amplification.  
- OpenAI clients keep working; they opt into pace with a header, and disconnect stops decode.  
- GPU producers are adapters. The runtime does not care whether the kernel is synthetic, vLLM, or llama.cpp.

That is a continuation runtime with a pull scheduler. Not a dashboard.
