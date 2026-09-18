# Honest limitations

DART’s claim is the inversion: **outstanding Interests are the only thing that may run a decode kernel.** The HTTP adapters implement that invariant without forking vLLM. They also have limits a reviewer will probe. This page is the paper’s limitations paragraph, not a disclaimer dump.

## What this is not (scope)

**Do state this.** DART is not a finished GPU fleet that replaces vLLM. It is a continuation runtime and CIP control plane. vLLM and llama.cpp remain the kernels.

**Do not state the stale version.** “The next slice is a live vLLM/llama.cpp process behind the same CIP” is already true: `dart serve --engine vllm` and `--engine llamacpp` sit in front of those processes today. That is the production GPU path, not future work.

**What is actually next** (engineering, not the idea):

- In-process vLLM scheduler plugin: request stays in `waiting` with pinned KV when `W=0` (avoids HTTP admission re-entry and prefix-cache eviction). The control-plane pin (`PinnedKVPool`) and mesh router already exist for in-process engines; they do not pin vLLM’s GPU blocks over HTTP.
- Optional real LMCache GPU pages / NIXL RDMA. The connectors are in-repo (`memory` / `file` / `lmcache` / `nixl`); without those libraries, transfer is memcpy and metrics say `rdma_available=false`.

Named KV handover and multi-node Interest routing are implemented. See [`mesh.md`](./mesh.md). Do not promise a fleet you did not build, and do not hide the adapters you did.

## HTTP path re-enters admission

`VLLMChatEngine` / `LlamaCppEngine` turn each credited Interest into a new `chat.completions` / `/completion` with `max_tokens=W` (resp. `n_predict=W`). That **re-enters the engine’s request admission path**. It is not a request that stayed in `scheduler.waiting` with its blocks pinned.

Consequences:

- A load spike can 503 the next Interest even though the continuation is live.
- Async abort/restart races that in-process `pause_generation(mode="keep")` was built to avoid still exist on this path.
- TTFT of Interest 0 includes remote prefill; later Interests rely on **prefix caching**, not a pinned decode request.

An in-process vLLM scheduler plugin that leaves the request in `waiting` when `W=0` would **tighten KV residency and admission**. It would not invent credit-gated decode. The scientific claim does not depend on that plugin.

## Prefix cache can drop KV

`--enable-prefix-caching` / llama.cpp `cache_prompt=true` keep KV warm **until eviction**. Under memory pressure the engine may drop the prefix. The next generate then **re-prefills**. DART records that as `engine_admission_reentries` / `engine_prefix_cache_misses` when `usage.prompt_tokens` jumps.

IDD still did not decode without credit. It may pay a re-prefill it would have avoided with pinned blocks. Measure `engine_probe.match` (`GET /v1/engine`): local HTTP POSTs vs the engine process’s own `/metrics` forward counter.

## What we do measure (engine, not wrapper)

| Counter | Source |
|---|---|
| `engine_kernel_launches` | Incremented only after a successful generate/completion **round-trip** |
| `vllm_engine_forward_calls_total` | Scraped from the vLLM (or fake) process `/metrics` |
| `llamacpp_decode_calls_total` | Scraped from llama.cpp `/metrics` |
| `dart_decode_kernel_launches` | Runtime scheduler ticks that called `engine.decode` and got `kernel_launched=True` |

Paper tables must show the **engine** column. A credit-gated run with no Interests must leave the engine forward counter unchanged after prefill bookkeeping (HTTP adapters do not even POST on `open()`).

## Andes-complete

`dart experiment --suite andes` runs push vs Andes watermark pause vs IDD refuse-to-decode. If Andes captures ≥90% of IDD’s generated-token win vs push, IDD is an Andes patch (kill criterion 1). Inventory (`pacer_inventory_hw`) is the tell: Andes still holds unread tokens; IDD does not.

## Named CAS peer

`dart peer --cas-dir DIR` serves `POST /v1/peer/interest` from FileCAS only (`CacheOnlyEngine`). A second process answering a name with `cache_hit=true` and `engine_kernel_launches=0` is the not-Andes defense: the continuation object is the cache/handover unit.

## Grammar

Jump-forward: one Interest, one kernel, one Data. Logit masking: one kernel per constrained token. `dart experiment --suite grammar`.
