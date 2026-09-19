# Pinned KV, NIXL/LMCache handover, Interest mesh

DART’s continuation is `(lease, kv_root)`, not a vLLM request. This page is what the repo actually implements for pinning that object, moving it, and routing the next Interest.

## Pin by `kv_root`

`PinnedKVPool` (`src/dart/kv/pin.py`) indexes **engine state** by Merkle `kv_root`. `FilePinnedKVPool` persists that table next to FileCAS:

- snapshot: token ids, sampler, extents, pos, segment index
- holder: which producer currently has it on GPU/host
- pin lease: sleep does not drop identity; expiry does

`DartRuntime.open` and every committed decode call `_publish_kv`. A second runtime calls `adopt(lease, kv_root=…)` and **does not call `engine.prefill`**. That is the in-process pin: resume is installing named KV, not thawing a process.

`KVStore` still tracks per-continuation GPU occupancy (`sleep` / `wake`). The pool is the **identity** index the mesh uses.

## Connectors (LMCache / NIXL shaped)

`src/dart/kv/kvconn.py`:

| Kind | `put` / `get` | `transfer(src → dst)` | GPU / RDMA required |
|---|---|---|---|
| `memory` | page table + metadata | move `KVPage` buffers | no |
| `file` | directory of JSON blobs + pages | same, survives process restart | no |
| `lmcache` | GPU pages keyed by `kv_root` | via LMCache if installed, else page buffers | no for tests |
| `nixl` | wraps a backend | NIXL RDMA of pages when `nixl` posts; else `nixl-pages` | no for tests |

A connector miss is a miss. Decode already succeeded if `put` fails; the runtime logs and continues. Handover without a blob is `InterestNack unknown_name`, never a silent re-prefill.

## Routing order

`InterestRouter` (`src/dart/mesh/router.py`):

```text
Interest
   │
   ├─ 1. CAS already has the name  → Data, cache_hit, no GPU
   ├─ 2. Live pin holder           → that node’s runtime.interest
   ├─ 3. Live continuation (desync)→ that node
   └─ 4. Cheapest node + pull      → connector.transfer, adopt, then decode
```

Cheapest is `cost × (1 + awake seqs) + gpu_bytes`. A pin holder is preferred even when it is expensive: moving KV to a cheaper GPU is a handover, not the default.

Handover:

1. `release_for_handover` on the old node (pending Interests NACK `busy`; CAS still answers).
2. `transfer` named KV to the dest (NIXL-shaped).
3. dest `adopt` — `prefills_skipped += 1`.
4. dest satisfies the Interest.

Per-`kv_root` asyncio lock: two concurrent Interests cannot split-brain adopt.

## HTTP / CLI

| Surface | Meaning |
|---|---|
| `GET /v1/kv?root=` | pin table or connector lookup |
| `POST /v1/handover` | adopt onto this process |
| `GET /v1/mesh` | nodes, pins, route counters |
| `POST /v1/mesh/interest` | router.route (CAS → pin → handover) |
| `dart serve --connector nixl` | single node + connector |
| `dart mesh --nodes 3 --connector nixl` | in-process mesh |
| `dart experiment --suite mesh` | pin / CAS / holder / handover assertions |

## What this is not

This is **not** a GPU fleet. HTTP vLLM/llama.cpp adapters still re-enter admission; their prefix cache may evict. `--engine vllm` keeps the request in `waiting` with pinned blocks. The pin pool still names the continuation across nodes.

Handover is `connector.transfer` of `KVPage`s, then `adopt(kv_root)` — **zero** `engine.prefill`. Real NIXL RDMA / LMCache GPU pages run when those libraries are installed. Metrics (`rdma_available`, `lmcache_available`, `transport`) say so.
