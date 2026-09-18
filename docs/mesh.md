# Pinned KV, NIXL/LMCache handover, Interest mesh

DART’s continuation is `(lease, kv_root)`, not a vLLM request. This page is what the repo actually implements for pinning that object, moving it, and routing the next Interest.

## Pin by `kv_root`

`PinnedKVPool` (`src/dart/pin.py`) indexes **engine state** by Merkle `kv_root`:

- snapshot: token ids, sampler, extents, pos, segment index
- holder: which producer currently has it on GPU/host
- pin lease: sleep does not drop identity; expiry does

`DartRuntime.open` and every committed decode call `_publish_kv`. A second runtime calls `adopt(lease, kv_root=…)` and **does not call `engine.prefill`**. That is the in-process pin: resume is installing named KV, not thawing a process.

`KVStore` still tracks per-continuation GPU occupancy (`sleep` / `wake`). The pool is the **identity** index the mesh uses.

## Connectors (LMCache / NIXL shaped)

`src/dart/kvconn.py`:

| Kind | `put` / `get` | `transfer(src → dst)` | GPU / RDMA required |
|---|---|---|---|
| `memory` | in-process dict | copy + holder update | no |
| `file` | directory of JSON blobs | same, survives process restart | no |
| `lmcache` | same API as LMCache (key = `kv_root`) | via store | no; uses real `lmcache` only if installed |
| `nixl` | wraps a backend | accounts bytes/hops; `nixl-rdma` if `nixl` is importable, else `nixl-memcpy` | no for tests |

A connector miss is a miss. Decode already succeeded if `put` fails; the runtime logs and continues. Handover without a blob is `InterestNack unknown_name`, never a silent re-prefill.

## Routing order

`InterestRouter` (`src/dart/mesh.py`):

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

This is **not** a vLLM fleet scheduler plugin. HTTP vLLM/llama.cpp adapters still re-enter admission; their prefix cache may evict. The pin pool still names the continuation. SyntheticEngine (and any in-process `Engine` that decodes from `EngineState`) actually skips prefill on adopt.

Real NIXL RDMA / LMCache GPU pages are optional libraries. Tests use memcpy. Metrics (`rdma_available`, `lmcache_available`) say so.
