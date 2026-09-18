# Congestion control

The scheduler **is** the congestion controller. Engine knobs are not yaml hyperparameters.

Implemented in `src/dart/cc.py` (`CongestionController`) and applied per continuation in `src/dart/runtime.py`.

## State

| Variable | Meaning |
|---|---|
| `cwnd` | Tokens the controller believes this continuation may have in flight |
| `credits` | Tokens authorized to decode, not yet produced |
| `in_flight` | Produced, waiting for consumer ACK / next Interest |
| `ssthresh` | Slow-start threshold |
| `rtt_s` | EWMA of Interest spacing |
| `K` | Speculative draft depth |

Invariant: `credits + in_flight ≤ min(cwnd, lease.w_max, w_max)`.

## Slow-start and AIMD

1. `cwnd = w_init` (default 16) so the first Interest can hide TTFT.  
2. Interest grants `min(window, room)`.  
3. Next Interest ACKs the previous segment (`on_ack`).  
   - If `cwnd < ssthresh`: `cwnd += consumed` (slow-start).  
   - Else: `cwnd += consumed / cwnd` (AIMD increase), capped at `w_max`.  
4. `InterestLifetime` expiry: `ssthresh = cwnd/2`, `cwnd = max(1, cwnd/2)`, credits dropped, draft discarded, KV pinned.

This is CUBIC/BBR-shaped in *intent*. v1 is AIMD because the signal (token ACK) is discrete and cheap. Replace `on_ack` / `on_timeout` without touching the engine.

## Speculative K

```text
K = clamp(0, round(cwnd * RTT / t_decode - cwnd), k_max)
```

`K` is computed and exported on every continuation (TCP-style prefetch *signal*). v1 does **not** run uncredited residual-stream steps: that would violate the inversion (a token with no consumer credit is wasted energy and KV). HTTP vLLM/llama.cpp adapters also cannot roll back a draft.

When a producer supports verified rollback (native KV snapshot), draft may run ahead of `W` and must be discarded on `InterestLifetime` expiry without being named in CAS. Until then, `tokens_drafted` counts the *would-have* prefetch, not GPU work.

## Grammar as a CC signal

Repeated Interest for a grammar nonterminal is not “need N tokens.” It is “need the next typed span.” The scheduler sets `grammar_span` and the producer may jump-forward. The span is one cacheable Data object; its token length still counts against quota and ACK.

## GPU occupancy

Credit-ready continuations are packed up to `max_num_seqs`. Continuations with `W=0` do not enter the batch.

If the credit-ready set is too small, **do not** mint unread tokens to fill the GPU. Fill with other paid work (prefill, embeddings, offline batch) at the fleet layer. The synthetic/single-node runtime sleeps instead — correct for the kill-test, incomplete for a packed cluster (documented, not swept under the rug).

## Handover

An Interest with a new `locator` is the migrate primitive. The runtime rebinds `producer_id`. A cluster implementation fetches named KV extents (NIXL pull, Llumnix-style copy) *because the Interest arrived*, not because a control plane rescheduled a request.

## What we refuse to do

- Use TCP byte windows as token credit.  
- Treat `max_tokens` abort/restart as a credit gate (re-enters admission, races async abort, loses KV).  
- Encode speculative tokens into the CAS (that would make timeout expensive and cache-poison peers).
