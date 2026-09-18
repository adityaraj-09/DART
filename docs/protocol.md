# Continuation Interest Protocol (CIP)

v1 is JSON over HTTP and WebSocket. It is **named computation**, not classic NDN: an Interest for a missing segment is a capability to run `decode(kv_root, pos, n)`. Intermediate nodes may aggregate Interests or refuse (no credit / no model). They may not mint decode credit.

## Address space

```text
/cip/<model-hash>/<kv-root>/tokens/seg/<i>
/cip/<model-hash>/<kv-root>/kv/layer/<ℓ>/blk/<b>
/cip/<model-hash>/<kv-root>/grammar/span/<s>
/cip/<model-hash>/<kv-root>/draft/k/<n>
```

- **Immutable Data** is named by `(model_hash, kv_root, kind, index)`. Never overwritten. Cacheable forever.  
- **`kv_root` in the name is the prefix identity** (root *before* this segment). The Data body carries `kv_root` after the commit so the next Interest can name the child. Historical names stay valid.  
- **Mutable cursor** is the `ContinuationLease` (capability). It is not a cache key.

`model_hash` is `ModelConfig.fingerprint()`: model id, tokenizer hash, RoPE θ, dtype, block size, layers, KV heads, head dim, vocab.

## Messages

### Interest

```json
{
  "name": "/cip/abc/.../tokens/seg/0",
  "window": 16,
  "lifetime_ms": 2000,
  "locator": "client-xyz",
  "kind": "tokens",
  "lease": "<payload>.<hmac>"
}
```

`window` is tokens, not bytes. `lifetime_ms` is `InterestLifetime`: on expiry the producer stops decode, pins KV, discards speculative draft.

Typed Interest: `kind=grammar` + `grammar_span=next-value` (or a name under `grammar/span/...`). The producer may emit a forced chunk in **one** Data — jump-forward as a named object, not a logit mask trapped in one process.

### Data

```json
{
  "name": "/cip/abc/.../tokens/seg/0",
  "text": "the named continuation ",
  "token_ids": [1, 2, 3],
  "kv_root": "<new>",
  "kv_root_prev": "<old>",
  "pos": 84,
  "cache_hit": false,
  "stopped": false
}
```

If a cache/peer already has the name for this `kv_root`, it answers with `cache_hit: true` and never touches a GPU.

### Nack

Reasons: `no_credit`, `no_model`, `busy`, `expired`, `amplification`, `unknown_name`, `done`.

## HTTP

| Method | Path | Meaning |
|---|---|---|
| `POST` | `/v1/continuations` | Prefill; return capability |
| `POST` | `/v1/continuations/{id}/interest` | Pull next object |
| `POST` | `/v1/continuations/{id}/ack` | Explicit consume ACK |
| `DELETE` | `/v1/continuations/{id}` | Zero credit, sleep KV |
| `GET` | `/v1/continuations/{id}` | Cursor + metrics |
| `WS` | `/v1/cip` | Framed Interest/Data/Nack/Ack |
| `POST` | `/v1/chat/completions` | OpenAI facade |

OpenAI facade headers:

- `X-Dart-Pace: 30` — grant credit at 30 tok/s (`ReadingPacer`).  
- `X-Dart-Window: 16` — segment size / burst.  
- Client disconnect cancels the generator and **closes the continuation**.

Without `X-Dart-Pace`, the facade uses `DrainPacer`: credit as each SSE chunk is pulled. That is already stronger than stock vLLM SSE, which keeps the kernel running after the gateway stops reading.

## Lease

`sign_lease` / `verify_lease` in `src/dart/lease.py`. Token is `base64url(json).base64url(hmac-sha256)`.

Fields: `cont_id`, `model_hash`, `model_id`, `kv_root`, `pos`, `w_max`, `decode_quota`, `expiry_unix`, `producer_hint`, `startup_credit`.

Secret: `DART_SECRET`. Default is a dev string; set it in production.

## MOQT later

`draft-liu-moq-live-agent-interaction` maps token *batches* onto MOQ objects and does not authorize decode. A future profile:

- Interest → `SUBSCRIBE` with a token window  
- Data → Object on `output/text`  
- grammar span → one Object / Subgroup  
- barge-in datagram → cancel outstanding Interests (stop decode)

Do not wait on that draft to ship v1.
