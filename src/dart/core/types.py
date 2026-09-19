"""Shared value objects for DART."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InterestKind(str, Enum):
    TOKENS = "tokens"
    GRAMMAR = "grammar"
    DRAFT = "draft"
    KV = "kv"


class NackReason(str, Enum):
    NO_CREDIT = "no_credit"
    NO_MODEL = "no_model"
    BUSY = "busy"
    EXPIRED = "expired"
    AMPLIFICATION = "amplification"
    UNKNOWN_NAME = "unknown_name"
    DONE = "done"


class ModelConfig(BaseModel):
    """Identity of a compatible producer — not a vLLM process snapshot."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    tokenizer_hash: str = "synth-v1"
    rope_theta: float = 10000.0
    dtype: str = "f16"
    block_size: int = 16
    n_layers: int = 8
    n_kv_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 32000

    def fingerprint(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for part in (
            self.model_id,
            self.tokenizer_hash,
            str(self.rope_theta),
            self.dtype,
            str(self.block_size),
            str(self.n_layers),
            str(self.n_kv_heads),
            str(self.head_dim),
            str(self.vocab_size),
        ):
            h.update(part.encode())
            h.update(b"|")
        return h.hexdigest()[:16]


class ChatMessage(BaseModel):
    role: str
    content: str


class Prompt(BaseModel):
    text: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    token_ids: list[int] = Field(default_factory=list)

    def as_text(self) -> str:
        if self.text:
            return self.text
        if self.messages:
            parts: list[str] = []
            for m in self.messages:
                parts.append(f"{m.role}: {m.content}")
            parts.append("assistant:")
            return "\n".join(parts)
        return ""


class KVExtent(BaseModel):
    layer: int
    block_id: int
    digest: str
    nbytes: int
    on_gpu: bool = True


class SamplerState(BaseModel):
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 0
    hash: str = "init"

    def advance(self, token_id: int) -> SamplerState:
        import hashlib

        h = hashlib.sha256(f"{self.hash}:{token_id}".encode()).hexdigest()[:16]
        return self.model_copy(update={"hash": h})


class GrammarStack(BaseModel):
    json_schema: dict[str, Any] | None = None
    nonterminal: str | None = None
    depth: int = 0


class EngineState(BaseModel):
    prompt_ids: list[int] = Field(default_factory=list)
    output_ids: list[int] = Field(default_factory=list)
    sampler: SamplerState = Field(default_factory=SamplerState)
    grammar: GrammarStack = Field(default_factory=GrammarStack)
    kv_extents: list[KVExtent] = Field(default_factory=list)
    kv_root: str = "0" * 64
    pos: int = 0
    stopped: bool = False
    prefix_text: str = ""
    assistant_text: str = ""
    engine_request_id: str = ""

    @property
    def all_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.output_ids]


class ContinuationMetrics(BaseModel):
    tokens_generated: int = 0
    tokens_consumed: int = 0
    tokens_drafted: int = 0
    tokens_draft_discarded: int = 0
    decode_kernel_launches: int = 0
    decode_steps_skipped: int = 0
    cache_hits: int = 0
    interests: int = 0
    nacks: int = 0
    kv_bytes_high_water: int = 0
    prefill_tokens: int = 0
    grammar_spans: int = 0
    joules: float = 0.0
    engine_kernel_launches: int = 0
    engine_tokens_predicted: int = 0
    admission_reentries: int = 0
    prefix_cache_misses: int = 0
    pacer_inventory_hw: int = 0
    grammar_masked_launches: int = 0
    grammar_jump_launches: int = 0
    prefills_skipped: int = 0
    handovers: int = 0
    pin_hits: int = 0

    def snapshot(self) -> dict[str, Any]:
        unused = max(0, self.tokens_generated - self.tokens_consumed)
        ratio = (
            self.tokens_generated / self.tokens_consumed
            if self.tokens_consumed
            else float("inf")
            if self.tokens_generated
            else 0.0
        )
        return {
            **self.model_dump(),
            "tokens_generated_unconsumed": unused,
            "generated_over_consumed": ratio,
        }


class RuntimeConfig(BaseModel):
    producer_id: str = "local-0"
    segment_size: int = 16
    w_init: int = 16
    w_max: int = 128
    k_max: int = 8
    max_num_seqs: int = 32
    poll_interval_s: float = 0.005
    lease_ttl_s: float = 600.0
    decode_quota: int = 512
    interest_lifetime_s: float = 2.0
    t_decode_s: float = 0.02
    rtt_init_s: float = 0.05
    startup_credit: int = 16
    secret: str = "dart-dev-secret-change-me"
    secret_previous: str | None = None
    cas_dir: str | None = None
    pin_dir: str | None = None
    tenant_interest_quota: int = 0
    default_tenant: str = "default"
    fill_with_idle_sleep: bool = True

    def secrets(self) -> list[str]:
        out = [self.secret]
        if self.secret_previous:
            out.append(self.secret_previous)
        return out

    @field_validator("segment_size")
    @classmethod
    def _seg(cls, v: int) -> int:
        if not 1 <= v <= 128:
            raise ValueError("segment_size must be in [1, 128]")
        return v
