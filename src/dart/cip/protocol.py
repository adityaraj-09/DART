"""Continuation Interest Protocol (CIP) names and messages.

A live generation is an address space of objects, not an HTTP request:

    /cip/<model-hash>/<kv-root>/tokens/seg/<i>
    /cip/<model-hash>/<kv-root>/kv/layer/<ℓ>/blk/<b>
    /cip/<model-hash>/<kv-root>/grammar/span/<s>
    /cip/<model-hash>/<kv-root>/draft/k/<n>

Immutable Data is named by (model, kv_root, kind, index). The mutable
cursor is the ContinuationLease, which is a capability, not a cache key.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dart.core.errors import NameParseError
from dart.core.types import InterestKind, NackReason

_NAME = re.compile(
    r"^/cip/"
    r"(?P<model>[0-9a-f]{8,64})/"
    r"(?P<kv>[0-9a-f]{8,64})/"
    r"(?P<rest>.+)$"
)

_TOKENS = re.compile(r"^tokens/seg/(?P<i>\d+)$")
_KV = re.compile(r"^kv/layer/(?P<layer>\d+)/blk/(?P<block>\d+)$")
_GRAMMAR = re.compile(r"^grammar/span/(?P<span>[A-Za-z0-9_.-]+)$")
_DRAFT = re.compile(r"^draft/k/(?P<n>\d+)$")


class CipName(BaseModel):
    """Parsed CIP address. Immutable Data names are cache keys."""

    model_config = ConfigDict(protected_namespaces=())

    model_hash: str
    kv_root: str
    kind: InterestKind
    segment: int | None = None
    layer: int | None = None
    block: int | None = None
    span: str | None = None
    draft_k: int | None = None

    def render(self) -> str:
        base = f"/cip/{self.model_hash}/{self.kv_root}"
        if self.kind is InterestKind.TOKENS:
            return f"{base}/tokens/seg/{self.segment}"
        if self.kind is InterestKind.KV:
            return f"{base}/kv/layer/{self.layer}/blk/{self.block}"
        if self.kind is InterestKind.GRAMMAR:
            return f"{base}/grammar/span/{self.span}"
        if self.kind is InterestKind.DRAFT:
            return f"{base}/draft/k/{self.draft_k}"
        raise NameParseError(f"unknown kind {self.kind}")

    def __str__(self) -> str:  # pragma: no cover
        return self.render()

    @classmethod
    def parse(cls, name: str) -> CipName:
        m = _NAME.match(name)
        if not m:
            raise NameParseError(f"not a CIP name: {name!r}")
        rest = m.group("rest")
        model, kv = m.group("model"), m.group("kv")
        if t := _TOKENS.match(rest):
            return cls(
                model_hash=model,
                kv_root=kv,
                kind=InterestKind.TOKENS,
                segment=int(t.group("i")),
            )
        if t := _KV.match(rest):
            return cls(
                model_hash=model,
                kv_root=kv,
                kind=InterestKind.KV,
                layer=int(t.group("layer")),
                block=int(t.group("block")),
            )
        if t := _GRAMMAR.match(rest):
            return cls(
                model_hash=model,
                kv_root=kv,
                kind=InterestKind.GRAMMAR,
                span=t.group("span"),
            )
        if t := _DRAFT.match(rest):
            return cls(
                model_hash=model,
                kv_root=kv,
                kind=InterestKind.DRAFT,
                draft_k=int(t.group("n")),
            )
        raise NameParseError(f"unknown CIP object: {name!r}")

    @classmethod
    def tokens(cls, model_hash: str, kv_root: str, segment: int) -> CipName:
        return cls(
            model_hash=model_hash, kv_root=kv_root, kind=InterestKind.TOKENS, segment=segment
        )

    @classmethod
    def grammar(cls, model_hash: str, kv_root: str, span: str) -> CipName:
        return cls(model_hash=model_hash, kv_root=kv_root, kind=InterestKind.GRAMMAR, span=span)


class Interest(BaseModel):
    """Pull: the only thing that may authorize a decode kernel."""

    name: str
    window: int = Field(default=16, ge=1, le=4096)
    lifetime_ms: int = Field(default=2000, ge=10, le=120_000)
    locator: str = "default"
    kind: InterestKind = InterestKind.TOKENS
    grammar_span: str | None = None
    json_schema: dict[str, Any] | None = None
    cont_id: str | None = None
    lease: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        CipName.parse(v)
        return v

    @model_validator(mode="after")
    def _kind(self) -> Interest:
        parsed = CipName.parse(self.name)
        if self.kind is InterestKind.TOKENS and parsed.kind is not InterestKind.TOKENS:
            object.__setattr__(self, "kind", parsed.kind)
        if parsed.kind is InterestKind.GRAMMAR and not self.grammar_span:
            object.__setattr__(self, "grammar_span", parsed.span)
        return self

    @property
    def parsed(self) -> CipName:
        return CipName.parse(self.name)


class TokenSegment(BaseModel):
    index: int
    token_ids: list[int]
    text: str
    pos_begin: int
    pos_end: int


class Data(BaseModel):
    """Producer (or cache peer) response to an Interest."""

    name: str
    tokens: TokenSegment | None = None
    text: str = ""
    token_ids: list[int] = Field(default_factory=list)
    kv_root: str
    kv_root_prev: str
    pos: int
    sampler_hash: str = ""
    grammar_stack_hash: str = ""
    producer_id: str = ""
    cache_hit: bool = False
    draft: bool = False
    stopped: bool = False
    kind: InterestKind = InterestKind.TOKENS

    def token_count(self) -> int:
        if self.tokens:
            return len(self.tokens.token_ids)
        return len(self.token_ids)


class Nack(BaseModel):
    name: str
    reason: NackReason
    detail: str = ""
    retry_after_ms: int | None = None


class CipMessage(BaseModel):
    """WebSocket frame: one of Interest / Data / Nack / Ack."""

    type: Literal["interest", "data", "nack", "ack", "open", "close"]
    interest: Interest | None = None
    data: Data | None = None
    nack: Nack | None = None
    consumed: int | None = None
    cont_id: str | None = None
    lease: str | None = None
