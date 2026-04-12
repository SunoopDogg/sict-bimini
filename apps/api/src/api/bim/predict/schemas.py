"""Pydantic schemas for the predict module.

Split into three bands:

- Public request/response
  (``PredictionRequest``, ``PredictionResponse``, ``PredictionCandidate``)
- Internal retrieval state (``Neighbor``, ``CandidatePool``)
- Mode enum (``PredictionMode``)

Dynamic response schemas (mode-dependent ``code`` field type) are built
at call time in ``build_strong_schema`` / ``build_weak_schema``.
"""
from __future__ import annotations

import functools
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.bim.schemas import BIMAttribute

# Single source of truth for the prediction target identifier.
TargetCode = Literal["kbims_code", "pps_code"]


class PredictionMode(StrEnum):
    STRONG = "strong"
    WEAK = "weak"


class PredictionRequest(BaseModel):
    attribute: BIMAttribute
    n: int = Field(default=5, ge=1, le=20)


class PredictionCandidate(BaseModel):
    code: str
    llm_confidence: float = Field(ge=0.0, le=1.0)
    retrieval_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: Literal["neighbor", "generated"]
    reasoning: str | None = None

    @model_validator(mode="after")
    def _check_source_score_consistency(self) -> PredictionCandidate:
        if self.source == "neighbor" and self.retrieval_score is None:
            raise ValueError("source='neighbor' requires retrieval_score")
        if self.source == "generated" and self.retrieval_score is not None:
            raise ValueError("source='generated' must not have retrieval_score")
        return self


class PredictionResponse(BaseModel):
    target: TargetCode
    mode: PredictionMode
    candidates: list[PredictionCandidate]   # 요청 n개, 실제 0..n (부분응답 허용)
    low_confidence_context: bool
    pool_size: int = Field(ge=0)
    retrieved_k: int = Field(ge=0)


class Neighbor(BaseModel):
    stable_id: str
    score: float = Field(ge=0.0, le=1.0)
    kbims_code: str = ""
    pps_code: str = ""
    ifc_type: str
    category: str


class CandidatePool(BaseModel):
    code_to_max_score: dict[str, float]
    top1_score: float = Field(ge=0.0, le=1.0)
    unique_count: int = Field(ge=0)


@functools.lru_cache(maxsize=512)
def build_strong_schema(pool_codes: frozenset[str]) -> type[PredictionResponse]:
    """Build a PredictionResponse subclass whose code field is Literal[*pool_codes].

    Cached by the frozenset of pool codes — Predictor calls tend to re-see
    the same pool across retries/similar queries.

    Requires a non-empty pool. Caller (Predictor) only invokes this path
    when evaluate_mode returns STRONG, which implies pool_size >= n >= 1.
    """
    if not pool_codes:
        raise ValueError("build_strong_schema requires non-empty pool_codes")

    # Literal[*...] needs a tuple for ordered semantics; frozenset is unordered
    # but Literal compares by set membership, not order. Sorted for stability.
    code_type = Literal[*sorted(pool_codes)]  # type: ignore[valid-type]

    class _StrongCandidate(PredictionCandidate):
        code: code_type        # type: ignore[valid-type]
        source: Literal["neighbor"]

    class _StrongResponse(PredictionResponse):
        candidates: list[_StrongCandidate]
        mode: Literal[PredictionMode.STRONG]

    return _StrongResponse


@functools.lru_cache(maxsize=4)
def build_weak_schema(code_regex: str) -> type[PredictionResponse]:
    """Build a PredictionResponse subclass whose code field is str + pattern."""

    class _WeakCandidate(PredictionCandidate):
        code: str = Field(pattern=code_regex)

    class _WeakResponse(PredictionResponse):
        candidates: list[_WeakCandidate]
        mode: Literal[PredictionMode.WEAK]

    return _WeakResponse


_json_schema_cache: dict[int, dict] = {}


def get_json_schema(cls: type[PredictionResponse]) -> dict:
    """Return (and cache) the JSON schema dict for a dynamic response class."""
    cached = _json_schema_cache.get(id(cls))
    if cached is None:
        cached = cls.model_json_schema()
        _json_schema_cache[id(cls)] = cached
    return cached
