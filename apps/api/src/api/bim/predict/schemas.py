"""Pydantic schemas for the predict module.

Split into three bands:

- Public request/response (``PredictionRequest``, ``PredictionResponse``, ``PredictionCandidate``)
- Internal retrieval state (``Neighbor``, ``CandidatePool``)
- Mode enum (``PredictionMode``)

Dynamic response schemas (mode-dependent ``code`` field type) are built
at call time in ``build_strong_schema`` / ``build_weak_schema``.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.bim.schemas import BIMAttribute


class PredictionMode(StrEnum):
    STRONG = "strong"
    WEAK = "weak"


class PredictionRequest(BaseModel):
    attribute: BIMAttribute
    n: int = Field(default=5, ge=1, le=20)


class PredictionCandidate(BaseModel):
    code: str
    llm_confidence: float = Field(ge=0.0, le=1.0)
    retrieval_score: float | None = None
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
    target: Literal["kbims_code", "pps_code"]
    mode: PredictionMode
    candidates: list[PredictionCandidate]   # 요청 n개, 실제 0..n (부분응답 허용)
    low_confidence_context: bool
    pool_size: int
    retrieved_k: int


class Neighbor(BaseModel):
    stable_id: str
    score: float
    kbims_code: str = ""
    pps_code: str = ""
    ifc_type: str
    category: str


class CandidatePool(BaseModel):
    code_to_max_score: dict[str, float]
    top1_score: float
    unique_count: int
