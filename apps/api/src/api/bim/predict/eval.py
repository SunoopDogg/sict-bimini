"""Evaluation harness for the RAG code Predictor.

Pure helpers to sample labeled records from an existing Qdrant
collection, re-predict with leave-one-out retrieval, aggregate
accuracy / mode / latency metrics, and write per-run reports under
``{BIM_DATA_ROOT}/reports/predict-eval/{UTC_ISO}_{target}/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from api.bim.predict.schemas import PredictionMode, TargetCode
from api.bim.schemas import BIMAttribute


@dataclass(frozen=True)
class EvalConfig:
    target: TargetCode
    ifc_type: str | None
    category: str | None
    limit: int | None
    seed: int
    top_k: int
    output_root: Path


@dataclass(frozen=True)
class EvalSample:
    stable_id: str
    attribute: BIMAttribute
    ground_truth: str


@dataclass(frozen=True)
class EvalOutcome:
    sample: EvalSample
    mode: PredictionMode | None
    top1: str | None
    top_k_codes: list[str]
    latency_ms: float
    pool_size: int
    error: str | None


class AggregatedMetrics(BaseModel):
    """Aggregated metrics produced by ``aggregate`` and serialized to summary.json."""

    filter_summary: dict[str, object] = Field(default_factory=dict)
    samples_total: int
    samples_with_error: int
    top1_correct: int
    top1_accuracy: float | None
    topk_correct: int
    topk_accuracy: float | None
    top_k: int
    mode_distribution: dict[str, int]
    accuracy_by_mode: dict[str, float | None]
    accuracy_by_ifc_type: dict[str, dict[str, int | float | None]]
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    errors_by_type: dict[str, int]
