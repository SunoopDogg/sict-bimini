"""Evaluation harness for the RAG code Predictor.

Pure helpers to sample labeled records from an existing Qdrant
collection, re-predict with leave-one-out retrieval, aggregate
accuracy / mode / latency metrics, and write per-run reports under
``{BIM_DATA_ROOT}/reports/predict-eval/{UTC_ISO}_{target}/``.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchExcept,
    MatchValue,
)

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


_SCROLL_PAGE_SIZE = 256


def _build_scroll_filter(cfg: EvalConfig) -> Filter:
    must: list = [
        FieldCondition(key=cfg.target, match=MatchExcept(**{"except": [""]})),
    ]
    if cfg.ifc_type is not None:
        must.append(
            FieldCondition(key="ifc_type", match=MatchValue(value=cfg.ifc_type))
        )
    if cfg.category is not None:
        must.append(
            FieldCondition(key="category", match=MatchValue(value=cfg.category))
        )
    return Filter(must=must)


def fetch_samples(
    qdrant: QdrantClient,
    collection: str,
    cfg: EvalConfig,
) -> list[EvalSample]:
    """Scroll the collection for labeled records, shuffle by seed, apply limit."""
    scroll_filter = _build_scroll_filter(cfg)
    records: list = []
    offset = None
    prev_offset: object = object()  # distinct sentinel
    while True:
        page, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
        )
        records.extend(page)
        if offset is None:
            break
        if offset == prev_offset:
            raise RuntimeError(
                f"Qdrant scroll returned identical offset twice ({offset!r}); "
                "aborting to prevent infinite loop"
            )
        prev_offset = offset

    if not records:
        raise ValueError(
            f"No samples match filter (target={cfg.target}, "
            f"ifc_type={cfg.ifc_type}, category={cfg.category})"
        )

    rng = _random.Random(cfg.seed)
    rng.shuffle(records)
    selected = records if cfg.limit is None else records[: cfg.limit]

    return [
        EvalSample(
            stable_id=r.payload["stable_id"],
            attribute=BIMAttribute.model_validate(r.payload),
            ground_truth=r.payload[cfg.target],
        )
        for r in selected
    ]
