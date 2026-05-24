"""Evaluation harness for the RAG code Predictor.

Pure helpers to sample labeled records from an existing Qdrant
collection, re-predict with leave-one-out retrieval, aggregate
accuracy / mode / latency metrics, and write per-run reports under
``{BIM_DATA_ROOT}/reports/predict-eval/{UTC_ISO}_{target}/``.
"""
from __future__ import annotations

import json as _json
import logging
import random as _random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.bim.predict.errors import PredictError
from api.bim.predict.predictor import Predictor
from api.bim.predict.retriever import non_empty_code_condition
from api.bim.predict.schemas import PredictionMode, PredictionRequest, TargetCode
from api.bim.schemas import BIMAttribute

logger = logging.getLogger(__name__)


class NoSamplesError(ValueError):
    """Raised by ``fetch_samples`` when the filter matches zero labeled records.

    Subclass of ValueError to preserve the catch-all ``except ValueError``
    behavior for callers that don't distinguish; CLI callers that want
    to surface only this specific case can catch it narrowly.
    """


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
    must: list = [non_empty_code_condition(cfg.target)]
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
    prev_offset = None
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
        raise NoSamplesError(
            f"No samples match filter (target={cfg.target}, "
            f"ifc_type={cfg.ifc_type}, category={cfg.category})"
        )

    rng = _random.Random(cfg.seed)
    rng.shuffle(records)
    selected = records if cfg.limit is None else records[: cfg.limit]

    return [
        EvalSample(
            stable_id=r.payload.get("stable_id", ""),
            attribute=BIMAttribute.model_validate(r.payload),
            ground_truth=r.payload.get(cfg.target) or "",
        )
        for r in selected
    ]


def evaluate_one(
    sample: EvalSample,
    predictor: Predictor,
    top_k: int,
) -> EvalOutcome:
    """Run a single leave-one-out prediction; domain errors become error outcomes."""
    exclude_self = Filter(
        must_not=[
            FieldCondition(
                key="stable_id",
                match=MatchValue(value=sample.stable_id),
            )
        ]
    )
    t0 = time.perf_counter()
    try:
        resp = predictor.predict(
            PredictionRequest(attribute=sample.attribute, n=top_k),
            extra_filter=exclude_self,
        )
    except PredictError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return EvalOutcome(
            sample=sample,
            mode=None,
            top1=None,
            top_k_codes=[],
            latency_ms=latency_ms,
            pool_size=0,
            error=exc.__class__.__name__,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    top1 = resp.candidates[0].code if resp.candidates else None
    top_k_codes = [c.code for c in resp.candidates]
    return EvalOutcome(
        sample=sample,
        mode=resp.mode,
        top1=top1,
        top_k_codes=top_k_codes,
        latency_ms=latency_ms,
        pool_size=resp.pool_size,
        error=None,
    )


def aggregate(outcomes: list[EvalOutcome], cfg: EvalConfig) -> AggregatedMetrics:
    """Roll outcomes into summary metrics; every denominator-0 path yields None."""
    total = len(outcomes)
    errored = sum(1 for o in outcomes if o.error)

    def _hits(subset: list[EvalOutcome]) -> int:
        return sum(1 for o in subset if o.top1 == o.sample.ground_truth)

    top1_correct = _hits(outcomes)
    topk_correct = sum(
        1 for o in outcomes if o.sample.ground_truth in o.top_k_codes
    )

    mode_dist: dict[str, int] = {mode.name: 0 for mode in PredictionMode}
    mode_dist["error"] = 0
    for o in outcomes:
        if o.mode is None:
            mode_dist["error"] += 1
        else:
            mode_dist[o.mode.name] += 1

    acc_by_mode: dict[str, float | None] = {}
    for mode in PredictionMode:
        subset = [o for o in outcomes if o.mode is mode]
        acc_by_mode[mode.name] = (_hits(subset) / len(subset)) if subset else None

    groups: dict[str, list[EvalOutcome]] = defaultdict(list)
    for o in outcomes:
        groups[o.sample.attribute.ifc_type].append(o)
    by_ifc: dict[str, dict[str, int | float | None]] = {}
    for ifc, group in groups.items():
        correct = _hits(group)
        by_ifc[ifc] = {
            "total": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
        }

    latencies = [o.latency_ms for o in outcomes]
    if len(latencies) >= 2:
        q = statistics.quantiles(latencies, n=100)
        p50: float | None = q[49]
        p95: float | None = q[94]
    elif len(latencies) == 1:
        p50 = latencies[0]
        p95 = latencies[0]
    else:
        p50 = None
        p95 = None

    errors_by_type: dict[str, int] = dict(
        Counter(o.error for o in outcomes if o.error)
    )

    return AggregatedMetrics(
        filter_summary={
            "target": cfg.target,
            "ifc_type": cfg.ifc_type,
            "category": cfg.category,
            "limit": cfg.limit,
            "seed": cfg.seed,
            "top_k": cfg.top_k,
        },
        samples_total=total,
        samples_with_error=errored,
        top1_correct=top1_correct,
        top1_accuracy=(top1_correct / total) if total else None,
        topk_correct=topk_correct,
        topk_accuracy=(topk_correct / total) if total else None,
        top_k=cfg.top_k,
        mode_distribution=mode_dist,
        accuracy_by_mode=acc_by_mode,
        accuracy_by_ifc_type=by_ifc,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        errors_by_type=errors_by_type,
    )


def _timestamp(now: datetime | None = None) -> str:
    t = now if now is not None else datetime.now(UTC)
    return t.strftime("%Y-%m-%dT%H-%M-%SZ")


def write_report(
    metrics: AggregatedMetrics,
    outcomes: list[EvalOutcome],
    output_dir: Path,
) -> None:
    """Create ``output_dir`` and write summary.json + predictions.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        metrics.model_dump_json(indent=2),
        encoding="utf-8",
    )
    lines = [
        _json.dumps(
            {
                "stable_id": o.sample.stable_id,
                "ground_truth": o.sample.ground_truth,
                "top1": o.top1,
                "top_k": o.top_k_codes,
                "mode": o.mode.value if o.mode is not None else None,
                "pool_size": o.pool_size,
                "latency_ms": o.latency_ms,
                "error": o.error,
            },
            ensure_ascii=False,
        )
        for o in outcomes
    ]
    body = ("\n".join(lines) + "\n") if lines else ""
    (output_dir / "predictions.jsonl").write_text(body, encoding="utf-8")


def run_eval(
    cfg: EvalConfig,
    predictor: Predictor,
    qdrant: QdrantClient,
    *,
    collection: str,
) -> tuple[AggregatedMetrics, Path]:
    """Orchestrator: fetch → per-record predict → aggregate → write_report."""
    samples = fetch_samples(qdrant, collection, cfg)
    logger.info("fetched %d samples", len(samples))

    outcomes: list[EvalOutcome] = []
    for i, sample in enumerate(samples, start=1):
        outcomes.append(evaluate_one(sample, predictor, cfg.top_k))
        if i % 25 == 0:
            logger.info("progress %d/%d", i, len(samples))

    metrics = aggregate(outcomes, cfg)
    run_dir = cfg.output_root / f"{_timestamp()}_{cfg.target}"
    write_report(metrics, outcomes, run_dir)
    logger.info("wrote report to %s", run_dir)
    return metrics, run_dir
