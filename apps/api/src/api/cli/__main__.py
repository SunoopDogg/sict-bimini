from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import get_args

import httpx
import typer
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient
from api.bim.clients.qdrant import QdrantWrapper
from api.bim.clients.vllm import VLLMClient
from api.bim.pipeline import run_ingest_xlsx, run_normalize, run_upsert_qdrant
from api.bim.predict.eval import (
    AggregatedMetrics,
    EvalConfig,
    NoSamplesError,
    run_eval,
)
from api.bim.predict.factory import build_kbims_predictor, build_pps_predictor
from api.bim.predict.schemas import TargetCode
from api.core.config import BIMSettings

app = typer.Typer(add_completion=False, no_args_is_help=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _settings_from_args(
    experiment_id: str | None,
    dim: int | None,
    embedding_url: str | None,
    qdrant_url: str | None,
    model: str | None,
    data_root: Path | None,
) -> BIMSettings:
    """Env → BIMSettings, then overlay explicit CLI args."""
    overrides = {
        k: v
        for k, v in {
            "experiment_id": experiment_id,
            "embedding_dim": dim,
            "embedding_url": embedding_url,
            "qdrant_url": qdrant_url,
            "embedding_model": model,
            "data_root": data_root,
        }.items()
        if v is not None
    }
    base = BIMSettings()
    return base.model_copy(update=overrides) if overrides else base


_DataRoot = typer.Option(
    None,
    help="Root directory for xlsx/json data. Defaults to BIM_DATA_ROOT env or 'data'.",
)


def _resolve_data_root(data_root: Path | None) -> Path:
    """None → env-driven BIMSettings.data_root; explicit value wins."""
    if data_root is not None:
        return data_root
    return BIMSettings().data_root


_ExpId = typer.Option(None, help="Override BIM_EXPERIMENT_ID.")
_Dim = typer.Option(None, help="Override BIM_EMBEDDING_DIM.")
_EmbeddingUrl = typer.Option(None, help="Override BIM_EMBEDDING_URL.")
_QdrantUrl = typer.Option(None, help="Override BIM_QDRANT_URL.")
_Model = typer.Option(None, help="Override BIM_EMBEDDING_MODEL.")


@app.command("ingest-xlsx")
def ingest_xlsx_cmd(data_root: Path | None = _DataRoot) -> None:
    """Stage 1: xlsx → data/json/raw/<stem>.json"""
    run_ingest_xlsx(_resolve_data_root(data_root))


@app.command("normalize")
def normalize_cmd(data_root: Path | None = _DataRoot) -> None:
    """Stage 2: raw JSON → normalized JSON (BIMAttribute per source)."""
    run_normalize(_resolve_data_root(data_root))


@app.command("upsert-qdrant")
def upsert_qdrant_cmd(
    data_root: Path | None = _DataRoot,
    experiment_id: str | None = _ExpId,
    dim: int | None = _Dim,
    embedding_url: str | None = _EmbeddingUrl,
    qdrant_url: str | None = _QdrantUrl,
    model: str | None = _Model,
) -> None:
    """Stage 3: normalized JSON → vLLM embeddings → Qdrant upsert."""
    s = _settings_from_args(
        experiment_id, dim, embedding_url, qdrant_url, model, data_root
    )
    with VLLMEmbedClient(
        url=s.embedding_url, model=s.embedding_model, dim=s.embedding_dim
    ) as embed:
        qw = QdrantWrapper.from_settings(url=s.qdrant_url, api_key=s.qdrant_api_key)
        run_upsert_qdrant(
            data_root=s.data_root,
            embed_client=embed,
            qdrant=qw,
            collection=s.collection_name,
            dim=s.embedding_dim,
        )


@app.command("pipeline")
def pipeline_cmd(
    data_root: Path | None = _DataRoot,
    experiment_id: str | None = _ExpId,
    dim: int | None = _Dim,
    embedding_url: str | None = _EmbeddingUrl,
    qdrant_url: str | None = _QdrantUrl,
    model: str | None = _Model,
) -> None:
    """Run all 3 stages sequentially."""
    s = _settings_from_args(
        experiment_id, dim, embedding_url, qdrant_url, model, data_root
    )
    run_ingest_xlsx(s.data_root)
    run_normalize(s.data_root)
    with VLLMEmbedClient(
        url=s.embedding_url, model=s.embedding_model, dim=s.embedding_dim
    ) as embed:
        qw = QdrantWrapper.from_settings(url=s.qdrant_url, api_key=s.qdrant_api_key)
        run_upsert_qdrant(
            data_root=s.data_root,
            embed_client=embed,
            qdrant=qw,
            collection=s.collection_name,
            dim=s.embedding_dim,
        )


@app.command("llm-check")
def llm_check_cmd() -> None:
    """Probe the external vLLM server and verify BIM_LLM_MODEL is served."""
    s = BIMSettings()
    try:
        resp = httpx.get(
            f"{s.llm_url.rstrip('/')}/v1/models",
            timeout=s.llm_timeout_seconds,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        typer.echo(f"Failed to reach vLLM at {s.llm_url}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    served = [item["id"] for item in resp.json().get("data", [])]
    if s.llm_model not in served:
        typer.echo(
            f"Model {s.llm_model!r} is not served by vLLM at {s.llm_url}. "
            f"Served: {served}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"OK: vLLM at {s.llm_url} serves {s.llm_model}")


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "n/a"


def _format_summary(metrics: AggregatedMetrics, run_dir: Path) -> str:
    filt = metrics.filter_summary
    filter_parts: list[str] = []
    if filt.get("ifc_type"):
        filter_parts.append(f"ifc_type={filt['ifc_type']}")
    if filt.get("category"):
        filter_parts.append(f"category={filt['category']}")
    filter_str = " AND ".join(filter_parts) if filter_parts else "none"

    lines = [
        f"=== predict-eval [{filt.get('target')}] ===",
        f"Samples: {metrics.samples_total} (filter: {filter_str})",
        f"Top-1 accuracy: {_pct(metrics.top1_accuracy)} "
        f"({metrics.top1_correct}/{metrics.samples_total})",
        f"Top-{metrics.top_k} accuracy: {_pct(metrics.topk_accuracy)} "
        f"({metrics.topk_correct}/{metrics.samples_total})",
        "Mode distribution:   "
        + "  ".join(
            f"{k}={v}" for k, v in metrics.mode_distribution.items() if v
        ),
        "Accuracy by mode:    "
        + "  ".join(
            f"{k}={_pct(v)}" for k, v in metrics.accuracy_by_mode.items()
        ),
        "ifc_type breakdown:",
    ]
    for ifc, stats in metrics.accuracy_by_ifc_type.items():
        lines.append(
            f"  {ifc:<16}  {stats['total']}  top1={_pct(stats['accuracy'])}"
        )
    lines.append(
        f"Latency p50={metrics.latency_p50_ms:.0f}ms  "
        f"p95={metrics.latency_p95_ms:.0f}ms"
        if metrics.latency_p50_ms is not None
        else "Latency: n/a"
    )
    if metrics.errors_by_type:
        lines.append(
            "Errors: "
            + "  ".join(
                f"{k}={v}" for k, v in metrics.errors_by_type.items()
            )
        )
    lines.append(f"Report: {run_dir}")
    return "\n".join(lines)


@app.command("predict-eval")
def predict_eval_cmd(
    target: str = typer.Option(
        ..., "--target", help="Code field to evaluate: kbims_code or pps_code."
    ),
    ifc_type: str | None = typer.Option(
        None, "--ifc-type", help="Optional payload filter on ifc_type."
    ),
    category: str | None = typer.Option(
        None, "--category", help="Optional payload filter on category."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Upper bound on samples (default: entire matching set)."
    ),
    seed: int = typer.Option(0, "--seed", help="Shuffling seed for reproducibility."),
    top_k: int = typer.Option(
        5, "--top-k", help="Candidates requested per prediction."
    ),
) -> None:
    """Leave-one-out evaluation of Predictor against labeled Qdrant records."""
    valid_targets = get_args(TargetCode)
    if target not in valid_targets:
        typer.echo(
            f"--target must be one of {valid_targets}, got {target!r}", err=True
        )
        raise typer.Exit(code=1)

    s = BIMSettings()
    cfg = EvalConfig(
        target=target,
        ifc_type=ifc_type,
        category=category,
        limit=limit,
        seed=seed,
        top_k=top_k,
        output_root=s.data_root / "reports" / "predict-eval",
    )
    qdrant = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    builder = (
        build_kbims_predictor if target == "kbims_code" else build_pps_predictor
    )
    with contextlib.closing(qdrant), VLLMEmbedClient(
        url=s.embedding_url, model=s.embedding_model, dim=s.embedding_dim
    ) as embed, VLLMClient(
        url=s.llm_url, model=s.llm_model, timeout=s.llm_timeout_seconds
    ) as vllm:
        predictor = builder(
            settings=s,
            embed_client=embed,
            qdrant_client=qdrant,
            vllm_client=vllm,
        )
        try:
            metrics, run_dir = run_eval(
                cfg, predictor, qdrant, collection=s.collection_name
            )
        except NoSamplesError as exc:
            typer.echo(f"predict-eval: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    typer.echo(_format_summary(metrics, run_dir))


if __name__ == "__main__":
    app()
