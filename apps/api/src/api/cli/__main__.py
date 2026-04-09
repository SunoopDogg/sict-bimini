from __future__ import annotations

import logging
from pathlib import Path

import typer

from api.bim.clients.qdrant import QdrantWrapper
from api.bim.clients.tei import TEIClient
from api.bim.pipeline import run_ingest_xlsx, run_normalize, run_upsert_qdrant
from api.core.config import BIMSettings

app = typer.Typer(add_completion=False, no_args_is_help=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _settings_from_args(
    experiment_id: str | None,
    dim: int | None,
    tei_url: str | None,
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
            "tei_url": tei_url,
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
_TeiUrl = typer.Option(None, help="Override BIM_TEI_URL.")
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
    tei_url: str | None = _TeiUrl,
    qdrant_url: str | None = _QdrantUrl,
    model: str | None = _Model,
) -> None:
    """Stage 3: normalized JSON → TEI embeddings → Qdrant upsert."""
    s = _settings_from_args(experiment_id, dim, tei_url, qdrant_url, model, data_root)
    with TEIClient(url=s.tei_url, model=s.embedding_model, dim=s.embedding_dim) as tei:
        qw = QdrantWrapper.from_settings(url=s.qdrant_url, api_key=s.qdrant_api_key)
        run_upsert_qdrant(
            data_root=s.data_root,
            tei_client=tei,
            qdrant=qw,
            collection=s.collection_name,
            dim=s.embedding_dim,
        )


@app.command("pipeline")
def pipeline_cmd(
    data_root: Path | None = _DataRoot,
    experiment_id: str | None = _ExpId,
    dim: int | None = _Dim,
    tei_url: str | None = _TeiUrl,
    qdrant_url: str | None = _QdrantUrl,
    model: str | None = _Model,
) -> None:
    """Run all 3 stages sequentially."""
    s = _settings_from_args(experiment_id, dim, tei_url, qdrant_url, model, data_root)
    run_ingest_xlsx(s.data_root)
    run_normalize(s.data_root)
    with TEIClient(url=s.tei_url, model=s.embedding_model, dim=s.embedding_dim) as tei:
        qw = QdrantWrapper.from_settings(url=s.qdrant_url, api_key=s.qdrant_api_key)
        run_upsert_qdrant(
            data_root=s.data_root,
            tei_client=tei,
            qdrant=qw,
            collection=s.collection_name,
            dim=s.embedding_dim,
        )


if __name__ == "__main__":
    app()
