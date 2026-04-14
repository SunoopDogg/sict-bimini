"""3-stage BIM converter pipeline orchestrator (pure domain, no HTTP/CLI).

Stages:
1. ``run_ingest_xlsx``: xlsx → ``data_root/json/raw/<stem>.json``
2. ``run_normalize``:   raw JSON → ``data_root/json/normalized/<stem>.json``
3. ``run_upsert_qdrant``: normalized JSON → Qdrant collection (embed via vLLM)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from api.bim.clients.embeddings_vllm import VLLMEmbedClient
from api.bim.clients.qdrant import QdrantWrapper
from api.bim.normalizer import normalize_raw_objects
from api.bim.schemas import BIMAttribute, BIMObjectRaw
from api.bim.xlsx_parser import parse_xlsx_to_raw

logger = logging.getLogger(__name__)

_RawList = TypeAdapter(list[BIMObjectRaw])
_AttrList = TypeAdapter(list[BIMAttribute])


def run_ingest_xlsx(data_root: Path) -> None:
    """Stage 1: xlsx → data_root/json/raw/<stem>.json (one JSON per source)."""
    xlsx_dir = data_root / "xlsx"
    out_dir = data_root / "json" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(xlsx_dir.glob("*.xlsx"))
    if not files:
        logger.info("No xlsx files under %s", xlsx_dir)
        return

    for path in files:
        try:
            objects = parse_xlsx_to_raw(path)
        except Exception:
            logger.exception("Failed to parse %s; skipping", path.name)
            continue
        out = out_dir / f"{path.stem}.json"
        out.write_text(
            _RawList.dump_json(objects, indent=2).decode("utf-8"),
            encoding="utf-8",
        )
        logger.info("Stage 1: %s → %s (%d objects)", path.name, out.name, len(objects))


def run_normalize(data_root: Path) -> None:
    """Stage 2: raw JSON → data_root/json/normalized/<stem>.json (per source)."""
    raw_dir = data_root / "json" / "raw"
    out_dir = data_root / "json" / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(raw_dir.glob("*.json")):
        try:
            raws = _RawList.validate_json(path.read_bytes())
        except Exception:
            logger.exception("Failed to read raw JSON %s; skipping", path.name)
            continue

        attrs = normalize_raw_objects(raws)
        out = out_dir / path.name
        out.write_text(
            _AttrList.dump_json(attrs, indent=2).decode("utf-8"),
            encoding="utf-8",
        )
        logger.info(
            "Stage 2: %s → %s (%d → %d)",
            path.name, out.name, len(raws), len(attrs),
        )


def run_upsert_qdrant(
    data_root: Path,
    embed_client: VLLMEmbedClient,
    qdrant: QdrantWrapper,
    *,
    collection: str,
    dim: int,
    batch_size: int = 32,
) -> int:
    """Stage 3: normalized JSON → Qdrant collection (embed + upsert).

    Returns the number of points upserted across all source files.
    """
    qdrant.ensure_collection(collection, dim=dim)
    normalized_dir = data_root / "json" / "normalized"
    total = 0
    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")

    # Stage 3 is intentionally fail-fast (no per-file try/except): embeddings/Qdrant
    # errors typically indicate systemic issues (auth, dim mismatch, bad
    # collection state) where continuing masks real problems. Per-file
    # isolation in stages 1-2 is safe because parse errors are local to a
    # file; stage 3 errors are not.
    for path in sorted(normalized_dir.glob("*.json")):
        attrs = _AttrList.validate_json(path.read_bytes())
        if not attrs:
            continue

        for chunk_start in range(0, len(attrs), batch_size):
            chunk = attrs[chunk_start : chunk_start + batch_size]
            vectors = embed_client.embed([a.embed_text() for a in chunk])
            total += qdrant.upsert_batch(
                collection=collection,
                attributes=chunk,
                vectors=vectors,
                source_file=path.stem + ".xlsx",
                ingested_at=ingested_at,
            )

    logger.info("Stage 3 complete: %d points upserted into %s", total, collection)
    return total
