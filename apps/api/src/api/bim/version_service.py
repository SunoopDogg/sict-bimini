"""Orchestrates creating a new DB version (Qdrant collection).

Pure data layer (no FastAPI). Caller (the route) does HTTP validation —
name format, 409 collision precheck, base existence, dim resolution — then
hands resolved collection names + dim here. On any failure after the
collection is created, best-effort deletes the new collection and re-raises;
the base collection is never touched.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from api.bim.attribute_service import BIMAttributeService
from api.bim.clients.embeddings_vllm import VLLMEmbedClient
from api.bim.clients.qdrant import QdrantWrapper
from api.bim.schemas import BIMAttribute

logger = logging.getLogger(__name__)


def create_version(
    qdrant: QdrantClient,
    embed: VLLMEmbedClient,
    *,
    new_collection: str,
    base_collection: str | None,
    dim: int,
    items: list[BIMAttribute],
) -> tuple[int, int, int]:
    """Create new_collection, optionally clone base, upsert embedded items.

    Returns (copied, added, total). Raises VLLMEmbedError/ValueError on
    embedding failure (route maps to 503).
    """
    wrapper = QdrantWrapper(qdrant)
    wrapper.init_collection(new_collection, dim=dim)
    try:
        copied = (
            wrapper.copy_collection(base_collection, new_collection)
            if base_collection is not None
            else 0
        )

        deduped = BIMAttributeService.dedup(items)
        added = 0
        if deduped:
            vectors = embed.embed([attr.embed_text() for attr in deduped])
            if len(vectors) != len(deduped):
                raise ValueError(
                    f"Embedding service returned {len(vectors)} vectors"
                    f" for {len(deduped)} inputs"
                )
            ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
            wrapper.upsert_batch(
                new_collection,
                deduped,
                vectors,
                source_file="version-create",
                ingested_at=ingested_at,
            )
            added = len(deduped)

        total = qdrant.count(collection_name=new_collection, exact=True).count
        return copied, added, total
    except Exception:
        try:
            wrapper.delete_collection(new_collection)
        except Exception as cleanup_err:  # noqa: BLE001
            logger.warning(
                "Failed to clean up partial collection '%s': %s",
                new_collection,
                cleanup_err,
            )
        raise
