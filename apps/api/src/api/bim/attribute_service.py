"""Business logic for the BIM-attributes API, decoupled from the HTTP layer.

Operates on a raw ``QdrantClient`` so it can be unit-tested without FastAPI.
Embedding and HTTP error mapping stay in the router; this service only does
the data work (dedup, point construction, upsert, paginated reads).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from api.bim.schemas import BIMAttribute, bim_attr_from_payload

# Cap per scroll request while skipping ahead to the requested page.
_SCROLL_SKIP_BATCH = 250


class BIMAttributeService:
    def __init__(self, qdrant: QdrantClient, collection: str) -> None:
        self._qdrant = qdrant
        self._collection = collection

    @staticmethod
    def dedup(items: list[BIMAttribute]) -> list[BIMAttribute]:
        """Collapse by stable_id (last wins) — matches normalizer semantics."""
        return list({attr.stable_id: attr for attr in items}.values())

    def count(self) -> int:
        result = self._qdrant.count(collection_name=self._collection, exact=True)
        return result.count

    def upsert_batch(
        self, deduped: list[BIMAttribute], vectors: list[list[float]]
    ) -> None:
        ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
        points = [
            PointStruct(
                id=attr.stable_id,
                vector=vec,
                payload={
                    **attr.model_dump(),
                    "stable_id": attr.stable_id,
                    "source_file": "",
                    "ingested_at": ingested_at,
                },
            )
            for attr, vec in zip(deduped, vectors, strict=True)
        ]
        self._qdrant.upsert(
            collection_name=self._collection, points=points, wait=True
        )

    def get_page(
        self, page: int, page_size: int
    ) -> tuple[list[BIMAttribute], int, int]:
        """Return (items, total, total_pages) for a 1-based page.

        Skips ``(page-1)*page_size`` records via scroll without loading their
        payloads, then fetches the requested page.
        """
        total = self.count()
        total_pages = math.ceil(total / page_size) if total > 0 else 0

        skip = (page - 1) * page_size
        offset = None
        while skip > 0:
            batch_size = min(skip, _SCROLL_SKIP_BATCH)
            points, offset = self._qdrant.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            skip -= len(points)
            if not points or offset is None:
                return [], total, total_pages

        points, _ = self._qdrant.scroll(
            collection_name=self._collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        items = [
            attr
            for point in points
            if (attr := bim_attr_from_payload(point.payload or {})) is not None
        ]
        return items, total, total_pages
