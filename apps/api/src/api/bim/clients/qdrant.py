"""Thin wrapper around qdrant-client for the BIM converter pipeline.

Responsibilities:
- ``ensure_collection``: create if missing, verify dim if exists.
- ``upsert_batch``: upsert a batch of BIMAttributes with vectors and
  payload (idempotent by ``stable_id``).

Payload indexes on ``ifc_type``, ``category``, ``kbims_code``, ``family``
are created at collection creation time (spec §5-4).
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from api.bim.schemas import BIMAttribute

logger = logging.getLogger(__name__)

_INDEXED_PAYLOAD_FIELDS: tuple[str, ...] = (
    "ifc_type",
    "category",
    "kbims_code",
    "family",
)


class DimensionMismatchError(RuntimeError):
    """Raised when an existing collection's vector size disagrees with config."""


class QdrantWrapper:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    @classmethod
    def from_settings(
        cls, *, url: str, api_key: str | None = None
    ) -> QdrantWrapper:
        return cls(QdrantClient(url=url, api_key=api_key))

    def ensure_collection(self, name: str, *, dim: int) -> None:
        """Create collection if missing; if present, verify dim matches."""
        info = self._get_collection_or_none(name)
        if info is not None:
            existing_dim = info.config.params.vectors.size
            if existing_dim != dim:
                raise DimensionMismatchError(
                    f"Collection '{name}' has dim={existing_dim}, "
                    f"configured dim={dim}. Pick a different experiment_id "
                    f"or recreate the collection."
                )
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        for field in _INDEXED_PAYLOAD_FIELDS:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d, indexes=%s)",
            name,
            dim,
            list(_INDEXED_PAYLOAD_FIELDS),
        )

    def upsert_batch(
        self,
        collection: str,
        attributes: list[BIMAttribute],
        vectors: list[list[float]],
        *,
        source_file: str = "",
        ingested_at: str = "",
    ) -> int:
        if len(attributes) != len(vectors):
            raise ValueError(
                f"Length mismatch: {len(attributes)} attributes vs "
                f"{len(vectors)} vectors"
            )
        if not attributes:
            return 0

        points = [
            PointStruct(
                id=attr.stable_id,
                vector=vec,
                payload={
                    **attr.model_dump(),
                    "source_file": source_file,
                    "ingested_at": ingested_at,
                },
            )
            for attr, vec in zip(attributes, vectors, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points, wait=True)
        logger.info("Upserted %d points into '%s'", len(points), collection)
        return len(points)

    def count(self, collection: str) -> int:
        return self._client.get_collection(collection).points_count or 0

    def _get_collection_or_none(self, name: str):
        try:
            return self._client.get_collection(name)
        except (UnexpectedResponse, ValueError):
            return None
