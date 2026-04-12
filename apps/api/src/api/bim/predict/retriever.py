"""Qdrant-backed neighbor retrieval for code prediction.

Uses ``MatchExcept(except_=[""])`` so only points with the target code
field populated come back. The indexed payload fields are kbims_code /
pps_code (both KEYWORD — see apps/api/src/api/bim/clients/qdrant.py).
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchExcept, ScoredPoint

from api.bim.predict.schemas import Neighbor, TargetCode

_WITH_PAYLOAD = [
    "stable_id",
    "kbims_code",
    "pps_code",
    "ifc_type",
    "category",
]


class NeighborRetriever:
    def __init__(self, client: QdrantClient, *, collection: str) -> None:
        self._client = client
        self._collection = collection

    def search(
        self,
        query_vector: list[float],
        *,
        code_field: TargetCode,
        k: int,
        extra_filter: Filter | None = None,
    ) -> list[Neighbor]:
        must: list = [
            FieldCondition(
                key=code_field,
                match=MatchExcept(**{"except": [""]}),
            )
        ]
        if extra_filter is not None:
            must.append(extra_filter)
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=Filter(must=must),
            limit=k,
            with_payload=_WITH_PAYLOAD,
        )
        return [_point_to_neighbor(h) for h in response.points]


def _point_to_neighbor(point: ScoredPoint) -> Neighbor:
    payload = point.payload or {}
    return Neighbor(
        stable_id=payload.get("stable_id", ""),
        score=point.score,
        kbims_code=payload.get("kbims_code", "") or "",
        pps_code=payload.get("pps_code", "") or "",
        ifc_type=payload.get("ifc_type", ""),
        category=payload.get("category", ""),
    )
