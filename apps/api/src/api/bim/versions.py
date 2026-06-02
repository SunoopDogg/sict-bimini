"""Vector-DB version discovery — maps Qdrant collections to selectable versions.

A version is an ``experiment_id``; its collection is ``bim__{version}``.
Decoupled from HTTP so it is unit-testable on a raw ``QdrantClient``.
"""

from __future__ import annotations

from pydantic import BaseModel
from qdrant_client import QdrantClient

COLLECTION_PREFIX = "bim__"


def collection_for_version(version: str) -> str:
    return f"{COLLECTION_PREFIX}{version}"


def version_from_collection(name: str) -> str | None:
    if not name.startswith(COLLECTION_PREFIX):
        return None
    return name[len(COLLECTION_PREFIX) :]


class DbVersion(BaseModel):
    name: str
    points: int


class VersionListResponse(BaseModel):
    versions: list[DbVersion]


def list_versions(qdrant: QdrantClient) -> list[DbVersion]:
    versions: list[DbVersion] = []
    for collection in qdrant.get_collections().collections:
        version = version_from_collection(collection.name)
        if version is None:
            continue
        # Approximate is fine — this only feeds a "N개" dropdown label.
        count = qdrant.count(collection_name=collection.name, exact=False).count
        versions.append(DbVersion(name=version, points=count))
    return sorted(versions, key=lambda v: v.name)
