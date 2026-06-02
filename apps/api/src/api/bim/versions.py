"""Vector-DB version discovery — maps Qdrant collections to selectable versions.

A version is an ``experiment_id``; its collection is ``bim__{version}``.
Decoupled from HTTP so it is unit-testable on a raw ``QdrantClient``.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

COLLECTION_PREFIX = "bim__"


def collection_for_version(version: str) -> str:
    return f"{COLLECTION_PREFIX}{version}"


def version_from_collection(name: str) -> str | None:
    if not name.startswith(COLLECTION_PREFIX):
        return None
    return name[len(COLLECTION_PREFIX) :]


@dataclass(frozen=True)
class VersionInfo:
    name: str
    points: int


class VersionService:
    def __init__(self, qdrant: QdrantClient) -> None:
        self._qdrant = qdrant

    def list_versions(self) -> list[VersionInfo]:
        collections = self._qdrant.get_collections().collections
        versions: list[VersionInfo] = []
        for collection in collections:
            version = version_from_collection(collection.name)
            if version is None:
                continue
            # Approximate is fine — this only feeds a "N개" dropdown label.
            count = self._qdrant.count(
                collection_name=collection.name, exact=False
            ).count
            versions.append(VersionInfo(name=version, points=count))
        return sorted(versions, key=lambda v: v.name)
