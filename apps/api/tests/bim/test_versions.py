from unittest.mock import MagicMock

from qdrant_client import QdrantClient

from api.bim.versions import (
    VersionInfo,
    VersionService,
    collection_for_version,
    version_from_collection,
)


def test_collection_for_version() -> None:
    assert collection_for_version("qwen4b_d2048") == "bim__qwen4b_d2048"


def test_version_from_collection_strips_prefix() -> None:
    assert version_from_collection("bim__qwen4b_d2048") == "qwen4b_d2048"


def test_version_from_collection_returns_none_for_non_bim() -> None:
    assert version_from_collection("other_collection") is None


def _collection(name: str) -> MagicMock:
    c = MagicMock()
    c.name = name  # NOT MagicMock(name=...) — that sets repr name
    return c


def test_list_versions_filters_non_bim_counts_and_sorts() -> None:
    qdrant = MagicMock(spec=QdrantClient)
    qdrant.get_collections.return_value = MagicMock(
        collections=[
            _collection("bim__v2"),
            _collection("other"),
            _collection("bim__v1"),
        ]
    )
    qdrant.count.return_value = MagicMock(count=42)

    versions = VersionService(qdrant).list_versions()

    assert versions == [
        VersionInfo(name="v1", points=42),
        VersionInfo(name="v2", points=42),
    ]
    qdrant.count.assert_any_call(collection_name="bim__v1", exact=True)
