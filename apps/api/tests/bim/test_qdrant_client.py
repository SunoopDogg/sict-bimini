import pytest
from qdrant_client import QdrantClient

from api.bim.clients.qdrant import (
    DimensionMismatchError,
    QdrantWrapper,
)
from api.bim.schemas import BIMAttribute


def _sample_attr(**overrides) -> BIMAttribute:
    base = dict(
        ifc_type="IfcColumn",
        category="건축",
        family_name="RC기둥",
        family="기둥",
        type="T1",
        type_id="X",
        kbims_code="AR-C-001",
        pps_code="",
    )
    base.update(overrides)
    return BIMAttribute(**base)


@pytest.fixture
def in_memory_client():
    return QdrantClient(":memory:")


@pytest.fixture
def wrapper(in_memory_client):
    return QdrantWrapper(client=in_memory_client)


class TestQdrantWrapper:
    def test_ensure_collection_creates_new(self, wrapper, in_memory_client):
        wrapper.ensure_collection("bim__test", dim=4)
        collections = [c.name for c in in_memory_client.get_collections().collections]
        assert "bim__test" in collections

    def test_ensure_collection_is_idempotent(self, wrapper):
        wrapper.ensure_collection("bim__test", dim=4)
        wrapper.ensure_collection("bim__test", dim=4)  # no error on 2nd call

    def test_ensure_collection_raises_on_dim_mismatch(self, wrapper):
        wrapper.ensure_collection("bim__test", dim=4)
        with pytest.raises(DimensionMismatchError):
            wrapper.ensure_collection("bim__test", dim=8)

    def test_upsert_batch_adds_points(self, wrapper, in_memory_client):
        wrapper.ensure_collection("bim__test", dim=4)
        attrs = [_sample_attr(type_id=f"T{i}") for i in range(3)]
        vectors = [[1.0, 0.0, 0.0, 0.0]] * 3

        count = wrapper.upsert_batch(
            collection="bim__test",
            attributes=attrs,
            vectors=vectors,
        )
        assert count == 3

        info = in_memory_client.get_collection("bim__test")
        assert info.points_count == 3

    def test_upsert_batch_idempotent_by_stable_id(self, wrapper, in_memory_client):
        """동일 identity → 동일 stable_id → 재삽입 시 덮어씀, 카운트 불변."""
        wrapper.ensure_collection("bim__test", dim=4)
        attr = _sample_attr()
        wrapper.upsert_batch("bim__test", [attr], [[1.0, 0.0, 0.0, 0.0]])
        wrapper.upsert_batch("bim__test", [attr], [[0.0, 1.0, 0.0, 0.0]])

        assert in_memory_client.get_collection("bim__test").points_count == 1

    def test_upsert_batch_length_mismatch_raises(self, wrapper):
        wrapper.ensure_collection("bim__test", dim=4)
        attrs = [_sample_attr()]
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        with pytest.raises(ValueError):
            wrapper.upsert_batch("bim__test", attrs, vectors)

    def test_payload_contains_all_bim_fields_and_metadata(
        self, wrapper, in_memory_client
    ):
        wrapper.ensure_collection("bim__test", dim=4)
        attr = _sample_attr()
        wrapper.upsert_batch(
            "bim__test",
            [attr],
            [[1.0, 0.0, 0.0, 0.0]],
            source_file="속성테이블(10층).xlsx",
            ingested_at="2026-04-16T10:30:00Z",
        )
        points = in_memory_client.retrieve(
            collection_name="bim__test",
            ids=[attr.stable_id],
            with_payload=True,
            with_vectors=False,
        )
        assert len(points) == 1
        payload = points[0].payload
        for field in (
            "ifc_type", "category", "family_name", "family", "type", "type_id",
            "kbims_code", "pps_code", "stable_id", "source_file", "ingested_at",
        ):
            assert field in payload
        assert payload["stable_id"] == attr.stable_id
        assert payload["source_file"] == "속성테이블(10층).xlsx"
        assert payload["ingested_at"] == "2026-04-16T10:30:00Z"

    def test_payload_indexes_are_created(self, wrapper, in_memory_client):
        """인덱스 생성 호출 자체는 in-memory qdrant-client에서 에러 없이 통과하면 OK."""
        wrapper.ensure_collection("bim__test", dim=4)
        info = in_memory_client.get_collection("bim__test")
        assert info is not None
