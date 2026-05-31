from unittest.mock import MagicMock

from qdrant_client import QdrantClient

from api.bim.attribute_service import BIMAttributeService
from api.bim.schemas import BIMAttribute


def _attr(type_id: str, kbims: str = "E1") -> BIMAttribute:
    return BIMAttribute(
        ifc_type="IfcColumn",
        category="구조기둥",
        family_name="RC기둥",
        family="콘크리트-직사각형-기둥",
        type="400x600",
        type_id=type_id,
        kbims_code=kbims,
    )


def _service(qdrant: MagicMock) -> BIMAttributeService:
    return BIMAttributeService(qdrant, "bim__test")


def test_dedup_keeps_last_per_stable_id() -> None:
    first = _attr("1", kbims="E1")
    second = _attr("1", kbims="E2")  # same identity → same stable_id
    other = _attr("2")

    deduped = BIMAttributeService.dedup([first, second, other])

    assert len(deduped) == 2
    by_id = {a.stable_id: a for a in deduped}
    assert by_id[first.stable_id].kbims_code == "E2"  # last wins


def test_upsert_batch_builds_points_and_calls_qdrant() -> None:
    qdrant = MagicMock(spec=QdrantClient)
    attrs = [_attr("1"), _attr("2")]
    vectors = [[0.1] * 4, [0.2] * 4]

    _service(qdrant).upsert_batch(attrs, vectors)

    qdrant.upsert.assert_called_once()
    _, kwargs = qdrant.upsert.call_args
    points = kwargs["points"]
    assert kwargs["collection_name"] == "bim__test"
    assert [p.id for p in points] == [a.stable_id for a in attrs]
    assert points[0].payload["source_file"] == ""


def test_get_page_first_page_returns_items_and_totals() -> None:
    qdrant = MagicMock(spec=QdrantClient)
    qdrant.count.return_value = MagicMock(count=1)
    point = MagicMock()
    point.payload = _attr("1").model_dump()
    qdrant.scroll.return_value = ([point], None)

    items, total, total_pages = _service(qdrant).get_page(page=1, page_size=20)

    assert total == 1
    assert total_pages == 1
    assert len(items) == 1
    assert items[0].type_id == "1"


def test_get_page_empty_collection() -> None:
    qdrant = MagicMock(spec=QdrantClient)
    qdrant.count.return_value = MagicMock(count=0)
    qdrant.scroll.return_value = ([], None)

    items, total, total_pages = _service(qdrant).get_page(page=1, page_size=20)

    assert items == []
    assert total == 0
    assert total_pages == 0
