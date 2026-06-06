from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient, VLLMEmbedError
from api.core.config import BIMSettings
from api.main import app


def _collection(name: str) -> MagicMock:
    c = MagicMock()
    c.name = name
    return c


def test_list_versions_endpoint() -> None:
    qdrant = MagicMock(spec=QdrantClient)
    qdrant.get_collections.return_value = MagicMock(
        collections=[_collection("bim__qwen4b_d2048"), _collection("logs")]
    )
    qdrant.count.return_value = MagicMock(count=1234)

    app.state.qdrant = qdrant
    app.state.bim = BIMSettings()

    response = TestClient(app).get("/versions")

    assert response.status_code == 200
    assert response.json() == {
        "versions": [{"name": "qwen4b_d2048", "points": 1234}]
    }


def _attr_payload() -> dict:
    return {
        "ifc_type": "IfcColumn",
        "category": "구조기둥",
        "family_name": "RC기둥",
        "family": "콘크리트-직사각형-기둥",
        "type": "400x600",
        "type_id": "1234",
        "kbims_code": "E275",
        "pps_code": "AMB",
    }


@pytest.fixture()
def create_client() -> TestClient:
    qdrant = MagicMock(spec=QdrantClient)
    qdrant.collection_exists.return_value = False  # new 미존재 (default)
    qdrant.count.return_value = MagicMock(count=1)
    qdrant.scroll.return_value = ([], None)  # base 복제 시 빈 소스 기본
    qdrant.get_collection.return_value = MagicMock(
        config=MagicMock(params=MagicMock(vectors=MagicMock(size=2048)))
    )

    embed = MagicMock(spec=VLLMEmbedClient)
    embed.embed.return_value = [[0.1] * 10]

    app.state.qdrant = qdrant
    app.state.embed = embed
    app.state.bim = BIMSettings()
    return TestClient(app)


def test_create_version_no_base_happy(create_client: TestClient) -> None:
    resp = create_client.post(
        "/versions", json={"name": "v_new", "items": [_attr_payload()]}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body == {"version": "v_new", "copied": 0, "added": 1, "total": 1}
    app.state.qdrant.create_collection.assert_called_once()
    app.state.qdrant.upsert.assert_called_once()


def test_create_version_with_base_copies(create_client: TestClient) -> None:
    # new=False, base=True
    app.state.qdrant.collection_exists.side_effect = (
        lambda name: name == "bim__base"
    )
    point = MagicMock(id="x", vector=[0.1] * 10, payload=_attr_payload())
    app.state.qdrant.scroll.side_effect = [([point], None)]
    try:
        resp = create_client.post(
            "/versions",
            json={"name": "v_new", "base": "base", "items": [_attr_payload()]},
        )
    finally:
        app.state.qdrant.collection_exists.side_effect = None
        app.state.qdrant.scroll.side_effect = None
    assert resp.status_code == 201
    body = resp.json()
    assert body["copied"] == 1
    assert body["added"] == 1


def test_create_version_name_collision_409(create_client: TestClient) -> None:
    app.state.qdrant.collection_exists.return_value = True
    try:
        resp = create_client.post(
            "/versions", json={"name": "dup", "items": [_attr_payload()]}
        )
    finally:
        app.state.qdrant.collection_exists.return_value = False
    assert resp.status_code == 409


def test_create_version_unknown_base_404(create_client: TestClient) -> None:
    app.state.qdrant.collection_exists.return_value = False  # new도 base도 없음
    resp = create_client.post(
        "/versions",
        json={"name": "v_new", "base": "ghost", "items": [_attr_payload()]},
    )
    assert resp.status_code == 404


def test_create_version_bad_name_422(create_client: TestClient) -> None:
    resp = create_client.post(
        "/versions", json={"name": "bad name!", "items": [_attr_payload()]}
    )
    assert resp.status_code == 422


def test_create_version_empty_items_no_base_422(create_client: TestClient) -> None:
    resp = create_client.post("/versions", json={"name": "v_new", "items": []})
    assert resp.status_code == 422


def test_create_version_dim_mismatch_422(create_client: TestClient) -> None:
    app.state.qdrant.collection_exists.side_effect = (
        lambda name: name == "bim__base"
    )
    app.state.qdrant.get_collection.return_value = MagicMock(
        config=MagicMock(params=MagicMock(vectors=MagicMock(size=999)))
    )
    try:
        resp = create_client.post(
            "/versions",
            json={"name": "v_new", "base": "base", "items": [_attr_payload()]},
        )
    finally:
        app.state.qdrant.collection_exists.side_effect = None
        app.state.qdrant.get_collection.return_value = MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=2048)))
        )
    assert resp.status_code == 422


def test_create_version_embed_down_503_and_cleanup(create_client: TestClient) -> None:
    app.state.embed.embed.side_effect = VLLMEmbedError("down")
    try:
        resp = create_client.post(
            "/versions", json={"name": "v_new", "items": [_attr_payload()]}
        )
    finally:
        app.state.embed.embed.side_effect = None
    assert resp.status_code == 503
    # 생성된 컬렉션 정리 시도
    app.state.qdrant.delete_collection.assert_called_once()


def test_create_version_vector_count_mismatch_503(create_client: TestClient) -> None:
    app.state.embed.embed.return_value = []  # 0 vectors for 1 item → ValueError
    try:
        resp = create_client.post(
            "/versions", json={"name": "v_new", "items": [_attr_payload()]}
        )
    finally:
        app.state.embed.embed.return_value = [[0.1] * 10]
    assert resp.status_code == 503
    app.state.qdrant.delete_collection.assert_called_once()


def test_create_version_init_failure_cleans_up(create_client: TestClient) -> None:
    app.state.qdrant.create_payload_index.side_effect = RuntimeError("boom")
    try:
        with pytest.raises(RuntimeError):
            create_client.post(
                "/versions", json={"name": "v_new", "items": [_attr_payload()]}
            )
    finally:
        app.state.qdrant.create_payload_index.side_effect = None
    app.state.qdrant.delete_collection.assert_called_once()
