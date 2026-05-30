from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient, VLLMEmbedError
from api.core.config import BIMSettings
from api.main import app


def _make_qdrant_point() -> MagicMock:
    point = MagicMock()
    point.payload = {
        "ifc_type": "IfcColumn",
        "category": "구조기둥",
        "family_name": "RC기둥",
        "family": "콘크리트-직사각형-기둥",
        "type": "400x600",
        "type_id": "1234",
        "kbims_code": "E275",
        "pps_code": "AMB",
    }
    return point


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
def client() -> TestClient:
    count_result = MagicMock()
    count_result.count = 1

    mock_qdrant = MagicMock(spec=QdrantClient)
    mock_qdrant.count.return_value = count_result
    mock_qdrant.scroll.return_value = ([_make_qdrant_point()], None)

    mock_embed = MagicMock(spec=VLLMEmbedClient)
    mock_embed.embed.return_value = [[0.1] * 10]

    app.state.qdrant = mock_qdrant
    app.state.embed = mock_embed
    app.state.bim = BIMSettings()

    return TestClient(app)


def test_list_bim_attributes_happy_path(client: TestClient) -> None:
    response = client.get("/bim-attributes")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["page"] == 1
    assert body["page_size"] == 20
    assert body["total_pages"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["ifc_type"] == "IfcColumn"


def test_list_bim_attributes_empty_collection(client: TestClient) -> None:
    app.state.qdrant.count.return_value.count = 0
    app.state.qdrant.scroll.return_value = ([], None)
    try:
        response = client.get("/bim-attributes")
    finally:
        app.state.qdrant.count.return_value.count = 1
        app.state.qdrant.scroll.return_value = ([_make_qdrant_point()], None)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []
    assert body["total_pages"] == 0


def test_create_bim_attributes_happy_path(client: TestClient) -> None:
    payload = {"items": [_attr_payload()]}
    response = client.post("/bim-attributes", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["added"] == 1
    assert "total" in body
    app.state.qdrant.upsert.assert_called_once()


def test_create_bim_attributes_embed_failure_returns_503(client: TestClient) -> None:
    app.state.embed.embed.side_effect = VLLMEmbedError("down")
    try:
        response = client.post("/bim-attributes", json={"items": [_attr_payload()]})
    finally:
        app.state.embed.embed.side_effect = None
    assert response.status_code == 503


def test_list_bim_attributes_pagination_params(client: TestClient) -> None:
    response = client.get("/bim-attributes?page=1&page_size=5")
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 5
