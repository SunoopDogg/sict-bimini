from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient, VLLMEmbedError
from api.core.config import BIMSettings
from api.main import app


def _make_qdrant_point(ifc_type: str = "IfcColumn", score: float = 0.95) -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = {
        "ifc_type": ifc_type,
        "category": "구조기둥",
        "family_name": "RC기둥",
        "family": "콘크리트-직사각형-기둥",
        "type": "400x600",
        "type_id": "1234",
        "kbims_code": "E275",
        "pps_code": "AMB",
    }
    return point


@pytest.fixture()
def client() -> TestClient:
    mock_embed = MagicMock(spec=VLLMEmbedClient)
    mock_embed.embed.return_value = [[0.1] * 10]

    mock_result = MagicMock()
    mock_result.points = [_make_qdrant_point()]

    mock_qdrant = MagicMock(spec=QdrantClient)
    mock_qdrant.query_points.return_value = mock_result

    app.state.embed = mock_embed
    app.state.qdrant = mock_qdrant
    app.state.bim = BIMSettings()

    return TestClient(app)


def test_search_happy_path(client: TestClient) -> None:
    response = client.get("/search?query=기둥&top_k=1")
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["score"] == 0.95
    assert body["results"][0]["attribute"]["ifc_type"] == "IfcColumn"


def test_search_embed_failure_returns_503(client: TestClient) -> None:
    app.state.embed.embed.side_effect = VLLMEmbedError("down")
    try:
        response = client.get("/search?query=기둥")
    finally:
        app.state.embed.embed.side_effect = None
    assert response.status_code == 503


def test_search_empty_query_returns_422(client: TestClient) -> None:
    response = client.get("/search?query=")
    assert response.status_code == 422


def test_search_missing_query_returns_422(client: TestClient) -> None:
    response = client.get("/search")
    assert response.status_code == 422
