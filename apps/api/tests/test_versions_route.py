from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

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
