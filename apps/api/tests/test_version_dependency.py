from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient
from api.core.config import BIMSettings
from api.main import app


def _client(collection_exists: bool = True) -> TestClient:
    count_result = MagicMock(count=0)
    qdrant = MagicMock(spec=QdrantClient)
    qdrant.count.return_value = count_result
    qdrant.scroll.return_value = ([], None)
    qdrant.collection_exists.return_value = collection_exists

    embed = MagicMock(spec=VLLMEmbedClient)

    app.state.qdrant = qdrant
    app.state.embed = embed
    app.state.bim = BIMSettings()
    return TestClient(app)


def test_omitted_version_uses_default_collection() -> None:
    client = _client()
    response = client.get("/bim-attributes")
    assert response.status_code == 200
    app.state.qdrant.count.assert_called_with(
        collection_name="bim__qwen4b_d2048", exact=True
    )


def test_explicit_version_targets_its_collection() -> None:
    client = _client()
    response = client.get("/bim-attributes?version=expA")
    assert response.status_code == 200
    app.state.qdrant.count.assert_called_with(
        collection_name="bim__expA", exact=True
    )


def test_unknown_version_returns_404() -> None:
    client = _client(collection_exists=False)
    response = client.get("/bim-attributes?version=ghost")
    assert response.status_code == 404
