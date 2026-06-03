from fastapi.testclient import TestClient

from api.core.config import BIMSettings
from api.main import app


def test_meta_returns_model_names() -> None:
    app.state.bim = BIMSettings(
        llm_model="gemma-4", embedding_model="qwen3-embedding-4b"
    )

    response = TestClient(app).get("/meta")

    assert response.status_code == 200
    assert response.json() == {
        "llm_model": "gemma-4",
        "embedding_model": "qwen3-embedding-4b",
    }
