import contextlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers.health import ServiceStatus


@pytest.fixture()
def client_with_state() -> TestClient:
    bim = MagicMock()
    bim.embedding_url = "http://embed-host"
    bim.llm_url = "http://llm-host"

    app.state.qdrant = MagicMock()
    app.state.bim = bim
    return TestClient(app)


def test_health_all_ok(client_with_state: TestClient) -> None:
    app.state.qdrant.get_collections.return_value = []

    _ok = ServiceStatus(status="ok")
    with patch("api.routers.health._probe_http", return_value=_ok):
        response = client_with_state.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"]["qdrant"]["status"] == "ok"


def test_health_qdrant_down(client_with_state: TestClient) -> None:
    app.state.qdrant.get_collections.side_effect = ConnectionError("refused")

    _ok = ServiceStatus(status="ok")
    with patch("api.routers.health._probe_http", return_value=_ok):
        response = client_with_state.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"]["qdrant"]["status"] == "error"


def test_health_embedding_down(client_with_state: TestClient) -> None:
    app.state.qdrant.get_collections.return_value = []

    def fake_probe(url: str) -> ServiceStatus:
        if "embed" in url:
            return ServiceStatus(status="error", detail="connection refused")
        return ServiceStatus(status="ok")

    with patch("api.routers.health._probe_http", side_effect=fake_probe):
        response = client_with_state.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"]["embedding"]["status"] == "error"


def test_health_not_initialized() -> None:
    client = TestClient(app)
    # Clear state so lifespan hasn't run
    for attr in ("qdrant", "bim"):
        with contextlib.suppress(AttributeError):
            delattr(app.state, attr)

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"]["api"]["status"] == "error"
