from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.bim.predict import (
    EmptyRetrievalError,
    LLMGenerationError,
    PredictionCandidate,
    PredictionMode,
    PredictionResponse,
    Predictor,
)
from api.main import app


def _make_response(target: str) -> PredictionResponse:
    return PredictionResponse(
        target=target,
        mode=PredictionMode.STRONG,
        candidates=[
            PredictionCandidate(
                code="E275" if target == "kbims_code" else "AMB",
                llm_confidence=0.9,
                retrieval_score=0.95,
                source="neighbor",
            )
        ],
        low_confidence_context=False,
        pool_size=10,
        retrieved_k=10,
    )


def _predict_body() -> dict:
    return {
        "attribute": {
            "ifc_type": "IfcColumn",
            "category": "구조기둥",
            "family_name": "RC기둥",
            "family": "콘크리트-직사각형-기둥",
            "type": "400x600",
            "type_id": "1234",
            "kbims_code": "",
            "pps_code": "",
        },
        "n": 5,
    }


@pytest.fixture()
def client() -> TestClient:
    mock_kbims = MagicMock(spec=Predictor)
    mock_kbims.predict.return_value = _make_response("kbims_code")

    mock_pps = MagicMock(spec=Predictor)
    mock_pps.predict.return_value = _make_response("pps_code")

    app.state.kbims = mock_kbims
    app.state.pps = mock_pps

    return TestClient(app)


def test_predict_happy_path(client: TestClient) -> None:
    response = client.post("/predict", json=_predict_body())
    assert response.status_code == 200
    body = response.json()
    assert "kbims" in body
    assert "pps" in body
    assert body["kbims"]["target"] == "kbims_code"
    assert body["pps"]["target"] == "pps_code"
    assert body["kbims"]["candidates"][0]["code"] == "E275"


def test_predict_empty_retrieval_returns_422(client: TestClient) -> None:
    app.state.kbims.predict.side_effect = EmptyRetrievalError("empty")
    try:
        response = client.post("/predict", json=_predict_body())
    finally:
        app.state.kbims.predict.side_effect = None
    assert response.status_code == 422


def test_predict_llm_error_returns_503(client: TestClient) -> None:
    app.state.kbims.predict.side_effect = LLMGenerationError("LLM down")
    try:
        response = client.post("/predict", json=_predict_body())
    finally:
        app.state.kbims.predict.side_effect = None
    assert response.status_code == 503


def test_batch_predict_happy_path(client: TestClient) -> None:
    payload = {
        "objects": [_predict_body()["attribute"]],
        "n": 5,
    }
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["successful"] == 1
    assert body["failed"] == 0
    assert body["results"][0]["prediction"] is not None
    assert body["results"][0]["error"] is None


def test_batch_predict_partial_failure(client: TestClient) -> None:
    app.state.kbims.predict.side_effect = [
        _make_response("kbims_code"),
        LLMGenerationError("LLM down"),
    ]
    app.state.pps.predict.side_effect = [
        _make_response("pps_code"),
    ]
    payload = {
        "objects": [_predict_body()["attribute"], _predict_body()["attribute"]],
        "n": 5,
    }
    try:
        response = client.post("/batch-predict", json=payload)
    finally:
        app.state.kbims.predict.side_effect = None
        app.state.pps.predict.side_effect = None
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["successful"] == 1
    assert body["failed"] == 1
    assert body["results"][1]["error"] is not None
