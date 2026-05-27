import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.bim.schemas import BIMObjectRaw
from api.bim.xlsx_parser import MissingColumnsError
from api.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _raw_object() -> BIMObjectRaw:
    return BIMObjectRaw(
        source_file="test.xlsx",
        object_name="기둥",
        ifc_type="IfcColumn",
        global_id="abc123",
        properties={},
    )


def test_convert_xlsx_happy_path(client: TestClient) -> None:
    with patch("api.routers.conversion.parse_xlsx_to_raw", return_value=[_raw_object()]):
        response = client.post(
            "/convert/xlsx-to-json",
            files={"file": ("test.xlsx", b"fake", "application/octet-stream")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["total_objects"] == 1
    assert body["source_filename"] == "test.xlsx"
    assert body["objects"][0]["source_file"] == "test.xlsx"


def test_convert_rejects_invalid_extension(client: TestClient) -> None:
    response = client.post(
        "/convert/xlsx-to-json",
        files={"file": ("report.pdf", b"fake", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_convert_rejects_xls_extension(client: TestClient) -> None:
    response = client.post(
        "/convert/xlsx-to-json",
        files={"file": ("report.xls", b"fake", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_convert_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/convert/xlsx-to-json",
        files={"file": ("test.xlsx", b"", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_convert_missing_columns_returns_422(client: TestClient) -> None:
    with patch(
        "api.routers.conversion.parse_xlsx_to_raw",
        side_effect=MissingColumnsError("Missing required columns: ['객체명']"),
    ):
        response = client.post(
            "/convert/xlsx-to-json",
            files={"file": ("test.xlsx", b"fake", "application/octet-stream")},
        )
    assert response.status_code == 422
