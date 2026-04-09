import json
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from api.bim.clients.qdrant import QdrantWrapper
from api.bim.clients.tei import TEIClient
from api.bim.pipeline import run_ingest_xlsx, run_normalize, run_upsert_qdrant


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    for sub in ("xlsx", "json/raw", "json/normalized"):
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_xlsx(data_root: Path) -> Path:
    """Build a minimal-valid BIM xlsx directly in ``data_root/xlsx/``.

    (We build here instead of reusing ``make_xlsx`` because the pipeline
    discovers inputs by scanning ``data_root/xlsx``.)
    """
    from openpyxl import Workbook
    rows = [
        (None, None, None, None),
        ("객체유형:Column", None, None, None),
        ("GlobalID:id1", None, None, None),
        ("기둥-1", "Other", "Category", "건축"),
        ("기둥-1", "Other", "Family Name", "RC기둥"),
        ("기둥-1", "Other", "Family", "기둥"),
        ("기둥-1", "Other", "Type", "T1"),
        ("기둥-1", "Other", "Type Id", "X1"),
        ("기둥-1", "Other", "KBIMS-부위코드", "AR-C-001"),
    ]
    wb = Workbook()
    ws = wb.active
    ws.append(["객체명", "속성세트", "속성명", "속성값"])
    for row in rows:
        ws.append(list(row))
    path = data_root / "xlsx" / "fixture.xlsx"
    wb.save(path)
    return path


class TestRunIngestXlsx:
    def test_produces_raw_json_per_source(self, data_root, sample_xlsx):
        run_ingest_xlsx(data_root)
        out = data_root / "json" / "raw" / "fixture.json"
        assert out.exists()
        payload = json.loads(out.read_text("utf-8"))
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["ifc_type"] == "IfcColumn"

    def test_no_xlsx_produces_nothing(self, data_root):
        run_ingest_xlsx(data_root)
        raw_dir = data_root / "json" / "raw"
        assert list(raw_dir.iterdir()) == []


class TestRunNormalize:
    def test_produces_normalized_json(self, data_root, sample_xlsx):
        run_ingest_xlsx(data_root)
        run_normalize(data_root)
        out = data_root / "json" / "normalized" / "fixture.json"
        assert out.exists()
        payload = json.loads(out.read_text("utf-8"))
        assert len(payload) == 1
        assert payload[0]["kbims_code"] == "AR-C-001"


class TestRunUpsertQdrant:
    def _mock_tei(self, dim: int = 4) -> TEIClient:
        def handler(req: httpx.Request) -> httpx.Response:
            body = req.read()
            import json as _json
            parsed = _json.loads(body)
            n = len(parsed["inputs"])
            return httpx.Response(
                200,
                json=[[1.0] + [0.0] * (dim - 1)] * n,
            )

        return TEIClient(
            url="http://tei.mock",
            model="m",
            dim=dim,
            transport=httpx.MockTransport(handler),
        )

    def test_upserts_into_qdrant(self, data_root, sample_xlsx):
        run_ingest_xlsx(data_root)
        run_normalize(data_root)

        qdrant = QdrantWrapper(QdrantClient(":memory:"))
        tei = self._mock_tei(dim=4)
        run_upsert_qdrant(
            data_root=data_root,
            tei_client=tei,
            qdrant=qdrant,
            collection="bim__test",
            dim=4,
        )
        assert qdrant.count("bim__test") == 1

    def test_is_idempotent_on_repeat(self, data_root, sample_xlsx):
        run_ingest_xlsx(data_root)
        run_normalize(data_root)

        qdrant = QdrantWrapper(QdrantClient(":memory:"))
        tei = self._mock_tei(dim=4)

        run_upsert_qdrant(data_root, tei, qdrant, collection="bim__test", dim=4)
        first = qdrant.count("bim__test")
        run_upsert_qdrant(data_root, tei, qdrant, collection="bim__test", dim=4)
        second = qdrant.count("bim__test")

        assert first == second == 1
