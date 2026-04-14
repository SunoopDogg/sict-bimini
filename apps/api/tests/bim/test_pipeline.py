import json
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient
from api.bim.clients.qdrant import QdrantWrapper
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

    def test_continues_on_bad_xlsx_and_processes_valid_sibling(
        self, data_root, sample_xlsx
    ):
        """A corrupt .xlsx alongside a valid one should be skipped, not halt run."""
        (data_root / "xlsx" / "corrupt.xlsx").write_bytes(b"not a real xlsx")
        run_ingest_xlsx(data_root)
        # Valid file's output must still exist
        assert (data_root / "json" / "raw" / "fixture.json").exists()
        # Corrupt file should produce no output (parser raised, loop continued)
        assert not (data_root / "json" / "raw" / "corrupt.json").exists()


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
    def _mock_embed(self, dim: int = 4) -> VLLMEmbedClient:
        def handler(req: httpx.Request) -> httpx.Response:
            body = req.read()
            import json as _json
            parsed = _json.loads(body)
            n = len(parsed["input"])
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "m",
                    "data": [
                        {"index": i, "object": "embedding",
                         "embedding": [1.0] + [0.0] * (dim - 1)}
                        for i in range(n)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        return VLLMEmbedClient(
            url="http://embed.mock",
            model="m",
            dim=dim,
            transport=httpx.MockTransport(handler),
        )

    def test_upserts_into_qdrant(self, data_root, sample_xlsx):
        run_ingest_xlsx(data_root)
        run_normalize(data_root)

        qdrant = QdrantWrapper(QdrantClient(":memory:"))
        embed = self._mock_embed(dim=4)
        run_upsert_qdrant(
            data_root=data_root,
            embed_client=embed,
            qdrant=qdrant,
            collection="bim__test",
            dim=4,
        )
        assert qdrant.count("bim__test") == 1

    def test_is_idempotent_on_repeat(self, data_root, sample_xlsx):
        run_ingest_xlsx(data_root)
        run_normalize(data_root)

        qdrant = QdrantWrapper(QdrantClient(":memory:"))
        embed = self._mock_embed(dim=4)

        run_upsert_qdrant(data_root, embed, qdrant, collection="bim__test", dim=4)
        first = qdrant.count("bim__test")
        run_upsert_qdrant(data_root, embed, qdrant, collection="bim__test", dim=4)
        second = qdrant.count("bim__test")

        assert first == second == 1

    def test_skips_empty_normalized_file(self, data_root):
        """Normalized file with zero attrs should not invoke embed or Qdrant upsert."""
        (data_root / "json" / "normalized" / "empty.json").write_text(
            "[]", encoding="utf-8"
        )

        embed_calls = {"n": 0}

        def handler(_req):
            embed_calls["n"] += 1
            return httpx.Response(
                200,
                json={"object": "list", "model": "m", "data": [],
                      "usage": {"prompt_tokens": 0, "total_tokens": 0}},
            )

        embed = VLLMEmbedClient(
            url="http://embed.mock", model="m", dim=4,
            transport=httpx.MockTransport(handler),
        )
        qdrant = QdrantWrapper(QdrantClient(":memory:"))
        total = run_upsert_qdrant(
            data_root=data_root,
            embed_client=embed,
            qdrant=qdrant,
            collection="bim__test",
            dim=4,
        )
        assert total == 0
        assert embed_calls["n"] == 0  # no embed call made for empty attrs

    def test_batches_at_configured_batch_size(self, data_root):
        """Stage 3 should issue ceil(N / batch_size) embed calls."""
        # Seed normalized dir with 5 distinct BIMAttribute records
        from pydantic import TypeAdapter

        from api.bim.schemas import BIMAttribute

        attrs = [
            BIMAttribute(
                ifc_type="IfcColumn",
                category="건축",
                family_name=f"F{i}",
                family="기둥",
                type="T1",
                type_id=f"X{i}",
                kbims_code=f"AR-C-{i:03d}",
            )
            for i in range(5)
        ]
        (data_root / "json" / "normalized" / "fixture.json").write_text(
            TypeAdapter(list[BIMAttribute]).dump_json(attrs, indent=2).decode("utf-8"),
            encoding="utf-8",
        )

        embed_call_sizes: list[int] = []

        def handler(req: httpx.Request) -> httpx.Response:
            import json as _json
            parsed = _json.loads(req.read())
            n = len(parsed["input"])
            embed_call_sizes.append(n)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "m",
                    "data": [
                        {"index": i, "object": "embedding",
                         "embedding": [1.0, 0.0, 0.0, 0.0]}
                        for i in range(n)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        embed = VLLMEmbedClient(
            url="http://embed.mock",
            model="m",
            dim=4,
            transport=httpx.MockTransport(handler),
        )
        qdrant = QdrantWrapper(QdrantClient(":memory:"))
        total = run_upsert_qdrant(
            data_root=data_root,
            embed_client=embed,
            qdrant=qdrant,
            collection="bim__test",
            dim=4,
            batch_size=2,  # expect 3 calls: 2, 2, 1
        )
        assert total == 5
        assert embed_call_sizes == [2, 2, 1]
