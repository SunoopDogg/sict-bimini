from pathlib import Path
from unittest.mock import patch

import httpx
from typer.testing import CliRunner

from api.cli.__main__ import app

runner = CliRunner()


@patch("api.cli.__main__.run_ingest_xlsx")
def test_ingest_xlsx_calls_pipeline_function(mock_run, tmp_path: Path):
    result = runner.invoke(app, ["ingest-xlsx", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(tmp_path)


@patch("api.cli.__main__.run_normalize")
def test_normalize_calls_pipeline_function(mock_run, tmp_path: Path):
    result = runner.invoke(app, ["normalize", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(tmp_path)


@patch("api.cli.__main__.run_upsert_qdrant")
@patch("api.cli.__main__.TEIClient")
@patch("api.cli.__main__.QdrantWrapper")
def test_upsert_qdrant_wires_clients(
    mock_qdrant_cls, mock_tei_cls, mock_run, tmp_path: Path
):
    mock_tei_cls.return_value.__enter__.return_value = "TEI"
    mock_qdrant_cls.from_settings.return_value = "QW"

    result = runner.invoke(
        app,
        [
            "upsert-qdrant",
            "--data-root", str(tmp_path),
            "--experiment-id", "qwen8b_d2048",
            "--dim", "2048",
            "--tei-url", "http://tei",
            "--qdrant-url", "http://qdrant",
            "--model", "Qwen/Qwen3-Embedding-8B",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["data_root"] == tmp_path
    assert kwargs["collection"] == "bim__qwen8b_d2048"
    assert kwargs["dim"] == 2048
    assert kwargs["tei_client"] == "TEI"
    assert kwargs["qdrant"] == "QW"


@patch("api.cli.__main__.run_upsert_qdrant")
@patch("api.cli.__main__.run_normalize")
@patch("api.cli.__main__.run_ingest_xlsx")
@patch("api.cli.__main__.TEIClient")
@patch("api.cli.__main__.QdrantWrapper")
def test_pipeline_runs_all_three_stages_in_order(
    mock_qdrant_cls, mock_tei_cls, mock_ingest, mock_norm, mock_upsert, tmp_path: Path
):
    mock_tei_cls.return_value.__enter__.return_value = "TEI"
    mock_qdrant_cls.from_settings.return_value = "QW"

    call_order: list[str] = []
    mock_ingest.side_effect = lambda *_a, **_kw: call_order.append("ingest")
    mock_norm.side_effect = lambda *_a, **_kw: call_order.append("normalize")
    mock_upsert.side_effect = lambda *_a, **_kw: call_order.append("upsert")

    result = runner.invoke(
        app,
        [
            "pipeline",
            "--data-root", str(tmp_path),
            "--experiment-id", "e1",
            "--dim", "8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert call_order == ["ingest", "normalize", "upsert"]


@patch("api.cli.__main__.run_ingest_xlsx")
def test_ingest_xlsx_uses_env_data_root_when_no_flag(
    mock_run, monkeypatch, tmp_path: Path
):
    """BIM_DATA_ROOT env var should drive data_root when --data-root omitted."""
    monkeypatch.setenv("BIM_DATA_ROOT", str(tmp_path))

    result = runner.invoke(app, ["ingest-xlsx"])  # no --data-root flag
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(tmp_path)


@patch("api.cli.__main__.run_ingest_xlsx")
def test_ingest_xlsx_cli_flag_overrides_env(mock_run, monkeypatch, tmp_path: Path):
    """Explicit --data-root should override BIM_DATA_ROOT env."""
    monkeypatch.setenv("BIM_DATA_ROOT", "/ignored")
    result = runner.invoke(app, ["ingest-xlsx", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(tmp_path)


def _healthy_models_handler(model_id: str):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"data": [{"id": model_id}]},
        )
    return handler


def test_llm_check_succeeds_when_model_served(monkeypatch):
    monkeypatch.setenv("BIM_LLM_URL", "http://vllm.local")
    monkeypatch.setenv("BIM_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    captured: dict = {}

    def fake_get(url, *args, **kwargs):
        captured["url"] = url
        return httpx.Response(
            200,
            json={"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("api.cli.__main__.httpx.get", fake_get)
    result = runner.invoke(app, ["llm-check"])
    assert result.exit_code == 0, result.output
    assert "Qwen/Qwen2.5-7B-Instruct" in result.output
    assert captured["url"].endswith("/v1/models")


def test_llm_check_fails_when_model_missing(monkeypatch):
    monkeypatch.setenv("BIM_LLM_URL", "http://vllm.local")
    monkeypatch.setenv("BIM_LLM_MODEL", "missing-model")

    def fake_get(url, *args, **kwargs):
        return httpx.Response(
            200,
            json={"data": [{"id": "other-model"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("api.cli.__main__.httpx.get", fake_get)
    result = runner.invoke(app, ["llm-check"])
    assert result.exit_code != 0
    assert "missing-model" in result.output


def test_llm_check_fails_on_network_error(monkeypatch):
    monkeypatch.setenv("BIM_LLM_URL", "http://vllm.local")
    monkeypatch.setenv("BIM_LLM_MODEL", "m")

    def fake_get(*_a, **_kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("api.cli.__main__.httpx.get", fake_get)
    result = runner.invoke(app, ["llm-check"])
    assert result.exit_code != 0
    assert "down" in result.output or "vLLM" in result.output
