from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from typer.testing import CliRunner

from api.bim.predict.eval import AggregatedMetrics, NoSamplesError
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
@patch("api.cli.__main__.VLLMEmbedClient")
@patch("api.cli.__main__.QdrantWrapper")
def test_upsert_qdrant_wires_clients(
    mock_qdrant_cls, mock_embed_cls, mock_run, tmp_path: Path
):
    mock_embed_cls.return_value.__enter__.return_value = "EMB"
    mock_qdrant_cls.from_settings.return_value = "QW"

    result = runner.invoke(
        app,
        [
            "upsert-qdrant",
            "--data-root", str(tmp_path),
            "--experiment-id", "qwen4b_d2048",
            "--dim", "2048",
            "--embedding-url", "http://embed",
            "--qdrant-url", "http://qdrant",
            "--model", "Qwen/Qwen3-Embedding-4B",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["data_root"] == tmp_path
    assert kwargs["collection"] == "bim__qwen4b_d2048"
    assert kwargs["dim"] == 2048
    assert kwargs["embed_client"] == "EMB"
    assert kwargs["qdrant"] == "QW"


@patch("api.cli.__main__.run_upsert_qdrant")
@patch("api.cli.__main__.run_normalize")
@patch("api.cli.__main__.run_ingest_xlsx")
@patch("api.cli.__main__.VLLMEmbedClient")
@patch("api.cli.__main__.QdrantWrapper")
def test_pipeline_runs_all_three_stages_in_order(
    mock_qdrant_cls, mock_embed_cls, mock_ingest, mock_norm, mock_upsert, tmp_path: Path
):
    mock_embed_cls.return_value.__enter__.return_value = "EMB"
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


def _metrics_stub() -> AggregatedMetrics:
    return AggregatedMetrics(
        filter_summary={"target": "kbims_code"},
        samples_total=3,
        samples_with_error=0,
        top1_correct=2, top1_accuracy=0.667,
        topk_correct=3, topk_accuracy=1.0,
        top_k=5,
        mode_distribution={"STRONG": 2, "WEAK": 1, "error": 0},
        accuracy_by_mode={"STRONG": 1.0, "WEAK": 0.0},
        accuracy_by_ifc_type={
            "IfcColumn": {"total": 3, "correct": 2, "accuracy": 0.667},
        },
        latency_p50_ms=100.0, latency_p95_ms=200.0,
        errors_by_type={},
    )


@patch("api.cli.__main__.run_eval")
@patch("api.cli.__main__.VLLMClient")
@patch("api.cli.__main__.VLLMEmbedClient")
@patch("api.cli.__main__.QdrantClient")
@patch("api.cli.__main__.build_kbims_predictor")
def test_predict_eval_builds_cfg_and_prints_summary(
    mock_build, mock_qdrant_cls, mock_embed_cls, mock_vllm_cls,
    mock_run_eval, tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("BIM_DATA_ROOT", str(tmp_path))

    mock_embed_cls.return_value.__enter__.return_value = "EMB"
    mock_vllm_cls.return_value.__enter__.return_value = "VLLM"
    mock_qdrant_cls.return_value = MagicMock()
    mock_build.return_value = "PRED"

    run_dir = tmp_path / "reports" / "predict-eval" / "2026-04-17T00-00-00Z_kbims_code"
    mock_run_eval.return_value = (_metrics_stub(), run_dir)

    result = runner.invoke(
        app,
        [
            "predict-eval",
            "--target", "kbims_code",
            "--ifc-type", "IfcColumn",
            "--limit", "3",
            "--seed", "42",
            "--top-k", "5",
        ],
    )

    assert result.exit_code == 0, result.output
    # cfg was constructed from flags
    call = mock_run_eval.call_args
    cfg = call.args[0]
    assert cfg.target == "kbims_code"
    assert cfg.ifc_type == "IfcColumn"
    assert cfg.limit == 3
    assert cfg.seed == 42
    assert cfg.top_k == 5
    assert cfg.output_root == tmp_path / "reports" / "predict-eval"
    # stdout contains summary markers
    assert "predict-eval" in result.output
    assert "Top-1" in result.output
    assert "kbims_code" in result.output


@patch("api.cli.__main__.run_eval")
@patch("api.cli.__main__.VLLMClient")
@patch("api.cli.__main__.VLLMEmbedClient")
@patch("api.cli.__main__.QdrantClient")
@patch("api.cli.__main__.build_pps_predictor")
@patch("api.cli.__main__.build_kbims_predictor")
def test_predict_eval_pps_target_uses_pps_builder(
    mock_kbims_build, mock_pps_build, mock_qdrant_cls, mock_embed_cls,
    mock_vllm_cls, mock_run_eval, tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("BIM_DATA_ROOT", str(tmp_path))

    mock_embed_cls.return_value.__enter__.return_value = "EMB"
    mock_vllm_cls.return_value.__enter__.return_value = "VLLM"
    mock_qdrant_cls.return_value = MagicMock()
    mock_pps_build.return_value = "PRED-PPS"
    mock_kbims_build.return_value = "PRED-KBIMS"
    mock_run_eval.return_value = (
        _metrics_stub(), tmp_path / "reports" / "predict-eval" / "x_pps_code",
    )

    result = runner.invoke(
        app, ["predict-eval", "--target", "pps_code"]
    )
    assert result.exit_code == 0, result.output
    mock_pps_build.assert_called_once()
    mock_kbims_build.assert_not_called()


def test_predict_eval_rejects_unknown_target(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BIM_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["predict-eval", "--target", "bogus"])
    assert result.exit_code != 0
    assert "kbims_code" in result.output or "target" in result.output


@patch("api.cli.__main__.run_eval")
@patch("api.cli.__main__.VLLMClient")
@patch("api.cli.__main__.VLLMEmbedClient")
@patch("api.cli.__main__.QdrantClient")
@patch("api.cli.__main__.build_kbims_predictor")
def test_predict_eval_empty_samples_exits_with_code_1(
    mock_build, mock_qdrant_cls, mock_embed_cls, mock_vllm_cls,
    mock_run_eval, tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("BIM_DATA_ROOT", str(tmp_path))

    mock_embed_cls.return_value.__enter__.return_value = "EMB"
    mock_vllm_cls.return_value.__enter__.return_value = "VLLM"
    mock_qdrant_cls.return_value = MagicMock()
    mock_build.return_value = "PRED"
    mock_run_eval.side_effect = NoSamplesError("No samples match filter")

    result = runner.invoke(app, ["predict-eval", "--target", "kbims_code"])
    assert result.exit_code != 0
    assert "No samples" in result.output
