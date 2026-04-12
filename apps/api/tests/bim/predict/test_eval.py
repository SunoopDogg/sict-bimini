"""Unit tests for the predict-eval harness."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import Filter

from api.bim.predict.errors import EmptyRetrievalError, LLMGenerationError
from api.bim.predict.eval import (
    AggregatedMetrics,
    EvalConfig,
    EvalOutcome,
    EvalSample,
    aggregate,
    evaluate_one,
    fetch_samples,
)
from api.bim.predict.schemas import (
    PredictionCandidate,
    PredictionMode,
    PredictionRequest,
    PredictionResponse,
)
from api.bim.schemas import BIMAttribute


def _attr(ifc: str = "IfcColumn") -> BIMAttribute:
    return BIMAttribute(
        ifc_type=ifc,
        category="건축",
        family_name="RC기둥",
        family="기둥",
        type="T1",
        type_id="X",
        kbims_code="KM001",
        pps_code="",
    )


class TestEvalConfig:
    def test_minimum_fields(self, tmp_path: Path):
        cfg = EvalConfig(
            target="kbims_code",
            ifc_type=None,
            category=None,
            limit=None,
            seed=0,
            top_k=5,
            output_root=tmp_path,
        )
        assert cfg.target == "kbims_code"
        assert cfg.output_root == tmp_path

    def test_frozen(self, tmp_path: Path):
        cfg = EvalConfig(
            target="kbims_code",
            ifc_type=None,
            category=None,
            limit=None,
            seed=0,
            top_k=5,
            output_root=tmp_path,
        )
        with pytest.raises(FrozenInstanceError):
            cfg.target = "pps_code"  # type: ignore[misc]


class TestEvalSample:
    def test_holds_attribute_and_ground_truth(self):
        a = _attr()
        s = EvalSample(stable_id="abc", attribute=a, ground_truth="KM001")
        assert s.attribute is a
        assert s.ground_truth == "KM001"


class TestEvalOutcome:
    def test_constructs_with_full_fields(self):
        s = EvalSample(stable_id="abc", attribute=_attr(), ground_truth="KM001")
        o = EvalOutcome(
            sample=s,
            mode=PredictionMode.STRONG,
            top1="KM001",
            top_k_codes=["KM001", "KM002"],
            latency_ms=123.4,
            pool_size=6,
            error=None,
        )
        assert o.top1 == "KM001"
        assert o.error is None

    def test_error_outcome_has_none_mode(self):
        s = EvalSample(stable_id="abc", attribute=_attr(), ground_truth="KM001")
        o = EvalOutcome(
            sample=s, mode=None, top1=None, top_k_codes=[],
            latency_ms=10.0, pool_size=0, error="LLMGenerationError",
        )
        assert o.mode is None
        assert o.error == "LLMGenerationError"


class TestAggregatedMetricsSerialization:
    def test_round_trip_json(self):
        m = AggregatedMetrics(
            filter_summary={"target": "kbims_code", "limit": None},
            samples_total=10,
            samples_with_error=1,
            top1_correct=7,
            top1_accuracy=0.7,
            topk_correct=9,
            topk_accuracy=0.9,
            top_k=5,
            mode_distribution={"STRONG": 8, "WEAK": 1, "error": 1},
            accuracy_by_mode={"STRONG": 0.875, "WEAK": 0.0},
            accuracy_by_ifc_type={
                "IfcColumn": {"total": 10, "correct": 7, "accuracy": 0.7},
            },
            latency_p50_ms=500.0,
            latency_p95_ms=1200.0,
            errors_by_type={"LLMGenerationError": 1},
        )
        data = m.model_dump_json()
        back = AggregatedMetrics.model_validate_json(data)
        assert back == m

    def test_allows_none_accuracies(self):
        m = AggregatedMetrics(
            filter_summary={},
            samples_total=0,
            samples_with_error=0,
            top1_correct=0,
            top1_accuracy=None,
            topk_correct=0,
            topk_accuracy=None,
            top_k=5,
            mode_distribution={"STRONG": 0, "WEAK": 0, "error": 0},
            accuracy_by_mode={"STRONG": None, "WEAK": None},
            accuracy_by_ifc_type={},
            latency_p50_ms=None,
            latency_p95_ms=None,
            errors_by_type={},
        )
        assert m.top1_accuracy is None


def _record(stable_id: str, ifc: str = "IfcColumn", category: str = "건축",
            kbims: str = "KM001", pps: str = "") -> MagicMock:
    r = MagicMock()
    r.payload = {
        "stable_id": stable_id,
        "ifc_type": ifc,
        "category": category,
        "family_name": "RC기둥",
        "family": "기둥",
        "type": "T1",
        "type_id": stable_id,   # uniquify
        "kbims_code": kbims,
        "pps_code": pps,
        "source_file": "dummy.xlsx",  # extra, should be ignored by BIMAttribute
        "ingested_at": "2026-04-17T00:00:00Z",
    }
    return r


class TestFetchSamples:
    def _cfg(self, tmp_path: Path, **overrides) -> EvalConfig:
        base = dict(
            target="kbims_code",
            ifc_type=None,
            category=None,
            limit=None,
            seed=0,
            top_k=5,
            output_root=tmp_path,
        )
        base.update(overrides)
        return EvalConfig(**base)

    def test_paginates_until_offset_none(self, tmp_path: Path):
        client = MagicMock()
        page_1 = [_record(f"id-{i}") for i in range(3)]
        page_2 = [_record(f"id-{i}") for i in range(3, 5)]
        client.scroll.side_effect = [(page_1, "cursor-1"), (page_2, None)]

        samples = fetch_samples(client, "bim__test", self._cfg(tmp_path))

        assert len(samples) == 5
        assert client.scroll.call_count == 2

    def test_returns_empty_raises_value_error(self, tmp_path: Path):
        client = MagicMock()
        client.scroll.return_value = ([], None)
        with pytest.raises(ValueError, match="No samples"):
            fetch_samples(client, "bim__test", self._cfg(tmp_path))

    def test_shuffle_is_seeded(self, tmp_path: Path):
        client_a = MagicMock()
        client_b = MagicMock()
        records = [_record(f"id-{i}") for i in range(10)]
        client_a.scroll.return_value = (list(records), None)
        client_b.scroll.return_value = (list(records), None)

        out_a = fetch_samples(client_a, "c", self._cfg(tmp_path, seed=42))
        out_b = fetch_samples(client_b, "c", self._cfg(tmp_path, seed=42))

        assert [s.stable_id for s in out_a] == [s.stable_id for s in out_b]

    def test_limit_applied_after_shuffle(self, tmp_path: Path):
        client = MagicMock()
        client.scroll.return_value = ([_record(f"id-{i}") for i in range(100)], None)

        out = fetch_samples(client, "c", self._cfg(tmp_path, limit=7))
        assert len(out) == 7

    def test_limit_none_returns_all(self, tmp_path: Path):
        client = MagicMock()
        client.scroll.return_value = ([_record(f"id-{i}") for i in range(4)], None)

        out = fetch_samples(client, "c", self._cfg(tmp_path, limit=None))
        assert len(out) == 4

    def test_filter_has_label_nonempty_condition(self, tmp_path: Path):
        client = MagicMock()
        client.scroll.return_value = ([_record("a")], None)

        fetch_samples(client, "c", self._cfg(tmp_path, target="kbims_code"))
        qfilter: Filter = client.scroll.call_args.kwargs["scroll_filter"]
        dumped = qfilter.model_dump(by_alias=True)
        keys = [c["key"] for c in dumped["must"]]
        assert "kbims_code" in keys

    def test_filter_includes_ifc_type_when_set(self, tmp_path: Path):
        client = MagicMock()
        client.scroll.return_value = ([_record("a")], None)

        fetch_samples(client, "c", self._cfg(tmp_path, ifc_type="IfcColumn"))
        qfilter = client.scroll.call_args.kwargs["scroll_filter"]
        dumped = qfilter.model_dump(by_alias=True)
        keys = [c["key"] for c in dumped["must"]]
        assert "ifc_type" in keys

    def test_filter_includes_category_when_set(self, tmp_path: Path):
        client = MagicMock()
        client.scroll.return_value = ([_record("a")], None)

        fetch_samples(client, "c", self._cfg(tmp_path, category="건축"))
        qfilter = client.scroll.call_args.kwargs["scroll_filter"]
        dumped = qfilter.model_dump(by_alias=True)
        keys = [c["key"] for c in dumped["must"]]
        assert "category" in keys

    def test_builds_bim_attribute_from_payload(self, tmp_path: Path):
        client = MagicMock()
        client.scroll.return_value = ([_record("a", ifc="IfcBeam")], None)

        [sample] = fetch_samples(client, "c", self._cfg(tmp_path))
        assert sample.attribute.ifc_type == "IfcBeam"
        assert sample.ground_truth == "KM001"

    def test_pagination_guard_detects_stuck_offset(self, tmp_path: Path):
        client = MagicMock()
        # Misbehaving client returns the same non-None offset every call
        client.scroll.return_value = ([_record("a")], "stuck-cursor")

        with pytest.raises(RuntimeError, match="identical offset"):
            fetch_samples(client, "c", self._cfg(tmp_path))


def _response(codes: list[str], mode: PredictionMode = PredictionMode.STRONG,
              pool_size: int = 6) -> PredictionResponse:
    cands = [
        PredictionCandidate(
            code=c, llm_confidence=0.9 - 0.1 * i,
            retrieval_score=0.9, source="neighbor",
        )
        for i, c in enumerate(codes)
    ]
    return PredictionResponse(
        target="kbims_code",
        mode=mode,
        candidates=cands,
        low_confidence_context=(mode == PredictionMode.WEAK),
        pool_size=pool_size,
        retrieved_k=pool_size,
    )


class TestEvaluateOne:
    def _sample(self) -> EvalSample:
        return EvalSample(stable_id="abc", attribute=_attr(), ground_truth="KM001")

    def test_success_populates_outcome(self):
        predictor = MagicMock()
        predictor.predict.return_value = _response(["KM001", "KM002"])

        out = evaluate_one(self._sample(), predictor, top_k=5)

        assert out.top1 == "KM001"
        assert out.top_k_codes == ["KM001", "KM002"]
        assert out.mode == PredictionMode.STRONG
        assert out.pool_size == 6
        assert out.error is None
        assert out.latency_ms >= 0

    def test_passes_leave_one_out_filter(self):
        predictor = MagicMock()
        predictor.predict.return_value = _response(["KM001"])

        evaluate_one(self._sample(), predictor, top_k=5)

        kwargs = predictor.predict.call_args.kwargs
        assert isinstance(kwargs["extra_filter"], Filter)
        dumped = kwargs["extra_filter"].model_dump(by_alias=True)
        assert dumped["must_not"][0]["key"] == "stable_id"
        assert dumped["must_not"][0]["match"]["value"] == "abc"

    def test_passes_top_k_as_n(self):
        predictor = MagicMock()
        predictor.predict.return_value = _response(["KM001"])

        evaluate_one(self._sample(), predictor, top_k=7)

        req = predictor.predict.call_args.args[0]
        assert isinstance(req, PredictionRequest)
        assert req.n == 7

    def test_empty_retrieval_becomes_error_outcome(self):
        predictor = MagicMock()
        predictor.predict.side_effect = EmptyRetrievalError("boom")

        out = evaluate_one(self._sample(), predictor, top_k=5)

        assert out.error == "EmptyRetrievalError"
        assert out.top1 is None
        assert out.top_k_codes == []
        assert out.mode is None

    def test_llm_generation_error_becomes_error_outcome(self):
        predictor = MagicMock()
        predictor.predict.side_effect = LLMGenerationError("boom")

        out = evaluate_one(self._sample(), predictor, top_k=5)
        assert out.error == "LLMGenerationError"

    def test_infra_error_propagates(self):
        predictor = MagicMock()
        predictor.predict.side_effect = ConnectionError("qdrant down")

        with pytest.raises(ConnectionError):
            evaluate_one(self._sample(), predictor, top_k=5)

    def test_empty_candidate_list_yields_none_top1(self):
        predictor = MagicMock()
        predictor.predict.return_value = PredictionResponse(
            target="kbims_code",
            mode=PredictionMode.WEAK,
            candidates=[],
            low_confidence_context=True,
            pool_size=1,
            retrieved_k=10,
        )

        out = evaluate_one(self._sample(), predictor, top_k=5)
        assert out.top1 is None
        assert out.top_k_codes == []


def _outcome(stable_id: str, truth: str, top1: str | None,
             top_k: list[str] | None = None,
             mode: PredictionMode | None = PredictionMode.STRONG,
             ifc: str = "IfcColumn",
             latency: float = 100.0,
             error: str | None = None,
             pool_size: int = 6) -> EvalOutcome:
    s = EvalSample(
        stable_id=stable_id,
        attribute=_attr(ifc),
        ground_truth=truth,
    )
    return EvalOutcome(
        sample=s,
        mode=mode,
        top1=top1,
        top_k_codes=top_k if top_k is not None else ([top1] if top1 else []),
        latency_ms=latency,
        pool_size=pool_size,
        error=error,
    )


class TestAggregate:
    def _cfg(self, tmp_path: Path) -> EvalConfig:
        return EvalConfig(
            target="kbims_code", ifc_type=None, category=None,
            limit=None, seed=0, top_k=5, output_root=tmp_path,
        )

    def test_top1_hit_counts(self, tmp_path: Path):
        outs = [
            _outcome("a", "KM001", "KM001"),
            _outcome("b", "KM002", "KM002"),
            _outcome("c", "KM003", "KM999"),
        ]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.top1_correct == 2
        assert m.samples_total == 3
        assert m.top1_accuracy == pytest.approx(2 / 3)

    def test_topk_hit_counts(self, tmp_path: Path):
        outs = [
            _outcome("a", "KM001", "KM999", top_k=["KM999", "KM001"]),
            _outcome("b", "KM002", "KM002", top_k=["KM002"]),
            _outcome("c", "KM003", "KM999", top_k=["KM999", "KM998"]),
        ]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.topk_correct == 2
        assert m.topk_accuracy == pytest.approx(2 / 3)
        assert m.top_k == 5

    def test_mode_distribution(self, tmp_path: Path):
        outs = [
            _outcome("a", "K", "K", mode=PredictionMode.STRONG),
            _outcome("b", "K", "K", mode=PredictionMode.STRONG),
            _outcome("c", "K", "K", mode=PredictionMode.WEAK),
            _outcome("d", "K", None, mode=None, error="LLMGenerationError"),
        ]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.mode_distribution == {"STRONG": 2, "WEAK": 1, "error": 1}

    def test_accuracy_by_mode(self, tmp_path: Path):
        outs = [
            _outcome("a", "K", "K", mode=PredictionMode.STRONG),
            _outcome("b", "K", "X", mode=PredictionMode.STRONG),
            _outcome("c", "K", "K", mode=PredictionMode.WEAK),
            _outcome("d", "K", "X", mode=PredictionMode.WEAK),
        ]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.accuracy_by_mode == {"STRONG": 0.5, "WEAK": 0.5}

    def test_accuracy_by_mode_when_one_mode_absent(self, tmp_path: Path):
        outs = [_outcome("a", "K", "K", mode=PredictionMode.STRONG)]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.accuracy_by_mode == {"STRONG": 1.0, "WEAK": None}

    def test_accuracy_by_ifc_type(self, tmp_path: Path):
        outs = [
            _outcome("a", "K", "K", ifc="IfcColumn"),
            _outcome("b", "K", "X", ifc="IfcColumn"),
            _outcome("c", "K", "K", ifc="IfcBeam"),
        ]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.accuracy_by_ifc_type["IfcColumn"] == {
            "total": 2, "correct": 1, "accuracy": 0.5,
        }
        assert m.accuracy_by_ifc_type["IfcBeam"] == {
            "total": 1, "correct": 1, "accuracy": 1.0,
        }

    def test_latency_percentiles(self, tmp_path: Path):
        latencies = [float(i * 10) for i in range(1, 101)]  # 10..1000
        outs = [
            _outcome(f"id-{i}", "K", "K", latency=latency)
            for i, latency in enumerate(latencies)
        ]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.latency_p50_ms is not None
        assert m.latency_p95_ms is not None
        assert 450 <= m.latency_p50_ms <= 550
        assert 930 <= m.latency_p95_ms <= 970

    def test_latency_single_sample_uses_that_value(self, tmp_path: Path):
        outs = [_outcome("a", "K", "K", latency=123.0)]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.latency_p50_ms == 123.0
        assert m.latency_p95_ms == 123.0

    def test_errors_by_type(self, tmp_path: Path):
        outs = [
            _outcome("a", "K", None, mode=None, error="EmptyRetrievalError"),
            _outcome("b", "K", None, mode=None, error="LLMGenerationError"),
            _outcome("c", "K", None, mode=None, error="LLMGenerationError"),
            _outcome("d", "K", "K"),
        ]
        m = aggregate(outs, self._cfg(tmp_path))
        assert m.errors_by_type == {
            "EmptyRetrievalError": 1,
            "LLMGenerationError": 2,
        }
        assert m.samples_with_error == 3

    def test_empty_outcomes_yields_none_metrics(self, tmp_path: Path):
        m = aggregate([], self._cfg(tmp_path))
        assert m.samples_total == 0
        assert m.top1_accuracy is None
        assert m.topk_accuracy is None
        assert m.latency_p50_ms is None
        assert m.latency_p95_ms is None
        assert m.accuracy_by_mode == {"STRONG": None, "WEAK": None}
        assert m.accuracy_by_ifc_type == {}

    def test_filter_summary_echoes_cfg(self, tmp_path: Path):
        cfg = EvalConfig(
            target="pps_code", ifc_type="IfcBeam", category="건축",
            limit=50, seed=7, top_k=3, output_root=tmp_path,
        )
        outs = [_outcome("a", "K", "K", ifc="IfcBeam")]
        m = aggregate(outs, cfg)
        assert m.filter_summary["target"] == "pps_code"
        assert m.filter_summary["ifc_type"] == "IfcBeam"
        assert m.filter_summary["category"] == "건축"
        assert m.filter_summary["limit"] == 50
        assert m.filter_summary["seed"] == 7
        assert m.filter_summary["top_k"] == 3
