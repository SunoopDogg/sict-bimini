"""Unit tests for the predict-eval harness."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from api.bim.predict.eval import (
    AggregatedMetrics,
    EvalConfig,
    EvalOutcome,
    EvalSample,
)
from api.bim.predict.schemas import PredictionMode
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
