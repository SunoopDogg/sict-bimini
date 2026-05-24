import pytest
from pydantic import ValidationError

from api.bim.predict.schemas import (
    CandidatePool,
    Neighbor,
    PredictionCandidate,
    PredictionMode,
    PredictionRequest,
    PredictionResponse,
    build_strong_schema,
    build_weak_schema,
)


class TestPredictionMode:
    def test_string_values(self):
        assert PredictionMode.STRONG.value == "strong"
        assert PredictionMode.WEAK.value == "weak"


class TestPredictionRequest:
    def test_default_n_is_5(self, sample_attribute):
        req = PredictionRequest(attribute=sample_attribute)
        assert req.n == 5

    def test_n_bounds(self, sample_attribute):
        with pytest.raises(ValidationError):
            PredictionRequest(attribute=sample_attribute, n=0)
        with pytest.raises(ValidationError):
            PredictionRequest(attribute=sample_attribute, n=21)


class TestPredictionCandidate:
    def test_neighbor_requires_retrieval_score(self):
        with pytest.raises(ValidationError):
            PredictionCandidate(
                code="KM001",
                llm_confidence=0.9,
                retrieval_score=None,
                source="neighbor",
            )

    def test_generated_must_not_have_retrieval_score(self):
        with pytest.raises(ValidationError):
            PredictionCandidate(
                code="KM001",
                llm_confidence=0.9,
                retrieval_score=0.8,
                source="generated",
            )

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            PredictionCandidate(
                code="KM001",
                llm_confidence=1.5,
                retrieval_score=0.8,
                source="neighbor",
            )

    def test_generated_with_none_score_is_valid(self):
        c = PredictionCandidate(
            code="KM001",
            llm_confidence=0.4,
            retrieval_score=None,
            source="generated",
        )
        assert c.source == "generated"
        assert c.retrieval_score is None


class TestPredictionResponse:
    def _candidate(self):
        return PredictionCandidate(
            code="KM001",
            llm_confidence=0.9,
            retrieval_score=0.8,
            source="neighbor",
        )

    def test_minimum_fields(self):
        resp = PredictionResponse(
            target="kbims_code",
            mode=PredictionMode.STRONG,
            candidates=[self._candidate()],
            low_confidence_context=False,
            pool_size=5,
            retrieved_k=15,
        )
        assert resp.target == "kbims_code"
        assert resp.pool_size == 5


class TestNeighbor:
    def test_both_codes_optional_empty(self):
        n = Neighbor(
            stable_id="abc",
            score=0.8,
            kbims_code="",
            pps_code="",
            ifc_type="IfcColumn",
            category="건축",
        )
        assert n.kbims_code == ""


class TestCandidatePool:
    def test_pool_fields(self):
        pool = CandidatePool(
            code_to_max_score={"KM001": 0.8, "KM002": 0.6},
            top1_score=0.8,
            unique_count=2,
        )
        assert pool.unique_count == 2
        assert pool.code_to_max_score["KM001"] == 0.8


class TestStrongSchemaBuilder:
    def test_code_restricted_to_pool(self):
        Schema = build_strong_schema(frozenset(["KM001", "KM002"]), target="kbims_code")
        payload = {
            "target": "kbims_code",
            "mode": "strong",
            "candidates": [
                {
                    "code": "KM001",
                    "llm_confidence": 0.9,
                    "retrieval_score": 0.8,
                    "source": "neighbor",
                }
            ],
            "low_confidence_context": False,
            "pool_size": 2,
            "retrieved_k": 10,
        }
        parsed = Schema.model_validate(payload)
        assert parsed.candidates[0].code == "KM001"

    def test_code_outside_pool_rejected(self):
        Schema = build_strong_schema(frozenset(["KM001", "KM002"]), target="kbims_code")
        bad = {
            "target": "kbims_code",
            "mode": "strong",
            "candidates": [
                {
                    "code": "KM999",
                    "llm_confidence": 0.9,
                    "retrieval_score": 0.8,
                    "source": "neighbor",
                }
            ],
            "low_confidence_context": False,
            "pool_size": 2,
            "retrieved_k": 10,
        }
        with pytest.raises(ValidationError):
            Schema.model_validate(bad)

    def test_json_schema_code_field_has_pool_enum(self):
        Schema = build_strong_schema(frozenset(["KM001", "KM002"]), target="kbims_code")
        json_schema = Schema.model_json_schema()
        # Locate the dynamically-created candidate definition (its $ref
        # is referenced from candidates.items)
        code_def = next(
            d for d in json_schema["$defs"].values()
            if "code" in d.get("properties", {})
            and "enum" in d["properties"]["code"]
        )
        assert sorted(code_def["properties"]["code"]["enum"]) == ["KM001", "KM002"]

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_strong_schema(frozenset(), target="kbims_code")

    def test_raw_llm_response_without_retrieval_score_accepted(self):
        """LLM은 retrieval_score를 만들지 않는다 — Predictor._decorate_candidate가
        pool에서 re-stamp한다. raw 응답에 retrieval_score가 없어도 파싱 통과해야 함
        (부모 validator의 source='neighbor' → retrieval_score 요구 규칙은 동적
        subclass에서 무력화)."""
        Schema = build_strong_schema(frozenset(["E77"]), target="kbims_code")
        raw_llm_payload = {
            "target": "kbims_code",
            "mode": "strong",
            "candidates": [
                {"code": "E77", "llm_confidence": 0.9, "source": "neighbor"}
            ],
            "low_confidence_context": False,
            "pool_size": 1,
            "retrieved_k": 10,
        }
        parsed = Schema.model_validate(raw_llm_payload)
        assert parsed.candidates[0].retrieval_score is None


class TestWeakSchemaBuilder:
    def test_code_matching_regex_accepted(self):
        Schema = build_weak_schema(r"^KM\d+$", target="kbims_code")
        payload = {
            "target": "kbims_code",
            "mode": "weak",
            "candidates": [
                {
                    "code": "KM12345",
                    "llm_confidence": 0.4,
                    "retrieval_score": None,
                    "source": "generated",
                }
            ],
            "low_confidence_context": True,
            "pool_size": 1,
            "retrieved_k": 10,
        }
        Schema.model_validate(payload)

    def test_raw_llm_response_without_retrieval_score_accepted(self):
        """WEAK 모드도 동일: LLM이 source='neighbor' 판단 시 retrieval_score는
        Predictor가 재주입하므로 raw에서 누락돼도 파싱 통과해야 함."""
        Schema = build_weak_schema(r"^KM\d+$", target="kbims_code")
        raw_llm_payload = {
            "target": "kbims_code",
            "mode": "weak",
            "candidates": [
                {"code": "KM001", "llm_confidence": 0.9, "source": "neighbor"}
            ],
            "low_confidence_context": True,
            "pool_size": 1,
            "retrieved_k": 10,
        }
        parsed = Schema.model_validate(raw_llm_payload)
        assert parsed.candidates[0].retrieval_score is None

    def test_code_violating_regex_rejected(self):
        Schema = build_weak_schema(r"^KM\d+$", target="kbims_code")
        bad = {
            "target": "kbims_code",
            "mode": "weak",
            "candidates": [
                {
                    "code": "not-a-code",
                    "llm_confidence": 0.4,
                    "retrieval_score": None,
                    "source": "generated",
                }
            ],
            "low_confidence_context": True,
            "pool_size": 1,
            "retrieved_k": 10,
        }
        with pytest.raises(ValidationError):
            Schema.model_validate(bad)

    def test_json_schema_code_field_has_pattern(self):
        regex = r"^KM\d+$"
        Schema = build_weak_schema(regex, target="kbims_code")
        json_schema = Schema.model_json_schema()
        code_def = next(
            d for d in json_schema["$defs"].values()
            if "code" in d.get("properties", {})
            and "pattern" in d["properties"]["code"]
        )
        assert code_def["properties"]["code"]["pattern"] == regex
