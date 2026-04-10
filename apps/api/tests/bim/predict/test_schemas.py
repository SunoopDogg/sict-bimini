import pytest
from pydantic import ValidationError

from api.bim.predict.schemas import (
    CandidatePool,
    Neighbor,
    PredictionCandidate,
    PredictionMode,
    PredictionRequest,
    PredictionResponse,
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
