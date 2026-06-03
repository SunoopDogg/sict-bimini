import logging
from unittest.mock import MagicMock

import pytest

from api.bim.clients.vllm import VLLMError
from api.bim.predict.catalog import NoOpCatalog
from api.bim.predict.errors import EmptyRetrievalError, LLMGenerationError, PredictError
from api.bim.predict.predictor import Predictor, PredictorConfig
from api.bim.predict.schemas import (
    Neighbor,
    PredictionMode,
    PredictionRequest,
)


def _n(score, code):
    return Neighbor(
        stable_id=f"id-{code}",
        score=score,
        kbims_code=code,
        pps_code="",
        ifc_type="IfcColumn",
        category="건축",
    )


@pytest.fixture
def kbims_config():
    return PredictorConfig(
        target="kbims_code",
        code_format_regex=r"^KM\d+$",
        catalog=NoOpCatalog(),
        k_min=10,
        k_multiplier=3,
        sim_threshold=0.55,
    )


@pytest.fixture
def wired_predictor(kbims_config):
    embed = MagicMock()
    embed.embed.return_value = [[0.1, 0.2, 0.3]]
    retriever = MagicMock()
    vllm = MagicMock()
    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "RENDERED PROMPT"
    return dict(
        predictor=Predictor(
            config=kbims_config,
            embed_client=embed,
            retriever=retriever,
            prompt_builder=prompt_builder,
            vllm_client=vllm,
        ),
        embed=embed,
        retriever=retriever,
        vllm=vllm,
        prompt_builder=prompt_builder,
    )


class TestPredictorRetrievalParams:
    def test_top_k_uses_max_of_k_min_and_n_times_multiplier(
        self, wired_predictor, sample_attribute
    ):
        w = wired_predictor
        w["retriever"].search.return_value = [_n(0.9, f"KM{i:03d}") for i in range(15)]
        w["vllm"].generate_json.return_value = _valid_strong_json_str(
            pool_codes=[f"KM{i:03d}" for i in range(15)],
            n=5,
        )

        w["predictor"].predict(PredictionRequest(attribute=sample_attribute, n=5))

        w["retriever"].search.assert_called_once()
        kwargs = w["retriever"].search.call_args.kwargs
        assert kwargs["code_field"] == "kbims_code"
        assert kwargs["k"] == 15  # max(10, 5*3)

    def test_extra_filter_is_passed_to_retriever(
        self, wired_predictor, sample_attribute
    ):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        w = wired_predictor
        w["retriever"].search.return_value = [_n(0.9, f"KM{i:03d}") for i in range(6)]
        pool_codes = [f"KM{i:03d}" for i in range(6)]
        w["vllm"].generate_json.return_value = _valid_strong_json_str(pool_codes, n=5)

        f = Filter(
            must_not=[FieldCondition(key="stable_id", match=MatchValue(value="xyz"))]
        )
        w["predictor"].predict(
            PredictionRequest(attribute=sample_attribute, n=5),
            extra_filter=f,
        )

        assert w["retriever"].search.call_args.kwargs["extra_filter"] is f

    def test_provided_vector_skips_embed_and_is_used_for_retrieval(
        self, wired_predictor, sample_attribute
    ):
        w = wired_predictor
        w["retriever"].search.return_value = [_n(0.9, f"KM{i:03d}") for i in range(6)]
        w["vllm"].generate_json.return_value = _valid_strong_json_str(
            pool_codes=[f"KM{i:03d}" for i in range(6)], n=5
        )

        w["predictor"].predict(
            PredictionRequest(attribute=sample_attribute, n=5),
            vector=[0.5, 0.6, 0.7],
        )

        w["embed"].embed.assert_not_called()
        assert w["retriever"].search.call_args.args[0] == [0.5, 0.6, 0.7]

    def test_top_k_floor_is_k_min(self, wired_predictor, sample_attribute):
        """If n=1 and K_MULTIPLIER=3, top_k = max(10, 3) = 10."""
        w = wired_predictor
        w["retriever"].search.return_value = [_n(0.9, "KM001")]
        w["vllm"].generate_json.return_value = _valid_strong_json_str(
            pool_codes=["KM001"], n=1
        )

        w["predictor"].predict(PredictionRequest(attribute=sample_attribute, n=1))
        assert w["retriever"].search.call_args.kwargs["k"] == 10


class TestPredictorModes:
    def test_strong_path_when_pool_ample_and_high_similarity(
        self, wired_predictor, sample_attribute
    ):
        w = wired_predictor
        pool_codes = [f"KM{i:03d}" for i in range(6)]
        w["retriever"].search.return_value = [_n(0.9, c) for c in pool_codes]
        w["vllm"].generate_json.return_value = _valid_strong_json_str(pool_codes, n=5)

        resp = w["predictor"].predict(
            PredictionRequest(attribute=sample_attribute, n=5)
        )

        assert resp.mode == PredictionMode.STRONG
        assert resp.low_confidence_context is False
        assert resp.pool_size == 6
        assert all(c.source == "neighbor" for c in resp.candidates)
        assert all(c.retrieval_score is not None for c in resp.candidates)

    def test_strong_path_when_pool_smaller_than_n_but_top1_confident(
        self, wired_predictor, sample_attribute
    ):
        """Regression: pool diversity < n used to force WEAK (`cond_a`), which
        broke the PPS IfcRamp case where 14/15 neighbors agreed on 'AD' but
        pool_size=2 < n=5. Now the only WEAK signal is top1 < sim_threshold."""
        w = wired_predictor
        w["retriever"].search.return_value = [_n(0.9, "KM001"), _n(0.8, "KM002")]
        w["vllm"].generate_json.return_value = _valid_strong_json_str(
            pool_codes=["KM001", "KM002"], n=2
        )

        resp = w["predictor"].predict(
            PredictionRequest(attribute=sample_attribute, n=5)
        )
        assert resp.mode == PredictionMode.STRONG
        assert resp.low_confidence_context is False
        # Partial response: pool has 2 codes, requested 5 — LLM returns 2.
        assert len(resp.candidates) == 2

    def test_weak_path_when_top1_below_threshold(
        self, wired_predictor, sample_attribute
    ):
        w = wired_predictor
        pool_codes = [f"KM{i:03d}" for i in range(10)]
        w["retriever"].search.return_value = [_n(0.40, c) for c in pool_codes]
        w["vllm"].generate_json.return_value = _valid_weak_json_str(n=5)

        resp = w["predictor"].predict(
            PredictionRequest(attribute=sample_attribute, n=5)
        )
        assert resp.mode == PredictionMode.WEAK


class TestPredictorErrors:
    def test_empty_retrieval_raises(self, wired_predictor, sample_attribute):
        w = wired_predictor
        w["retriever"].search.return_value = []

        with pytest.raises(EmptyRetrievalError):
            w["predictor"].predict(
                PredictionRequest(attribute=sample_attribute, n=5)
            )

    def test_embed_error_propagates(self, wired_predictor, sample_attribute):
        w = wired_predictor
        w["embed"].embed.side_effect = RuntimeError("embed down")

        with pytest.raises(RuntimeError, match="embed down"):
            w["predictor"].predict(
                PredictionRequest(attribute=sample_attribute, n=5)
            )

    def test_partial_response_logged_but_returned(
        self, wired_predictor, sample_attribute, caplog
    ):
        """LLM returns fewer than n candidates — warn but return."""
        w = wired_predictor
        pool_codes = [f"KM{i:03d}" for i in range(6)]
        w["retriever"].search.return_value = [_n(0.9, c) for c in pool_codes]
        # Only 2 candidates returned for n=5
        w["vllm"].generate_json.return_value = _valid_strong_json_str(
            pool_codes, n=2
        )

        with caplog.at_level(logging.WARNING):
            resp = w["predictor"].predict(
                PredictionRequest(attribute=sample_attribute, n=5)
            )

        assert len(resp.candidates) == 2
        assert any("partial" in r.message.lower() for r in caplog.records)

    def test_vllm_error_is_translated_to_llm_generation_error(
        self, wired_predictor, sample_attribute
    ):
        w = wired_predictor
        w["retriever"].search.return_value = [_n(0.9, f"KM{i:03d}") for i in range(6)]
        w["vllm"].generate_json.side_effect = VLLMError("backend down")

        with pytest.raises(LLMGenerationError) as excinfo:
            w["predictor"].predict(
                PredictionRequest(attribute=sample_attribute, n=5)
            )
        # single-root contract: callers only catch PredictError
        assert isinstance(excinfo.value, PredictError)
        # original infra error preserved as __cause__
        assert isinstance(excinfo.value.__cause__, VLLMError)

    def test_invalid_llm_json_is_translated_to_llm_generation_error(
        self, wired_predictor, sample_attribute
    ):
        w = wired_predictor
        w["retriever"].search.return_value = [_n(0.9, f"KM{i:03d}") for i in range(6)]
        # Malformed JSON (missing required fields) — schema validation fails
        w["vllm"].generate_json.return_value = '{"target": "kbims_code"}'

        with pytest.raises(LLMGenerationError):
            w["predictor"].predict(
                PredictionRequest(attribute=sample_attribute, n=5)
            )

    def test_strong_branch_raises_when_code_not_in_pool(self, kbims_config):
        """Belt-and-suspenders: if the guided_json Literal constraint is
        ever violated server-side, _decorate_candidate must raise
        LLMGenerationError rather than silently producing an invalid
        PredictionCandidate (source='neighbor', retrieval_score=None)."""
        from api.bim.predict.predictor import Predictor
        from api.bim.predict.schemas import (
            CandidatePool,
            PredictionCandidate,
            PredictionMode,
        )

        predictor = Predictor(
            config=kbims_config,
            embed_client=MagicMock(),
            retriever=MagicMock(),
            prompt_builder=MagicMock(),
            vllm_client=MagicMock(),
        )
        pool = CandidatePool(
            code_to_max_score={"KM001": 0.9}, top1_score=0.9, unique_count=1
        )
        # A LLM-returned candidate whose code is NOT in the pool — should be
        # impossible when guided_json works, but this simulates the failure mode.
        rogue = PredictionCandidate(
            code="KM999",
            llm_confidence=0.8,
            retrieval_score=0.5,
            source="neighbor",
        )
        with pytest.raises(LLMGenerationError, match="STRONG invariant broken"):
            predictor._decorate_candidate(rogue, pool, PredictionMode.STRONG)


class TestPredictorAssembly:
    def test_pps_config_uses_pps_code_field(self, sample_attribute):
        cfg = PredictorConfig(
            target="pps_code",
            code_format_regex=r"^[A-Z]-\d+(-\d+)*$",
            catalog=NoOpCatalog(),
            k_min=10,
            k_multiplier=3,
            sim_threshold=0.55,
        )
        embed = MagicMock()
        embed.embed.return_value = [[0.1]]
        retriever = MagicMock()
        retriever.search.return_value = [
            Neighbor(
                stable_id="x",
                score=0.9,
                kbims_code="",
                pps_code="A-1",
                ifc_type="IfcBeam",
                category="건축",
            )
        ]
        vllm = MagicMock()
        vllm.generate_json.return_value = _valid_strong_json_str(
            pool_codes=["A-1"], n=1, target="pps_code"
        )
        prompt = MagicMock()
        prompt.build.return_value = "P"

        predictor = Predictor(
            config=cfg,
            embed_client=embed,
            retriever=retriever,
            prompt_builder=prompt,
            vllm_client=vllm,
        )
        resp = predictor.predict(PredictionRequest(attribute=sample_attribute, n=1))

        assert retriever.search.call_args.kwargs["code_field"] == "pps_code"
        assert resp.target == "pps_code"


# ---------- helpers ----------


def _valid_strong_json_str(
    pool_codes: list[str], *, n: int, target: str = "kbims_code"
) -> str:
    import json as _json
    items = [
        {
            "code": pool_codes[i],
            "llm_confidence": 0.9 - (0.1 * i),
            "retrieval_score": 0.9,
            "source": "neighbor",
        }
        for i in range(min(n, len(pool_codes)))
    ]
    return _json.dumps(
        {
            "target": target,
            "mode": "strong",
            "candidates": items,
            "low_confidence_context": False,
            "pool_size": len(pool_codes),
            "retrieved_k": len(pool_codes),
        }
    )


def _valid_weak_json_str(*, n: int, target: str = "kbims_code") -> str:
    import json as _json
    code = "KM999" if target == "kbims_code" else "A-9"
    items = [
        {
            "code": code,
            "llm_confidence": 0.3,
            "retrieval_score": None,
            "source": "generated",
        }
        for _ in range(n)
    ]
    return _json.dumps(
        {
            "target": target,
            "mode": "weak",
            "candidates": items,
            "low_confidence_context": True,
            "pool_size": 1,
            "retrieved_k": 10,
        }
    )
