"""Prediction orchestrator — 8-step pipeline.

embed → retrieve → pool → evaluate_mode → prompt → LLM → assemble → catalog

The Predictor itself is sync, stateless, and target-agnostic: injection of
PredictorConfig picks kbims vs pps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from api.bim.clients.tei import TEIClient
from api.bim.clients.vllm import VLLMClient
from api.bim.predict.catalog import CatalogSource
from api.bim.predict.errors import EmptyRetrievalError
from api.bim.predict.pool import build_pool, evaluate_mode
from api.bim.predict.prompt import PromptBuilder
from api.bim.predict.retriever import NeighborRetriever
from api.bim.predict.schemas import (
    CandidatePool,
    PredictionCandidate,
    PredictionMode,
    PredictionRequest,
    PredictionResponse,
    build_strong_schema,
    build_weak_schema,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictorConfig:
    target: Literal["kbims_code", "pps_code"]
    code_format_regex: str
    catalog: CatalogSource
    k_min: int = 10
    k_multiplier: int = 3
    sim_threshold: float = 0.55


class Predictor:
    def __init__(
        self,
        *,
        config: PredictorConfig,
        tei_client: TEIClient,
        retriever: NeighborRetriever,
        prompt_builder: PromptBuilder,
        vllm_client: VLLMClient,
    ) -> None:
        self._config = config
        self._tei = tei_client
        self._retriever = retriever
        self._prompt = prompt_builder
        self._vllm = vllm_client

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        cfg = self._config
        attr = request.attribute
        n = request.n
        tag = f"predict[{cfg.target}] stable_id={attr.stable_id}"

        # 1. embed
        [vec] = self._tei.embed([attr.embed_text()])

        # 2. retrieve
        top_k = max(cfg.k_min, n * cfg.k_multiplier)
        neighbors = self._retriever.search(
            vec, code_field=cfg.target, k=top_k,
        )
        if not neighbors:
            raise EmptyRetrievalError(
                f"Qdrant returned 0 neighbors with non-empty {cfg.target}"
            )

        # 3. pool
        pool = build_pool(neighbors, cfg.target)

        # 4. evaluate mode
        mode = evaluate_mode(pool, n, sim_threshold=cfg.sim_threshold)
        logger.info(
            "%s mode=%s top1=%.3f pool_size=%d → request LLM",
            tag, mode.value, pool.top1_score, pool.unique_count,
        )

        # 5. prompt
        prompt_text = self._prompt.build(
            target=cfg.target,
            mode=mode,
            attribute=attr,
            pool=pool,
            n=n,
        )

        # 6. LLM
        schema_cls = (
            build_strong_schema(list(pool.code_to_max_score))
            if mode == PredictionMode.STRONG
            else build_weak_schema(cfg.code_format_regex)
        )
        raw = self._vllm.generate_json(
            prompt=prompt_text,
            response_schema=schema_cls.model_json_schema(),
        )
        parsed = schema_cls.model_validate_json(raw)

        # 7. assemble — decorate with retrieval_score and source, apply catalog hook
        candidates = [
            self._decorate_candidate(c, pool, mode)
            for c in parsed.candidates
        ]
        candidates = [cfg.catalog.validate(c) for c in candidates]

        if len(candidates) < n:
            logger.warning(
                "%s LLM returned %d candidates (requested %d) — partial",
                tag, len(candidates), n,
            )

        return PredictionResponse(
            target=cfg.target,
            mode=mode,
            candidates=candidates,
            low_confidence_context=(mode == PredictionMode.WEAK),
            pool_size=pool.unique_count,
            retrieved_k=len(neighbors),
        )

    @staticmethod
    def _decorate_candidate(
        raw_candidate: PredictionCandidate,
        pool: CandidatePool,
        mode: PredictionMode,
    ) -> PredictionCandidate:
        """Re-stamp retrieval_score/source from the pool rather than trusting the LLM.

        Strong: code MUST be in pool (Literal enforced) → score is always present.
        Weak: code MAY happen to be in pool. If so, mark as neighbor with score;
        otherwise keep as generated with None.
        """
        pool_score = pool.code_to_max_score.get(raw_candidate.code)
        if mode == PredictionMode.STRONG:
            return raw_candidate.model_copy(
                update={
                    "source": "neighbor",
                    "retrieval_score": pool_score,
                }
            )
        # WEAK
        if pool_score is not None:
            return raw_candidate.model_copy(
                update={"source": "neighbor", "retrieval_score": pool_score}
            )
        return raw_candidate.model_copy(
            update={"source": "generated", "retrieval_score": None}
        )
