"""Factory: BIMSettings + clients → Predictor instances.

Two separate entry points (kbims vs pps) so callers can hold whichever
predictor(s) they need. Factory is responsible for wiring the shared
PromptBuilder, retriever, and config; clients (TEI/Qdrant/vLLM) are
injected because their lifecycle (context managers, shared connections)
lives at the application boundary.
"""
from __future__ import annotations

from qdrant_client import QdrantClient

from api.bim.clients.tei import TEIClient
from api.bim.clients.vllm import VLLMClient
from api.bim.predict.catalog import NoOpCatalog
from api.bim.predict.predictor import Predictor, PredictorConfig
from api.bim.predict.prompt import PromptBuilder
from api.bim.predict.retriever import NeighborRetriever
from api.bim.predict.schemas import TargetCode
from api.core.config import BIMSettings


def build_kbims_predictor(
    *,
    settings: BIMSettings,
    tei_client: TEIClient,
    qdrant_client: QdrantClient,
    vllm_client: VLLMClient,
) -> Predictor:
    return _build(
        settings=settings,
        tei_client=tei_client,
        qdrant_client=qdrant_client,
        vllm_client=vllm_client,
        target="kbims_code",
        code_regex=settings.kbims_code_regex,
    )


def build_pps_predictor(
    *,
    settings: BIMSettings,
    tei_client: TEIClient,
    qdrant_client: QdrantClient,
    vllm_client: VLLMClient,
) -> Predictor:
    return _build(
        settings=settings,
        tei_client=tei_client,
        qdrant_client=qdrant_client,
        vllm_client=vllm_client,
        target="pps_code",
        code_regex=settings.pps_code_regex,
    )


def _build(
    *,
    settings: BIMSettings,
    tei_client: TEIClient,
    qdrant_client: QdrantClient,
    vllm_client: VLLMClient,
    target: TargetCode,
    code_regex: str,
) -> Predictor:
    config = PredictorConfig(
        target=target,
        code_format_regex=code_regex,
        catalog=NoOpCatalog(),
        k_min=settings.retrieval_k_min,
        k_multiplier=settings.retrieval_k_multiplier,
        sim_threshold=settings.mode_sim_threshold,
    )
    return Predictor(
        config=config,
        tei_client=tei_client,
        retriever=NeighborRetriever(
            qdrant_client, collection=settings.collection_name
        ),
        prompt_builder=PromptBuilder(),
        vllm_client=vllm_client,
    )
