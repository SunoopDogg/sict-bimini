import logging
from typing import NoReturn

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from api.bim.clients.embeddings_vllm import VLLMEmbedError
from api.bim.predict.schemas import PredictionResponse
from api.bim.schemas import BIMAttribute

_logger = logging.getLogger(__name__)


def raise_embedding_unavailable(exc: Exception) -> NoReturn:
    """Map an embedding-client failure to a 503 HTTPException.

    Shared by routers that call ``embed.embed(...)``; keeps the
    VLLMEmbedError / ValueError → 503 mapping in one place.
    """
    if isinstance(exc, VLLMEmbedError):
        detail = f"Embedding service unavailable: {exc}"
    else:
        detail = f"Embedding service returned unexpected result: {exc}"
    raise HTTPException(status_code=503, detail=detail) from exc


def bim_attr_from_payload(payload: dict) -> BIMAttribute | None:
    """Parse BIMAttribute from a Qdrant point payload; returns None on invalid data."""
    try:
        return BIMAttribute.model_validate(payload)
    except ValidationError:
        _logger.warning(
            "Skipping point with invalid payload: stable_id=%s",
            payload.get("stable_id"),
        )
        return None


class CombinedPredictionResponse(BaseModel):
    kbims: PredictionResponse
    pps: PredictionResponse


class BatchPredictRequest(BaseModel):
    objects: list[BIMAttribute] = Field(min_length=1, max_length=100)
    n: int = Field(default=5, ge=1, le=20)


class BatchItemResult(BaseModel):
    input: BIMAttribute
    prediction: CombinedPredictionResponse | None = None
    error: str | None = None


class BatchPredictResult(BaseModel):
    results: list[BatchItemResult]
    total: int
    successful: int
    failed: int


class SearchResult(BaseModel):
    attribute: BIMAttribute
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class BIMAttributeListResponse(BaseModel):
    items: list[BIMAttribute]
    total: int
    page: int
    page_size: int
    total_pages: int


class BIMAttributeCreateRequest(BaseModel):
    items: list[BIMAttribute] = Field(min_length=1, max_length=1000)


class BIMAttributeCreateResponse(BaseModel):
    added: int
    total: int


class XLSXConversionResult(BaseModel):
    objects: list[dict]
    total_objects: int
    processing_time_seconds: float
    source_filename: str
