import logging

from pydantic import BaseModel, Field, ValidationError

from api.bim.predict.schemas import PredictionResponse
from api.bim.schemas import BIMAttribute

_logger = logging.getLogger(__name__)


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
