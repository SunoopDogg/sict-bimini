from pydantic import BaseModel, Field

from api.bim.predict.schemas import PredictionResponse
from api.bim.schemas import BIMAttribute


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
