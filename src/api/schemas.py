"""Pydantic schemas for API request/response validation."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from src.utils import BIMAttribute

T = TypeVar("T")


class BIMObjectInput(BaseModel):
    """Input schema for BIM object prediction."""

    object_type: str = Field(
        default="",
        description="IFC object type (e.g., IfcColumn, IfcBeam)",
        examples=["IfcColumn"],
    )
    category: str = Field(
        default="",
        description="Object category",
        examples=["구조기둥"],
    )
    family_name: str = Field(
        default="",
        description="Family name",
        examples=["RC기둥"],
    )
    family: str = Field(
        default="",
        description="Family description",
        examples=["콘크리트-직사각형-기둥"],
    )
    type: str = Field(
        default="",
        description="Type specification",
        examples=["400 x 600mm"],
    )
    type_id: str = Field(
        default="",
        description="Type identifier",
        examples=["1234567"],
    )
    pps_code: str = Field(
        default="",
        description="PPS (Public Procurement Service) code",
        examples=["41.23.15.10"],
    )

    def to_bim_attribute(self) -> BIMAttribute:
        """Convert to BIMAttribute for internal processing."""
        return BIMAttribute(
            ifc_type=self.object_type,
            category=self.category,
            family_name=self.family_name,
            kbims_code="",
            pps_code=self.pps_code,
            family=self.family,
            type=self.type,
            type_id=self.type_id,
        )

    def to_query_string(self) -> str:
        """Convert to query string matching the embedding format for vector search."""
        attr = self.to_bim_attribute()
        return attr.to_search_text()


class PredictionResult(BaseModel):
    """Prediction result schema."""

    predicted_code: str | None = Field(
        description="Predicted KBIMS part code",
        examples=["25.21.10.01"],
    )
    reasoning: str = Field(
        description="Explanation for the prediction",
        examples=["Based on similarity analysis..."],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)",
        examples=[0.85],
    )

    @classmethod
    def from_dict(cls, data: dict) -> "PredictionResult":
        """Create PredictionResult from a prediction result dictionary."""
        return cls(
            predicted_code=data.get("predicted_code"),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.0),
        )


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool = Field(description="Whether the request was successful")
    data: T | None = Field(default=None, description="Response data")
    error: str | None = Field(default=None, description="Error message if failed")


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction."""

    objects: list[BIMObjectInput] = Field(
        description="List of BIM objects to predict",
        min_length=1,
        max_length=100,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar objects to retrieve",
    )


class BatchItemResult(BaseModel):
    """Single item result in batch prediction."""

    input: BIMObjectInput = Field(description="Original input object")
    prediction: PredictionResult | None = Field(
        default=None,
        description="Prediction result",
    )
    error: str | None = Field(
        default=None,
        description="Error message if prediction failed",
    )


class BatchPredictResult(BaseModel):
    """Result schema for batch prediction."""

    results: list[BatchItemResult] = Field(description="List of prediction results")
    total: int = Field(description="Total number of objects processed")
    successful: int = Field(description="Number of successful predictions")
    failed: int = Field(description="Number of failed predictions")


class SearchResult(BaseModel):
    """Single search result item."""

    score: float = Field(description="Similarity score")
    ifc_type: str = Field(default="", description="IFC type identifier")
    category: str = Field(default="", description="Object category")
    family_name: str = Field(default="", description="Family name")
    kbims_code: str = Field(default="", description="KBIMS part code")
    pps_code: str = Field(default="", description="PPS code")
    family: str = Field(default="", description="Family description")
    type: str = Field(default="", description="Type specification")
    type_id: str = Field(default="", description="Type identifier")


class SearchResponse(BaseModel):
    """Response schema for search endpoint."""

    results: list[SearchResult] = Field(description="List of search results")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(description="Server status", examples=["healthy"])
    version: str = Field(description="API version", examples=["0.1.0"])
    ollama_connected: bool = Field(description="Whether Ollama is connected")
    milvus_connected: bool = Field(description="Whether Milvus is connected")


class XLSXConversionResult(BaseModel):
    """Response schema for XLSX to JSON conversion endpoint."""

    objects: list[dict[str, Any]] = Field(
        description="Converted BIM objects",
        examples=[[{"IFCType": "IfcColumn", "GlobalID": "abc123", "Name": "기둥"}]],
    )
    total_objects: int = Field(
        description="Total count of converted objects",
        examples=[42],
    )
    processing_time_seconds: float = Field(
        description="Processing time in seconds",
        examples=[1.23],
    )
    source_filename: str = Field(
        description="Original filename",
        examples=["속성테이블(10층).xlsx"],
    )
