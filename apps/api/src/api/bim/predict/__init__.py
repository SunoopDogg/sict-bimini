"""RAG-based BIM code prediction module.

Public API: construct a Predictor via the factory, then call `predict()`.

Error handling: `PredictError` is the base for all domain-level failures
surfaced by `Predictor.predict`. Subclasses: `EmptyRetrievalError`,
`LLMGenerationError`. Infra-layer errors from `VLLMClient`
(`VLLMError`, `VLLMSchemaError`, `VLLMTimeoutError`) are translated to
`LLMGenerationError` at the Predictor boundary — callers only need to
catch `PredictError`.
"""
from __future__ import annotations

from api.bim.predict.errors import (
    EmptyRetrievalError,
    LLMGenerationError,
    PredictError,
)
from api.bim.predict.factory import build_kbims_predictor, build_pps_predictor
from api.bim.predict.predictor import Predictor, PredictorConfig
from api.bim.predict.schemas import (
    PredictionCandidate,
    PredictionMode,
    PredictionRequest,
    PredictionResponse,
)

__all__ = [
    # Public factory entry points
    "build_kbims_predictor",
    "build_pps_predictor",
    # Core types callers need
    "Predictor",
    "PredictorConfig",
    "PredictionRequest",
    "PredictionResponse",
    "PredictionCandidate",
    "PredictionMode",
    # Error hierarchy (single-root for call sites)
    "PredictError",
    "EmptyRetrievalError",
    "LLMGenerationError",
]
