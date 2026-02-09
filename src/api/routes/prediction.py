import logging

from fastapi import APIRouter, HTTPException, Query

from src.api.dependencies import get_rag_system, run_prediction
from src.api.schemas import (
    APIResponse,
    BatchItemResult,
    BatchPredictRequest,
    BatchPredictResult,
    BIMObjectInput,
    PredictionCandidates,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post("/predict", response_model=APIResponse[PredictionCandidates])
async def predict_part_code(
    bim_object: BIMObjectInput,
    top_k: int = Query(default=5, ge=1, le=20, description="Number of similar objects to retrieve"),
) -> APIResponse[PredictionCandidates]:
    """Predict KBIMS part code for a single BIM object."""
    rag = get_rag_system()

    try:
        prediction = run_prediction(rag, bim_object, top_k)
        return APIResponse(success=True, data=prediction)
    except ValueError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch-predict", response_model=APIResponse[BatchPredictResult])
async def batch_predict_part_codes(
    request: BatchPredictRequest,
) -> APIResponse[BatchPredictResult]:
    """Predict KBIMS part codes for multiple BIM objects."""
    rag = get_rag_system()

    results: list[BatchItemResult] = []
    successful = 0
    failed = 0

    for bim_object in request.objects:
        try:
            prediction = run_prediction(rag, bim_object, request.top_k)
            results.append(BatchItemResult(input=bim_object, prediction=prediction))
            successful += 1
        except Exception as e:
            logger.error(f"Batch prediction failed for object: {e}")
            results.append(BatchItemResult(input=bim_object, error=str(e)))
            failed += 1

    return APIResponse(
        success=True,
        data=BatchPredictResult(
            results=results,
            total=len(request.objects),
            successful=successful,
            failed=failed,
        ),
    )
