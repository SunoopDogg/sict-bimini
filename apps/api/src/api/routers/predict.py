import logging

from fastapi import APIRouter, HTTPException, Request

from api.bim.predict import (
    EmptyRetrievalError,
    LLMGenerationError,
    PredictError,
    PredictionRequest,
)
from api.routers.schemas import (
    BatchItemResult,
    BatchPredictRequest,
    BatchPredictResult,
    CombinedPredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


def _predict_both(
    request: Request, pred_req: PredictionRequest
) -> CombinedPredictionResponse:
    kbims_pred = request.app.state.kbims
    pps_pred = request.app.state.pps

    try:
        kbims_result = kbims_pred.predict(pred_req)
    except EmptyRetrievalError as e:
        raise HTTPException(status_code=422, detail="No similar objects found") from e
    except LLMGenerationError as e:
        raise HTTPException(status_code=503, detail="LLM unavailable") from e
    except PredictError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        pps_result = pps_pred.predict(pred_req)
    except EmptyRetrievalError as e:
        raise HTTPException(status_code=422, detail="No similar objects found") from e
    except LLMGenerationError as e:
        raise HTTPException(status_code=503, detail="LLM unavailable") from e
    except PredictError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return CombinedPredictionResponse(kbims=kbims_result, pps=pps_result)


@router.post("/predict", response_model=CombinedPredictionResponse)
def predict(request: Request, body: PredictionRequest) -> CombinedPredictionResponse:
    return _predict_both(request, body)


@router.post("/batch-predict", response_model=BatchPredictResult)
def batch_predict(request: Request, body: BatchPredictRequest) -> BatchPredictResult:
    results: list[BatchItemResult] = []
    successful = 0
    failed = 0

    for attr in body.objects:
        pred_req = PredictionRequest(attribute=attr, n=body.n)
        try:
            prediction = _predict_both(request, pred_req)
            results.append(BatchItemResult(input=attr, prediction=prediction))
            successful += 1
        except HTTPException as e:
            results.append(BatchItemResult(input=attr, error=e.detail))
            failed += 1
        except Exception as e:
            logger.error("Batch prediction error: %s", e)
            results.append(BatchItemResult(input=attr, error=str(e)))
            failed += 1

    return BatchPredictResult(
        results=results,
        total=len(body.objects),
        successful=successful,
        failed=failed,
    )
