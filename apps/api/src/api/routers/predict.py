import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.bim.predict import (
    EmptyRetrievalError,
    LLMGenerationError,
    PredictError,
    PredictionRequest,
    PredictionResponse,
)
from api.bim.versions import version_from_collection
from api.routers.deps import resolve_collection
from api.routers.schemas import (
    BatchItemResult,
    BatchPredictRequest,
    BatchPredictResult,
    CombinedPredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


def _call_predictor(
    pred, req: PredictionRequest, collection: str
) -> PredictionResponse:
    try:
        return pred.predict(req, collection=collection)
    except EmptyRetrievalError as e:
        raise HTTPException(status_code=422, detail="No similar objects found") from e
    except LLMGenerationError as e:
        raise HTTPException(status_code=503, detail="LLM unavailable") from e
    except PredictError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _predict_both(
    request: Request, pred_req: PredictionRequest, collection: str
) -> CombinedPredictionResponse:
    return CombinedPredictionResponse(
        version=version_from_collection(collection) or collection,
        kbims=_call_predictor(request.app.state.kbims, pred_req, collection),
        pps=_call_predictor(request.app.state.pps, pred_req, collection),
    )


@router.post("/predict", response_model=CombinedPredictionResponse)
def predict(
    request: Request,
    body: PredictionRequest,
    collection: str = Depends(resolve_collection),
) -> CombinedPredictionResponse:
    return _predict_both(request, body, collection)


@router.post("/batch-predict", response_model=BatchPredictResult)
def batch_predict(
    request: Request,
    body: BatchPredictRequest,
    collection: str = Depends(resolve_collection),
) -> BatchPredictResult:
    results: list[BatchItemResult] = []
    successful = 0
    failed = 0

    for attr in body.objects:
        pred_req = PredictionRequest(attribute=attr, n=body.n)
        try:
            prediction = _predict_both(request, pred_req, collection)
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
