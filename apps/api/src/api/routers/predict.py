import asyncio
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
    pred, req: PredictionRequest, collection: str, vector: list[float]
) -> PredictionResponse:
    try:
        return pred.predict(req, collection=collection, vector=vector)
    except EmptyRetrievalError as e:
        raise HTTPException(status_code=422, detail="No similar objects found") from e
    except LLMGenerationError as e:
        raise HTTPException(status_code=503, detail="LLM unavailable") from e
    except PredictError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _predict_both(
    request: Request, pred_req: PredictionRequest, collection: str
) -> CombinedPredictionResponse:
    attr = pred_req.attribute
    # Embed once, then fan out: both predictors retrieve with the same vector.
    [vec] = await asyncio.to_thread(
        request.app.state.embed.embed, [attr.embed_text()]
    )
    kbims_res, pps_res = await asyncio.gather(
        asyncio.to_thread(
            _call_predictor, request.app.state.kbims, pred_req, collection, vec
        ),
        asyncio.to_thread(
            _call_predictor, request.app.state.pps, pred_req, collection, vec
        ),
        return_exceptions=True,
    )
    # Reraise in positional order so kbims's error wins (matches prior behavior).
    for res in (kbims_res, pps_res):
        if isinstance(res, BaseException):
            raise res
    return CombinedPredictionResponse(
        version=version_from_collection(collection) or collection,
        kbims=kbims_res,
        pps=pps_res,
    )


@router.post("/predict", response_model=CombinedPredictionResponse)
async def predict(
    request: Request,
    body: PredictionRequest,
    collection: str = Depends(resolve_collection),
) -> CombinedPredictionResponse:
    return await _predict_both(request, body, collection)


async def _predict_one(
    request: Request, attr, n: int, collection: str
) -> BatchItemResult:
    pred_req = PredictionRequest(attribute=attr, n=n)
    try:
        prediction = await _predict_both(request, pred_req, collection)
        return BatchItemResult(input=attr, prediction=prediction)
    except HTTPException as e:
        return BatchItemResult(input=attr, error=e.detail)
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        return BatchItemResult(input=attr, error=str(e))


@router.post("/batch-predict", response_model=BatchPredictResult)
async def batch_predict(
    request: Request,
    body: BatchPredictRequest,
    collection: str = Depends(resolve_collection),
) -> BatchPredictResult:
    # Predict all objects in the request concurrently (order preserved by gather).
    # The client sends them in small chunks (PREDICT_CHUNK), so each chunk's
    # objects fan out together; downstream vLLM batches the concurrent load.
    results = await asyncio.gather(
        *(_predict_one(request, attr, body.n, collection) for attr in body.objects)
    )
    successful = sum(1 for r in results if r.error is None)
    failed = len(results) - successful

    return BatchPredictResult(
        results=results,
        total=len(body.objects),
        successful=successful,
        failed=failed,
    )
