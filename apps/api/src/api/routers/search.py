import logging

from fastapi import APIRouter, HTTPException, Query, Request

from api.bim.clients.embeddings_vllm import VLLMEmbedError
from api.routers.schemas import SearchResponse, SearchResult, bim_attr_from_payload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search_similar_objects(
    request: Request,
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
) -> SearchResponse:
    embed = request.app.state.embed
    qdrant = request.app.state.qdrant
    bim = request.app.state.bim

    try:
        [vector] = embed.embed([query])
    except VLLMEmbedError as e:
        raise HTTPException(
            status_code=503, detail=f"Embedding service unavailable: {e}"
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service returned unexpected result: {e}",
        ) from e

    response = qdrant.query_points(
        collection_name=bim.collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for point in response.points:
        attr = bim_attr_from_payload(point.payload or {})
        if attr is None:
            continue
        results.append(SearchResult(attribute=attr, score=point.score))

    return SearchResponse(results=results)
