import logging

from fastapi import APIRouter, Query, Request

from api.bim.clients.embeddings_vllm import VLLMEmbedError
from api.bim.schemas import bim_attr_from_payload
from api.routers.schemas import (
    SearchResponse,
    SearchResult,
    raise_embedding_unavailable,
)

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
    bim_settings = request.app.state.bim

    try:
        [vector] = embed.embed([query])
    except (VLLMEmbedError, ValueError) as e:
        raise_embedding_unavailable(e)

    response = qdrant.query_points(
        collection_name=bim_settings.collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )

    results = [
        SearchResult(attribute=attr, score=point.score)
        for point in response.points
        if (attr := bim_attr_from_payload(point.payload or {})) is not None
    ]

    return SearchResponse(results=results)
