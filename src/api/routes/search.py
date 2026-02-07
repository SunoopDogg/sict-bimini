import logging

from fastapi import APIRouter, HTTPException, Query

from src.api.dependencies import get_rag_system
from src.api.schemas import APIResponse, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.get("/search", response_model=APIResponse[SearchResponse])
async def search_similar_objects(
    query: str = Query(..., min_length=1, description="Search query string"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return"),
) -> APIResponse[SearchResponse]:
    """Search for similar BIM objects in the vector store."""
    rag = get_rag_system()

    try:
        results = rag.search(query, top_k=top_k)
        search_results = [
            SearchResult(**{k: v for k, v in r.items() if k != "id"})
            for r in results
        ]
        return APIResponse(
            success=True,
            data=SearchResponse(results=search_results),
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")
