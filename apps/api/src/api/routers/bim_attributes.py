import logging

from fastapi import APIRouter, HTTPException, Query, Request

from api.bim.attribute_service import BIMAttributeService
from api.bim.clients.embeddings_vllm import VLLMEmbedError
from api.routers.schemas import (
    BIMAttributeCreateRequest,
    BIMAttributeCreateResponse,
    BIMAttributeListResponse,
    raise_embedding_unavailable,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bim-attributes"])


@router.get("/bim-attributes", response_model=BIMAttributeListResponse)
def list_bim_attributes(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> BIMAttributeListResponse:
    service = BIMAttributeService(
        request.app.state.qdrant, request.app.state.bim.collection_name
    )
    items, total, total_pages = service.get_page(page, page_size)
    return BIMAttributeListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.post("/bim-attributes", response_model=BIMAttributeCreateResponse)
def create_bim_attributes(
    request: Request,
    body: BIMAttributeCreateRequest,
) -> BIMAttributeCreateResponse:
    embed = request.app.state.embed
    service = BIMAttributeService(
        request.app.state.qdrant, request.app.state.bim.collection_name
    )

    deduped = service.dedup(body.items)

    try:
        vectors = embed.embed([attr.embed_text() for attr in deduped])
        if len(vectors) != len(deduped):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Embedding service returned {len(vectors)} vectors"
                    f" for {len(deduped)} inputs"
                ),
            )
    except (VLLMEmbedError, ValueError) as e:
        raise_embedding_unavailable(e)

    service.upsert_batch(deduped, vectors)
    return BIMAttributeCreateResponse(added=len(deduped), total=service.count())
