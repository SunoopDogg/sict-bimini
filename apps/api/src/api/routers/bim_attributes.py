import logging
import math
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request
from qdrant_client.models import PointStruct

from api.bim.clients.embeddings_vllm import VLLMEmbedError
from api.routers.schemas import (
    BIMAttributeCreateRequest,
    BIMAttributeCreateResponse,
    BIMAttributeListResponse,
    bim_attr_from_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bim-attributes"])


@router.get("/bim-attributes", response_model=BIMAttributeListResponse)
def list_bim_attributes(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> BIMAttributeListResponse:
    qdrant = request.app.state.qdrant
    bim = request.app.state.bim

    count_result = qdrant.count(collection_name=bim.collection_name, exact=True)
    total = count_result.count
    total_pages = math.ceil(total / page_size) if total > 0 else 0

    # Skip (page-1)*page_size records without loading payload
    skip = (page - 1) * page_size
    offset = None

    while skip > 0:
        batch_size = min(skip, 250)
        points, offset = qdrant.scroll(
            collection_name=bim.collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        skip -= len(points)
        if not points or offset is None:
            return BIMAttributeListResponse(
                items=[], total=total, page=page,
                page_size=page_size, total_pages=total_pages,
            )

    # Fetch the requested page
    points, _ = qdrant.scroll(
        collection_name=bim.collection_name,
        limit=page_size,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    items = [
        attr for point in points
        if (attr := bim_attr_from_payload(point.payload or {})) is not None
    ]

    return BIMAttributeListResponse(
        items=items, total=total, page=page,
        page_size=page_size, total_pages=total_pages,
    )


@router.post("/bim-attributes", response_model=BIMAttributeCreateResponse)
def create_bim_attributes(
    request: Request,
    body: BIMAttributeCreateRequest,
) -> BIMAttributeCreateResponse:
    qdrant = request.app.state.qdrant
    embed = request.app.state.embed
    bim = request.app.state.bim

    # Dedup by stable_id (last wins) — matches pipeline normalizer semantics
    deduped = list({attr.stable_id: attr for attr in body.items}.values())

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
    except VLLMEmbedError as e:
        raise HTTPException(
            status_code=503, detail=f"Embedding service unavailable: {e}"
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service returned unexpected result: {e}",
        ) from e

    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
    stable_ids = [attr.stable_id for attr in deduped]
    points = [
        PointStruct(
            id=sid,
            vector=vec,
            payload={
                **attr.model_dump(),
                "stable_id": sid,
                "source_file": "",
                "ingested_at": ingested_at,
            },
        )
        for attr, vec, sid in zip(deduped, vectors, stable_ids, strict=False)
    ]
    qdrant.upsert(collection_name=bim.collection_name, points=points, wait=True)

    count_result = qdrant.count(collection_name=bim.collection_name, exact=True)
    return BIMAttributeCreateResponse(added=len(deduped), total=count_result.count)
