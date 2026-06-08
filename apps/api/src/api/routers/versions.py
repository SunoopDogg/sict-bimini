import re

from fastapi import APIRouter, HTTPException, Request

from api.bim.clients.embeddings_vllm import VLLMEmbedError
from api.bim.version_service import create_version as create_version_svc
from api.bim.versions import (
    VersionListResponse,
    collection_for_version,
    list_versions,
)
from api.routers.schemas import (
    VersionCreateRequest,
    VersionCreateResponse,
    raise_embedding_unavailable,
)

router = APIRouter(tags=["versions"])

_VERSION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@router.get("/versions", response_model=VersionListResponse)
def get_versions(request: Request) -> VersionListResponse:
    return VersionListResponse(versions=list_versions(request.app.state.qdrant))


@router.post("/versions", response_model=VersionCreateResponse, status_code=201)
def create_version(
    request: Request, body: VersionCreateRequest
) -> VersionCreateResponse:
    qdrant = request.app.state.qdrant
    settings = request.app.state.bim

    if not _VERSION_NAME_RE.fullmatch(body.name):
        raise HTTPException(
            status_code=422,
            detail="Invalid version name; allowed chars: A-Z a-z 0-9 _ -",
        )

    new_collection = collection_for_version(body.name)
    if qdrant.collection_exists(new_collection):
        raise HTTPException(
            status_code=409, detail=f"Version already exists: {body.name}"
        )

    base_collection: str | None = None
    if body.base is not None:
        base_collection = collection_for_version(body.base)
        if not qdrant.collection_exists(base_collection):
            raise HTTPException(
                status_code=404, detail=f"Unknown base version: {body.base}"
            )

    if not body.items and base_collection is None:
        raise HTTPException(
            status_code=422,
            detail="items required when no base version is given",
        )

    # When cloning a base, its vectors must match the embed dim we'll create the
    # new collection at, else the copy/upsert would mix sizes. dim is therefore
    # always settings.embedding_dim — the base check just rejects a mismatch.
    if base_collection is not None:
        base_dim = qdrant.get_collection(base_collection).config.params.vectors.size
        if base_dim != settings.embedding_dim:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Base version dim {base_dim} != embedding dim "
                    f"{settings.embedding_dim}; cannot mix dimensions"
                ),
            )

    try:
        copied, added, total = create_version_svc(
            qdrant,
            request.app.state.embed,
            new_collection=new_collection,
            base_collection=base_collection,
            dim=settings.embedding_dim,
            items=body.items,
        )
    except (VLLMEmbedError, ValueError) as e:
        raise_embedding_unavailable(e)

    return VersionCreateResponse(
        version=body.name, copied=copied, added=added, total=total
    )
