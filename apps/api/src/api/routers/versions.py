from fastapi import APIRouter, Request

from api.bim.versions import VersionListResponse, list_versions

router = APIRouter(tags=["versions"])


@router.get("/versions", response_model=VersionListResponse)
def get_versions(request: Request) -> VersionListResponse:
    return VersionListResponse(versions=list_versions(request.app.state.qdrant))
