from fastapi import APIRouter, Request

from api.bim.versions import VersionService
from api.routers.schemas import DbVersion, VersionListResponse

router = APIRouter(tags=["versions"])


@router.get("/versions", response_model=VersionListResponse)
def list_versions(request: Request) -> VersionListResponse:
    service = VersionService(request.app.state.qdrant)
    versions = [
        DbVersion(name=v.name, points=v.points) for v in service.list_versions()
    ]
    return VersionListResponse(versions=versions)
