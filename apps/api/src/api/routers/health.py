import logging
from typing import Literal

import httpx
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_PROBE_TIMEOUT = 3.0


class ServiceStatus(BaseModel):
    status: Literal["ok", "error"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, ServiceStatus]


def _probe_qdrant(qdrant) -> ServiceStatus:
    try:
        qdrant.get_collections()
        return ServiceStatus(status="ok")
    except Exception as e:
        return ServiceStatus(status="error", detail=str(e))


def _probe_http(url: str) -> ServiceStatus:
    try:
        with httpx.Client(timeout=_PROBE_TIMEOUT) as client:
            resp = client.get(f"{url.rstrip('/')}/health")
        if resp.is_success:
            return ServiceStatus(status="ok")
        return ServiceStatus(status="error", detail=f"HTTP {resp.status_code}")
    except Exception as e:
        return ServiceStatus(status="error", detail=str(e))


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request, response: Response) -> HealthResponse:
    qdrant = getattr(request.app.state, "qdrant", None)
    bim = getattr(request.app.state, "bim", None)

    if qdrant is None or bim is None:
        response.status_code = 503
        return HealthResponse(
            status="degraded",
            services={"api": ServiceStatus(status="error", detail="not initialized")},
        )

    services = {
        "qdrant": _probe_qdrant(qdrant),
        "embedding": _probe_http(bim.embedding_url),
        "llm": _probe_http(bim.llm_url),
    }

    overall: Literal["ok", "degraded"] = (
        "ok" if all(s.status == "ok" for s in services.values()) else "degraded"
    )
    if overall == "degraded":
        response.status_code = 503
    return HealthResponse(status=overall, services=services)
