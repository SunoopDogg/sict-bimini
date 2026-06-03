from fastapi import APIRouter, Request

from api.routers.schemas import MetaResponse

router = APIRouter(tags=["meta"])


@router.get("/meta", response_model=MetaResponse)
def get_meta(request: Request) -> MetaResponse:
    bim = request.app.state.bim
    return MetaResponse(
        llm_model=bim.llm_model,
        embedding_model=bim.embedding_model,
    )
