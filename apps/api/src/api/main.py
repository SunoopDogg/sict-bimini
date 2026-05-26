import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient
from api.bim.clients.vllm import VLLMClient
from api.bim.predict import build_kbims_predictor, build_pps_predictor
from api.core.config import BIMSettings, settings
from api.routers import health
from api.routers import bim_attributes, conversion, predict, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    bim = BIMSettings()
    embed = VLLMEmbedClient(bim.embedding_url, bim.embedding_model, bim.embedding_dim)
    qdrant = QdrantClient(url=bim.qdrant_url, api_key=bim.qdrant_api_key)
    vllm = VLLMClient(url=bim.llm_url, model=bim.llm_model, timeout=bim.llm_timeout_seconds)

    app.state.kbims = build_kbims_predictor(
        settings=bim, embed_client=embed, qdrant_client=qdrant, vllm_client=vllm
    )
    app.state.pps = build_pps_predictor(
        settings=bim, embed_client=embed, qdrant_client=qdrant, vllm_client=vllm
    )
    app.state.qdrant = qdrant
    app.state.embed = embed
    app.state.bim = bim
    yield
    with contextlib.closing(qdrant):
        pass


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(search.router)
app.include_router(conversion.router)
app.include_router(bim_attributes.router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": settings.app_name}
