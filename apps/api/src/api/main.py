import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from api.bim.clients.embeddings_vllm import VLLMEmbedClient
from api.bim.clients.vllm import VLLMClient
from api.bim.predict import build_kbims_predictor, build_pps_predictor
from api.core.config import BIMSettings, settings
from api.routers import bim_attributes, conversion, health, predict, search

logger = logging.getLogger(__name__)


def _close_clients(*clients) -> None:
    """Close each client independently; one failure won't prevent others."""
    for client in clients:
        try:
            client.close()
        except Exception as e:
            logger.warning("Error closing %s: %s", type(client).__name__, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bim = BIMSettings()
    embed = VLLMEmbedClient(bim.embedding_url, bim.embedding_model, bim.embedding_dim)
    qdrant = QdrantClient(url=bim.qdrant_url, api_key=bim.qdrant_api_key)
    vllm = VLLMClient(
        url=bim.llm_url, model=bim.llm_model, timeout=bim.llm_timeout_seconds
    )

    try:
        app.state.kbims = build_kbims_predictor(
            settings=bim, embed_client=embed, qdrant_client=qdrant, vllm_client=vllm
        )
        app.state.pps = build_pps_predictor(
            settings=bim, embed_client=embed, qdrant_client=qdrant, vllm_client=vllm
        )
        app.state.qdrant = qdrant
        app.state.embed = embed
        app.state.bim = bim
    except Exception:
        _close_clients(qdrant, embed, vllm)
        raise

    yield

    _close_clients(qdrant, embed, vllm)


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(search.router)
app.include_router(conversion.router)
app.include_router(bim_attributes.router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": settings.app_name}
