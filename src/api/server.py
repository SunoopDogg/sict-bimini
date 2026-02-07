import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import dependencies as deps
from src.api.routes import (
    bim_attributes_router,
    conversion_router,
    prediction_router,
    search_router,
)
from src.api.schemas import APIResponse, HealthResponse
from src.rag import BIMRAGSystem
from src.utils import CSV_PATH, load_bim_attributes_from_csv

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources."""
    logger.info("Starting BIM RAG API server...")
    try:
        deps.rag_system = BIMRAGSystem()
        logger.info("BIM RAG System initialized successfully")

        deps.bim_attributes_cache = load_bim_attributes_from_csv(CSV_PATH)
        logger.info(f"Loaded {len(deps.bim_attributes_cache)} BIM attributes from CSV")

        yield
    finally:
        if deps.rag_system:
            deps.rag_system.close()
            logger.info("BIM RAG System closed")


app = FastAPI(
    title="SICT-BIMINI API",
    description="BIM Object KBIMS Part Code Prediction API using RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(prediction_router)
app.include_router(search_router)
app.include_router(conversion_router)
app.include_router(bim_attributes_router)


@app.get("/api/v1/health", response_model=APIResponse[HealthResponse])
async def health_check() -> APIResponse[HealthResponse]:
    """Check server health and service connectivity."""
    ollama_connected = False
    milvus_connected = False

    try:
        if deps.rag_system:
            # Check Milvus connection
            milvus_connected = deps.rag_system.vector_store.client.has_collection(
                deps.rag_system.vector_store.collection_name
            )

            # Check Ollama connection with a lightweight API ping
            try:
                from urllib.request import urlopen
                with urlopen(f"{deps.rag_system.llm.base_url}/api/tags", timeout=5) as resp:
                    ollama_connected = resp.status == 200
            except Exception:
                ollama_connected = False
    except Exception as e:
        logger.error(f"Health check error: {e}")

    status = "healthy" if (ollama_connected and milvus_connected) else "degraded"

    return APIResponse(
        success=True,
        data=HealthResponse(
            status=status,
            version="0.1.0",
            ollama_connected=ollama_connected,
            milvus_connected=milvus_connected,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
