"""FastAPI server for BIM KBIMS part code prediction."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    APIResponse,
    BatchItemResult,
    BatchPredictRequest,
    BatchPredictResult,
    BIMObjectInput,
    HealthResponse,
    PredictionResult,
    SearchResponse,
    SearchResult,
    XLSXConversionResult,
)
from src.converters import convert_bim_xlsx_from_bytes
from src.rag import BIMRAGSystem

logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: BIMRAGSystem | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources."""
    global rag_system
    logger.info("Starting BIM RAG API server...")
    try:
        rag_system = BIMRAGSystem()
        logger.info("BIM RAG System initialized successfully")
        yield
    finally:
        if rag_system:
            rag_system.close()
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


def get_rag_system() -> BIMRAGSystem:
    """Get the global RAG system instance."""
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized",
        )
    return rag_system


def _run_prediction(
    rag: BIMRAGSystem, bim_object: BIMObjectInput, top_k: int
) -> PredictionResult:
    """Run KBIMS prediction for a single BIM object.

    Raises:
        ValueError: If query string is empty or prediction parsing fails
    """
    query_string = bim_object.to_query_string()
    if not query_string:
        raise ValueError("At least one field must be provided")
    result = rag.predict_part_code(query_string, top_k=top_k)
    return PredictionResult.from_dict(result)


@app.get("/api/v1/health", response_model=APIResponse[HealthResponse])
async def health_check() -> APIResponse[HealthResponse]:
    """Check server health and service connectivity."""
    ollama_connected = False
    milvus_connected = False

    try:
        if rag_system:
            # Check Milvus connection
            milvus_connected = rag_system.vector_store.client.has_collection(
                rag_system.vector_store.collection_name
            )

            # Check Ollama connection by making a simple request
            try:
                rag_system.llm.invoke("test")
                ollama_connected = True
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


@app.post("/api/v1/predict", response_model=APIResponse[PredictionResult])
async def predict_part_code(
    bim_object: BIMObjectInput,
    top_k: int = Query(default=5, ge=1, le=20, description="Number of similar objects to retrieve"),
) -> APIResponse[PredictionResult]:
    """Predict KBIMS part code for a single BIM object."""
    rag = get_rag_system()

    try:
        prediction = _run_prediction(rag, bim_object, top_k)
        return APIResponse(success=True, data=prediction)
    except ValueError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/batch-predict", response_model=APIResponse[BatchPredictResult])
async def batch_predict_part_codes(
    request: BatchPredictRequest,
) -> APIResponse[BatchPredictResult]:
    """Predict KBIMS part codes for multiple BIM objects."""
    rag = get_rag_system()

    results: list[BatchItemResult] = []
    successful = 0
    failed = 0

    for bim_object in request.objects:
        try:
            prediction = _run_prediction(rag, bim_object, request.top_k)
            results.append(BatchItemResult(input=bim_object, prediction=prediction))
            successful += 1
        except Exception as e:
            logger.error(f"Batch prediction failed for object: {e}")
            results.append(BatchItemResult(input=bim_object, error=str(e)))
            failed += 1

    return APIResponse(
        success=True,
        data=BatchPredictResult(
            results=results,
            total=len(request.objects),
            successful=successful,
            failed=failed,
        ),
    )


@app.get("/api/v1/search", response_model=APIResponse[SearchResponse])
async def search_similar_objects(
    query: str = Query(..., min_length=1, description="Search query string"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return"),
) -> APIResponse[SearchResponse]:
    """Search for similar BIM objects in the vector store."""
    rag = get_rag_system()

    try:
        results = rag.search(query, top_k=top_k)
        search_results = [
            SearchResult(
                score=r.get("score", 0.0),
                ifc_type=r.get("ifc_type", ""),
                category=r.get("category", ""),
                family_name=r.get("family_name", ""),
                kbims_code=r.get("kbims_code", ""),
                pps_code=r.get("pps_code", ""),
                family=r.get("family", ""),
                type=r.get("type", ""),
                type_id=r.get("type_id", ""),
            )
            for r in results
        ]
        return APIResponse(
            success=True,
            data=SearchResponse(results=search_results),
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}


@app.post("/api/v1/convert/xlsx-to-json", response_model=APIResponse[XLSXConversionResult])
async def convert_xlsx_to_json(
    file: UploadFile,
) -> APIResponse[XLSXConversionResult]:
    """Convert an uploaded Excel file containing BIM property tables to JSON."""
    # Validate file extension
    filename = file.filename or "uploaded_file"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content
    start_time = time.time()
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")

    # Validate file is not empty
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Validate file size
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    # Convert the file
    try:
        objects = convert_bim_xlsx_from_bytes(file_content, filename)
    except ValueError as e:
        # Missing columns or parse error
        logger.error(f"XLSX conversion validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"XLSX conversion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during conversion")

    processing_time = time.time() - start_time

    return APIResponse(
        success=True,
        data=XLSXConversionResult(
            objects=objects,
            total_objects=len(objects),
            processing_time_seconds=round(processing_time, 3),
            source_filename=filename,
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
