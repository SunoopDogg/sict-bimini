import logging

from fastapi import HTTPException

from src.api.schemas import BIMObjectInput, PredictionCandidates, PredictionResult
from src.rag import BIMRAGSystem
from src.utils import BIMAttribute

logger = logging.getLogger(__name__)

# Global instances managed by server lifespan
rag_system: BIMRAGSystem | None = None
bim_attributes_cache: list[BIMAttribute] | None = None


def get_rag_system() -> BIMRAGSystem:
    """Get the global RAG system instance."""
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized",
        )
    return rag_system


def run_prediction(
    rag: BIMRAGSystem, bim_object: BIMObjectInput, top_k: int
) -> PredictionCandidates:
    """Run KBIMS prediction for a single BIM object.

    Raises:
        ValueError: If query string is empty or prediction parsing fails
    """
    query_string = bim_object.to_query_string()
    if not query_string:
        raise ValueError("At least one field must be provided")
    predictions = rag.predict_part_code(query_string, top_k=top_k)
    return PredictionCandidates(
        predictions=[PredictionResult.from_dict(p) for p in predictions]
    )
