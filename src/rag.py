import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from src.bim_vector_store import BIMVectorStore, MILVUS_DB_PATH
from src.utils import (
    load_prompt,
    parse_json_response,
    format_prediction_result,
    select_json_file,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.8


def format_bim_object_for_prediction(obj: dict) -> str:
    """
    Convert a BIM JSON object to a string for prediction.

    Supports both English and Korean property keys for bilingual compatibility.

    Args:
        obj: BIM object dictionary (loaded from JSON)

    Returns:
        String to use for prediction
    """
    def get_bilingual(data: dict, en_key: str, ko_key: str) -> str:
        """Get value using English key with Korean fallback."""
        return str(data.get(en_key, '') or data.get(ko_key, '')).strip()

    other = obj.get("Other", {}) or obj.get("기타", {})
    parts = []

    if obj.get("ObjectType"):
        parts.append(f"ObjectType: {obj['ObjectType']}")

    category = get_bilingual(other, "Category", "카테고리")
    if category:
        parts.append(f"Category: {category}")

    family_name = get_bilingual(other, "Family Name", "패밀리 이름")
    if family_name:
        parts.append(f"Family Name: {family_name}")

    family = get_bilingual(other, "Family", "패밀리")
    if family:
        parts.append(f"Family: {family}")

    type_val = get_bilingual(other, "Type", "유형")
    if type_val:
        parts.append(f"Type: {type_val}")

    type_id = get_bilingual(other, "Type Id", "유형 ID")
    if type_id:
        parts.append(f"Type ID: {type_id}")

    pps_code = str(other.get("조달청표준공사코드", '')).strip()
    if pps_code:
        parts.append(f"조달청표준공사코드: {pps_code}")

    return ", ".join(parts) if parts else str(obj)


class BIMRAGSystem:
    """
    RAG system for BIM object part code prediction using Ollama and Milvus.

    Combines vector similarity search with LLM generation for accurate
    KBIMS part code prediction.
    """

    def __init__(self,
                 milvus_db_path: str = MILVUS_DB_PATH,
                 ollama_model: str = DEFAULT_OLLAMA_MODEL,
                 ollama_url: str = DEFAULT_OLLAMA_URL,
                 temperature: float = DEFAULT_TEMPERATURE):
        """
        Initialize BIM RAG System.

        Args:
            milvus_db_path: Path to Milvus-lite database file
            ollama_model: Ollama model name (default: gpt-oss:latest)
            ollama_url: Ollama server URL
            temperature: LLM temperature for generation
        """
        logger.info("Initializing BIM RAG System...")

        # Initialize vector store
        logger.info(f"Loading vector store from {milvus_db_path}")
        self.vector_store = BIMVectorStore(db_path=milvus_db_path)

        # Initialize Ollama LLM
        logger.info(f"Connecting to Ollama model: {ollama_model}")
        self.llm = OllamaLLM(
            model=ollama_model,
            temperature=temperature,
            base_url=ollama_url
        )

        # Load prompt templates from files
        logger.info("Loading prompt templates from files...")
        self.prediction_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=load_prompt("kbims_prediction")
        )

        logger.info("BIM RAG System initialized successfully")

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as context string for LLM.

        Args:
            results: List of search results from vector store

        Returns:
            Formatted context string
        """
        if not results:
            return "No similar BIM objects found."

        context_parts = []
        for i, result in enumerate(results, 1):
            score = result.get('score', 0.0)
            context_parts.append(
                f"{i}. [Similarity: {score:.4f}]\n"
                f"   - Category: {result.get('category', 'N/A')}\n"
                f"   - Family Name: {result.get('family_name', 'N/A')}\n"
                f"   - KBIMS Code: {result.get('kbims_code', 'N/A')}\n"
                f"   - Family: {result.get('family', 'N/A')}\n"
                f"   - Type: {result.get('type', 'N/A')}\n"
                f"   - Type ID: {result.get('type_id', 'N/A')}"
            )

        return "\n\n".join(context_parts)

    def search(self,
               query: str,
               top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Search for similar BIM objects.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of search results with scores
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        results = self.vector_store.search(query, limit=top_k)
        logger.info(f"Found {len(results)} results")
        return results

    def predict_part_code(self,
                          bim_object_info: str,
                          top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Predict KBIMS part code for a BIM object.

        Args:
            bim_object_info: BIM object information (family, type, etc.)
            top_k: Number of similar objects to retrieve

        Returns:
            Dictionary with keys:
                - predicted_code: Predicted KBIMS part code
                - reasoning: Explanation for the prediction
                - confidence: Confidence score (0.0 to 1.0)
        """
        logger.info(f"Predicting part code for: '{bim_object_info}'")

        # Step 1: Retrieve similar BIM objects
        search_results = self.search(bim_object_info, top_k=top_k)

        # Step 2: Format context
        context = self._format_search_results(search_results)

        # Step 3: Generate prediction using LLM
        prompt = self.prediction_prompt.format(
            context=context,
            query=bim_object_info
        )

        logger.info("Generating prediction with LLM...")
        response = self.llm.invoke(prompt)

        # Step 4: Parse JSON response
        logger.info("Parsing JSON response...")
        result = parse_json_response(response)

        logger.info("Prediction complete")
        return result

    def batch_predict(self,
                      bim_objects: List[str],
                      top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Predict part codes for multiple BIM objects.

        Args:
            bim_objects: List of BIM object information strings
            top_k: Number of similar objects to retrieve per query

        Returns:
            List of dictionaries with input and prediction result (JSON dict)
        """
        results = []
        total = len(bim_objects)

        for i, bim_info in enumerate(bim_objects, 1):
            logger.info(f"Processing {i}/{total}: {bim_info[:50]}...")

            try:
                prediction = self.predict_part_code(bim_info, top_k=top_k)
                results.append({
                    "input": bim_info,
                    "prediction": prediction
                })
                logger.info(format_prediction_result(prediction))
            except ValueError as e:
                logger.error(f"Failed to parse prediction for: {bim_info[:50]}... Error: {e}")
                results.append({
                    "input": bim_info,
                    "prediction": None,
                    "error": str(e)
                })

        logger.info(f"Batch prediction complete: {total} objects processed")
        return results

    def close(self) -> None:
        """Close connections and cleanup resources."""
        self.vector_store.close()
        logger.info("BIM RAG System closed")


if __name__ == "__main__":
    # User selects JSON file
    predict_path = select_json_file()

    if predict_path is None:
        exit(0)

    logger.info(f"Loading objects from: {predict_path}")
    with open(predict_path, 'r', encoding='utf-8') as f:
        objects = json.load(f)

    logger.info(f"Loaded {len(objects)} objects")

    # Generate prediction strings from each object
    bim_infos = [format_bim_object_for_prediction(obj) for obj in objects]

    # Initialize RAG system and run batch prediction
    rag = BIMRAGSystem()
    try:
        results = rag.batch_predict(bim_infos, top_k=DEFAULT_TOP_K)

        # Save results to file
        output_path = predict_path.parent / f"{predict_path.stem}_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to: {output_path}")
    finally:
        rag.close()
