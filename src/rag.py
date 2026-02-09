import logging
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from src.bim_vector_store import BIMVectorStore, MILVUS_DB_PATH
from src.utils import BIM_ATTRIBUTE_FIELDS, load_prompt, parse_json_response, format_prediction_result
from src.utils.bim_attribute import _FIELD_LABELS

logger = logging.getLogger(__name__)

# Constants
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.8


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
            ollama_model: Ollama model name
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

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
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
            field_lines = "\n".join(
                f"   - {_FIELD_LABELS[field]}: {result.get(field, 'N/A')}"
                for field in BIM_ATTRIBUTE_FIELDS
                if field != "ifc_type"
            )
            context_parts.append(f"{i}. [Similarity: {score:.4f}]\n{field_lines}")

        return "\n\n".join(context_parts)

    def search(self,
               query: str,
               top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
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
                          top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """
        Predict KBIMS part code for a BIM object.

        Args:
            bim_object_info: BIM object information (family, type, etc.)
            top_k: Number of similar objects to retrieve

        Returns:
            List of prediction candidate dictionaries, each with keys:
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
        parsed = parse_json_response(response)
        predictions = parsed.get("predictions") or [parsed]

        logger.info("Prediction complete")
        return predictions

    def batch_predict(self,
                      bim_objects: list[str],
                      top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """
        Predict part codes for multiple BIM objects.

        Args:
            bim_objects: List of BIM object information strings
            top_k: Number of similar objects to retrieve per query

        Returns:
            List of dictionaries with input and prediction result (list of candidate dicts)
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
    import json
    from src.utils import select_json_file, format_bim_object_for_prediction

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
