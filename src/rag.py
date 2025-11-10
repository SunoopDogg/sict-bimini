import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from bim_vector_store import BIMVectorStore, MILVUS_DB_PATH
from utils import (
    load_prompt,
    parse_json_response,
    format_prediction_result,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.3


def format_bim_object_for_prediction(obj: dict) -> str:
    """
    BIM JSON 객체를 예측용 문자열로 변환

    Args:
        obj: BIM 객체 딕셔너리 (JSON에서 로드된 형태)

    Returns:
        예측에 사용할 문자열
    """
    other = obj.get("Other", {})
    parts = []

    if obj.get("ObjectType"):
        parts.append(f"ObjectType: {obj['ObjectType']}")
    if other.get("Category"):
        parts.append(f"Category: {other['Category']}")
    if other.get("Family Name"):
        parts.append(f"Family Name: {other['Family Name']}")
    if other.get("Family"):
        parts.append(f"Family: {other['Family']}")
    if other.get("Type"):
        parts.append(f"Type: {other['Type']}")

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
            return "유사한 BIM 객체를 찾을 수 없습니다."

        context_parts = []
        for i, result in enumerate(results, 1):
            score = result.get('score', 0.0)
            context_parts.append(
                f"{i}. [유사도: {score:.4f}]\n"
                f"   - 카테고리: {result.get('category', 'N/A')}\n"
                f"   - 패밀리명: {result.get('family_name', 'N/A')}\n"
                f"   - KBIMS 코드: {result.get('kbims_code', 'N/A')}\n"
                f"   - 패밀리: {result.get('family', 'N/A')}\n"
                f"   - 타입: {result.get('type', 'N/A')}\n"
                f"   - 타입ID: {result.get('type_id', 'N/A')}"
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
                logger.info(format_prediction_result(prediction))  # For logging purposes
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


def main():
    """Main function for testing the RAG system."""
    import argparse

    parser = argparse.ArgumentParser(description="BIM RAG System CLI")
    parser.add_argument("--predict", type=str, help="BIM object info for part code prediction")
    parser.add_argument("--search", type=str, help="Search query for similar objects")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")

    args = parser.parse_args()

    # Initialize RAG system
    rag = BIMRAGSystem(ollama_model=args.model)

    try:
        if args.search:
            print(f"\nSearching for: '{args.search}'")
            print("=" * 50)
            results = rag.search(args.search, top_k=args.top_k)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.get('score', 0):.4f}")
                print(f"   Category: {result.get('category', 'N/A')}")
                print(f"   Family Name: {result.get('family_name', 'N/A')}")
                print(f"   KBIMS Code: {result.get('kbims_code', 'N/A')}")
                print(f"   Family: {result.get('family', 'N/A')}")
                print(f"   Type: {result.get('type', 'N/A')}")

        if args.predict:
            predict_path = Path(args.predict)

            if predict_path.is_file():
                # 파일 경로인 경우 - 배치 예측
                print(f"\nBatch predicting from file: '{args.predict}'")
                print("=" * 50)

                with open(predict_path, 'r', encoding='utf-8') as f:
                    objects = json.load(f)

                print(f"Loaded {len(objects)} objects from file")

                # 각 객체에서 예측용 문자열 생성
                bim_infos = [format_bim_object_for_prediction(obj) for obj in objects]

                # 배치 예측 수행
                results = rag.batch_predict(bim_infos, top_k=args.top_k)

                # 결과 출력
                print(f"\n=== Batch Prediction Results ({len(results)} objects) ===\n")
                for i, result in enumerate(results, 1):
                    print(f"[{i}/{len(results)}] Input: {result['input'][:80]}...")
                    if result.get('prediction'):
                        print(
                            f"  Predicted Code: {result['prediction'].get('predicted_code', 'N/A')}")
                        print(f"  Confidence: {result['prediction'].get('confidence', 'N/A')}")
                        print(
                            f"  Reasoning: {result['prediction'].get('reasoning', 'N/A')[:100]}...")
                    else:
                        print(f"  Error: {result.get('error', 'Unknown error')}")
                    print()

                # 결과를 파일로 저장
                output_path = predict_path.parent / f"{predict_path.stem}_predictions.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nResults saved to: {output_path}")

            else:
                # 텍스트 문자열인 경우 - 단일 예측
                print(f"\nPredicting part code for: '{args.predict}'")
                print("=" * 50)
                try:
                    prediction = rag.predict_part_code(args.predict, top_k=args.top_k)
                    print("\nPrediction Result:")
                    print(format_prediction_result(prediction))
                except ValueError as e:
                    print(f"\nError: {e}")

        # If no specific action, show help
        if not any([args.search, args.predict]):
            print("\nBIM RAG System Ready")
            print("Use --help to see available options")
            print("\nExamples:")
            print("  --predict '콘크리트 기둥 RC기둥-600x600'  # 텍스트 단일 예측")
            print("  --predict data/json/no_kbims_objects.json  # JSON 파일 배치 예측")
            print("  --search '철근콘크리트 보'")

    finally:
        rag.close()


if __name__ == "__main__":
    main()
