import logging
from typing import List, Dict, Any, Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from bim_vector_store import BIMVectorStore, MILVUS_DB_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.3

# Korean prompt template for KBIMS part code prediction
KBIMS_PREDICTION_PROMPT = """당신은 건축 BIM 객체 분류 전문가입니다. KBIMS(한국 BIM 표준) 부위코드를 예측하는 것이 당신의 역할입니다.

아래는 입력된 BIM 객체와 유사한 기존 BIM 객체들입니다:

[검색된 유사 BIM 객체]
{context}

[입력 BIM 객체 정보]
{query}

위의 유사 객체들을 참고하여 입력된 BIM 객체에 가장 적합한 KBIMS 부위코드를 예측하세요.

응답 형식:
- 예측 부위코드: [코드]
- 근거: [유사 객체와의 비교를 통한 근거 설명]
- 신뢰도: [높음/중간/낮음]

예측 결과:"""

# Chat prompt template for general Q&A
CHAT_PROMPT = """당신은 건축 BIM(Building Information Modeling) 전문가입니다.
KBIMS(한국 BIM 표준) 분류 체계에 대한 질문에 답변합니다.

아래는 질문과 관련된 BIM 객체 정보입니다:

[관련 BIM 객체]
{context}

[질문]
{question}

위 정보를 바탕으로 질문에 상세하게 답변하세요.

답변:"""


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

        # Initialize prompt templates
        self.prediction_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=KBIMS_PREDICTION_PROMPT
        )

        self.chat_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=CHAT_PROMPT
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
                          top_k: int = DEFAULT_TOP_K) -> str:
        """
        Predict KBIMS part code for a BIM object.

        Args:
            bim_object_info: BIM object information (family, type, etc.)
            top_k: Number of similar objects to retrieve

        Returns:
            LLM response with predicted part code and reasoning
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

        logger.info("Prediction complete")
        return response

    def chat(self,
             question: str,
             top_k: int = DEFAULT_TOP_K) -> str:
        """
        Answer a question about BIM objects using RAG.

        Args:
            question: User question
            top_k: Number of relevant objects to retrieve

        Returns:
            LLM response to the question
        """
        logger.info(f"Processing question: '{question}'")

        # Step 1: Retrieve relevant BIM objects
        search_results = self.search(question, top_k=top_k)

        # Step 2: Format context
        context = self._format_search_results(search_results)

        # Step 3: Generate answer using LLM
        prompt = self.chat_prompt.format(
            context=context,
            question=question
        )

        logger.info("Generating answer with LLM...")
        response = self.llm.invoke(prompt)

        logger.info("Answer generation complete")
        return response

    def batch_predict(self,
                      bim_objects: List[str],
                      top_k: int = DEFAULT_TOP_K) -> List[Dict[str, str]]:
        """
        Predict part codes for multiple BIM objects.

        Args:
            bim_objects: List of BIM object information strings
            top_k: Number of similar objects to retrieve per query

        Returns:
            List of dictionaries with input and prediction
        """
        results = []
        total = len(bim_objects)

        for i, bim_info in enumerate(bim_objects, 1):
            logger.info(f"Processing {i}/{total}: {bim_info[:50]}...")

            prediction = self.predict_part_code(bim_info, top_k=top_k)

            results.append({
                "input": bim_info,
                "prediction": prediction
            })

        logger.info(f"Batch prediction complete: {total} objects processed")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with system stats
        """
        vector_stats = self.vector_store.get_stats()

        return {
            "vector_store": vector_stats,
            "llm_model": DEFAULT_OLLAMA_MODEL,
            "ollama_url": DEFAULT_OLLAMA_URL,
            "temperature": DEFAULT_TEMPERATURE
        }

    def close(self) -> None:
        """Close connections and cleanup resources."""
        self.vector_store.close()
        logger.info("BIM RAG System closed")


def main():
    """Main function for testing the RAG system."""
    import argparse

    parser = argparse.ArgumentParser(description="BIM RAG System CLI")
    parser.add_argument("--predict", type=str, help="BIM object info for part code prediction")
    parser.add_argument("--chat", type=str, help="Question for chat mode")
    parser.add_argument("--search", type=str, help="Search query for similar objects")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--stats", action="store_true", help="Show system stats")

    args = parser.parse_args()

    # Initialize RAG system
    rag = BIMRAGSystem(ollama_model=args.model)

    try:
        if args.stats:
            stats = rag.get_stats()
            print("\nSystem Statistics:")
            print("=" * 50)
            print(f"Vector Store:")
            for key, value in stats["vector_store"].items():
                print(f"  {key}: {value}")
            print(f"\nLLM Model: {stats['llm_model']}")
            print(f"Ollama URL: {stats['ollama_url']}")
            print(f"Temperature: {stats['temperature']}")
            print("=" * 50)

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
            print(f"\nPredicting part code for: '{args.predict}'")
            print("=" * 50)
            prediction = rag.predict_part_code(args.predict, top_k=args.top_k)
            print("\nPrediction Result:")
            print(prediction)

        if args.chat:
            print(f"\nQuestion: '{args.chat}'")
            print("=" * 50)
            answer = rag.chat(args.chat, top_k=args.top_k)
            print("\nAnswer:")
            print(answer)

        # If no specific action, show help
        if not any([args.stats, args.search, args.predict, args.chat]):
            print("\nBIM RAG System Ready")
            print("Use --help to see available options")
            print("\nExamples:")
            print("  --predict '콘크리트 기둥 RC기둥-600x600'")
            print("  --chat '기초의 종류에는 무엇이 있나요?'")
            print("  --search '철근콘크리트 보'")
            print("  --stats")

    finally:
        rag.close()


if __name__ == "__main__":
    main()
