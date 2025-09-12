

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SICT-BIMINI RAG Chat System

Interactive chat system using retrievers with gpt-oss:latest LLM model
for Korean Building Information Modeling Standard (KBIMS) documents.
"""

import sys
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    # Fallback for when langchain-ollama is not installed
    from langchain_community.llms import Ollama as OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from store import KBIMSVectorStore, create_kbims_vector_store
from config import OLLAMA_MODEL, CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KBIMSChatSystem:
    """
    Korean Building Information Modeling Standard (KBIMS) Chat System.

    Provides interactive chat interface using retrieval-augmented generation (RAG)
    with Ollama gpt-oss:latest model and ChromaDB vector store.
    """

    def __init__(self,
                 model_name: str = OLLAMA_MODEL,
                 collection_name: str = CHROMA_COLLECTION_NAME,
                 persist_directory: str = CHROMA_PERSIST_DIRECTORY,
                 top_k: int = 5,
                 temperature: float = 0.3):
        """
        Initialize KBIMS Chat System.

        Args:
            model_name: Ollama model name (default: gpt-oss:latest)
            collection_name: ChromaDB collection name
            persist_directory: ChromaDB persistence directory
            top_k: Number of documents to retrieve for context
            temperature: LLM temperature for response generation
        """
        self.model_name = model_name
        self.top_k = top_k
        self.temperature = temperature

        # Initialize components
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self.memory = None
        self.conversation_history = []

        logger.info(f"Initializing KBIMS Chat System with model: {model_name}")

        # Initialize vector store
        self._initialize_vector_store(collection_name, persist_directory)

        # Initialize LLM
        self._initialize_llm()

        # Initialize retriever
        self._initialize_retriever()

        # Initialize conversation chain
        self._initialize_conversation_chain()

    def _initialize_vector_store(self, collection_name: str, persist_directory: str) -> None:
        """Initialize ChromaDB vector store."""
        try:
            self.vector_store = create_kbims_vector_store(
                collection_name=collection_name,
                persist_directory=persist_directory,
                reset=False
            )

            # Check if vector store has documents
            stats = self.vector_store.get_collection_stats()
            doc_count = stats.get('document_count', 0)

            if doc_count == 0:
                logger.warning(
                    "Vector store is empty. Processing KBIMS documents...")
                result = self.vector_store.process_and_store_kbims_documents()
                logger.info(
                    f"Processed and stored {result['processed_documents']} documents")
            else:
                logger.info(f"Vector store loaded with {doc_count} documents")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def _initialize_llm(self) -> None:
        """Initialize Ollama LLM."""
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                base_url="http://localhost:11434"  # Default Ollama URL
            )
            logger.info(f"Initialized Ollama LLM: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise

    def _initialize_retriever(self) -> None:
        """Initialize document retriever from vector store."""
        try:
            self.retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
            logger.info(f"Initialized retriever with top_k={self.top_k}")

        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

    def _initialize_conversation_chain(self) -> None:
        """Initialize conversational retrieval chain."""
        try:
            # Initialize memory for conversation history
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Create conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )

            logger.info("Initialized conversational retrieval chain")

        except Exception as e:
            logger.error(f"Failed to initialize conversation chain: {e}")
            raise

    def chat(self, query: str) -> Dict[str, Any]:
        """
        Process user query and generate response using RAG.

        Args:
            query: User question in Korean or English

        Returns:
            Dictionary containing answer, source documents, and metadata
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Get response from conversational chain
            result = self.qa_chain.invoke({"question": query})

            # Extract answer and source documents
            answer = result["answer"]
            source_docs = result.get("source_documents", [])

            # Format source information
            sources = []
            for i, doc in enumerate(source_docs):
                source_info = {
                    "index": i + 1,
                    "category": doc.metadata.get("category", "Unknown"),
                    "classification": {
                        "major": doc.metadata.get("classification_major"),
                        "medium": doc.metadata.get("classification_medium"),
                        "minor": doc.metadata.get("classification_minor")
                    },
                    "library_name": doc.metadata.get("library_name"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)

            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "answer": answer,
                "sources_count": len(sources)
            })

            response = {
                "answer": answer,
                "sources": sources,
                "query": query,
                "retrieved_docs_count": len(source_docs)
            }

            logger.info(
                f"Generated response with {len(sources)} source documents")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"죄송합니다. 질문을 처리하는 중에 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "query": query,
                "retrieved_docs_count": 0,
                "error": str(e)
            }

    def search_by_category(self, query: str, category: str) -> Dict[str, Any]:
        """
        Search within a specific building component category.

        Args:
            query: Search query
            category: Building component category (e.g., "기초", "벽체", "지붕")

        Returns:
            Dictionary containing filtered search results
        """
        try:
            # Perform similarity search with category filter
            filter_metadata = {"category": category}
            docs = self.vector_store.similarity_search_with_scores(
                query=query,
                k=self.top_k,
                filter_metadata=filter_metadata
            )

            if not docs:
                return {
                    "answer": f"'{category}' 카테고리에서 '{query}'에 대한 관련 문서를 찾을 수 없습니다.",
                    "sources": [],
                    "query": query,
                    "category": category,
                    "retrieved_docs_count": 0
                }

            # Generate context from retrieved documents
            context = "\n\n".join([doc.page_content for doc, score in docs])

            # Create a focused prompt for category-specific search
            category_prompt = f"""다음은 '{category}' 카테고리의 한국 건축정보모델링 표준(KBIMS) 문서입니다.

컨텍스트:
{context}

질문: {query}

위 컨텍스트를 바탕으로 '{category}' 카테고리에 특화된 답변을 한국어로 제공해주세요."""

            # Get response from LLM
            answer = self.llm.invoke(category_prompt)

            # Format sources
            sources = []
            for i, (doc, score) in enumerate(docs):
                source_info = {
                    "index": i + 1,
                    "score": float(score),
                    "category": doc.metadata.get("category", "Unknown"),
                    "classification": {
                        "major": doc.metadata.get("classification_major"),
                        "medium": doc.metadata.get("classification_medium"),
                        "minor": doc.metadata.get("classification_minor")
                    },
                    "library_name": doc.metadata.get("library_name"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)

            return {
                "answer": answer,
                "sources": sources,
                "query": query,
                "category": category,
                "retrieved_docs_count": len(docs)
            }

        except Exception as e:
            logger.error(f"Error in category search: {e}")
            return {
                "answer": f"카테고리 검색 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "query": query,
                "category": category,
                "retrieved_docs_count": 0,
                "error": str(e)
            }

    def get_available_categories(self) -> List[str]:
        """Get list of available building component categories."""
        try:
            stats = self.vector_store.get_collection_stats()
            categories = list(stats.get('categories', {}).keys())
            return sorted(categories)
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history."""
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear conversation history and memory."""
        self.conversation_history.clear()
        if self.memory:
            self.memory.clear()
        logger.info("Conversation history cleared")


def print_welcome_message():
    """Print welcome message and instructions."""
    print("=" * 80)
    print("🏗️  SICT-BIMINI RAG Chat System")
    print("   Korean Building Information Modeling Standard (KBIMS) Assistant")
    print("=" * 80)
    print()
    print("💬 한국어나 영어로 건축 정보 모델링 관련 질문을 해보세요!")
    print("   (Ask questions about Korean building standards in Korean or English)")
    print()
    print("🔧 Available commands:")
    print("   - '/categories' : 사용 가능한 건축 부품 카테고리 보기")
    print("   - '/category <name>' : 특정 카테고리에서 검색")
    print("   - '/history' : 대화 기록 보기")
    print("   - '/clear' : 대화 기록 지우기")
    print("   - '/help' : 도움말 보기")
    print("   - '/quit' or '/exit' : 종료")
    print()


def print_help():
    """Print help information."""
    print("\n📖 Help - KBIMS Chat System")
    print("-" * 40)
    print("🎯 Purpose: AI assistant for Korean Building Information Modeling Standards")
    print()
    print("💡 Example questions:")
    print("   - '기초 콘크리트에 대해 알려주세요'")
    print("   - '벽체 구조 시스템은 어떤 것들이 있나요?'")
    print("   - 'What are the KBIMS wall classifications?'")
    print("   - '라이브러리 파일 형식에 대해 설명해주세요'")
    print()
    print("🔍 Search tips:")
    print("   - Use specific Korean building terms for better results")
    print("   - Try category-specific searches with '/category <name>'")
    print("   - Ask follow-up questions for detailed information")
    print()


def format_response(response: Dict[str, Any]) -> None:
    """Format and print chat response."""
    print("\n" + "🤖 " + "="*70)
    print(response["answer"])

    if response.get("sources"):
        print(f"\n📚 Sources ({len(response['sources'])} documents):")
        print("-" * 50)

        for source in response["sources"][:3]:  # Show top 3 sources
            print(f"\n📄 {source['index']}. Category: {source['category']}")

            # Show classification if available
            classification = source.get('classification', {})
            if classification.get('major'):
                print(f"   Classification: {classification['major']}")
                if classification.get('medium'):
                    print(f"   Material: {classification['medium']}")
                if classification.get('minor'):
                    print(f"   Type: {classification['minor']}")

            # Show library name if available
            if source.get('library_name'):
                print(f"   Library: {source['library_name']}")

            # Show content preview
            print(f"   Preview: {source['content_preview']}")

            # Show similarity score if available
            if 'score' in source:
                print(f"   Similarity: {source['score']:.3f}")

    print("="*75 + "\n")


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []

    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            missing_deps.append("langchain-ollama or langchain-community")

    try:
        from store import KBIMSVectorStore
    except ImportError as e:
        missing_deps.append(f"store module: {e}")

    try:
        from config import OLLAMA_MODEL
    except ImportError as e:
        missing_deps.append(f"config module: {e}")

    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 To install missing dependencies:")
        print("   uv sync  # or pip install langchain-ollama")
        print("\n🔧 Make sure Ollama is running:")
        print("   ollama serve  # Start Ollama server")
        print("   ollama pull gpt-oss:latest  # Download the model")
        return False

    print("✅ All dependencies are available")
    return True


def test_ollama_connection():
    """Test connection to Ollama server."""
    try:
        # Import the right Ollama class
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            from langchain_community.llms import Ollama as OllamaLLM

        # Test connection
        llm = OllamaLLM(model="gpt-oss:latest",
                        base_url="http://localhost:11434")
        response = llm.invoke("Hello")
        print("✅ Ollama connection successful")
        return True

    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Start Ollama server: ollama serve")
        print("   2. Pull the model: ollama pull gpt-oss:latest")
        print("   3. Check if Ollama is running: curl http://localhost:11434")
        return False


def main():
    """Main execution function."""
    print_welcome_message()

    # Check dependencies first
    if not check_dependencies():
        print("🛑 Please install missing dependencies before continuing.")
        return

    # Check Ollama connection
    print("🔌 Testing Ollama connection...")
    if not test_ollama_connection():
        print("🛑 Please ensure Ollama is running and gpt-oss:latest model is available.")
        print("   You can still use the system if Ollama becomes available later.")

        response = input("❓ Continue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            return

    try:
        # Initialize chat system
        print("🚀 Initializing KBIMS Chat System...")
        chat_system = KBIMSChatSystem()
        print("✅ System initialized successfully!\n")

        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("\n👋 감사합니다! KBIMS Chat을 종료합니다.")
                    break

                elif user_input.lower() == '/help':
                    print_help()
                    continue

                elif user_input.lower() == '/categories':
                    categories = chat_system.get_available_categories()
                    if categories:
                        print(f"\n📂 Available categories ({len(categories)}):")
                        for i, cat in enumerate(categories, 1):
                            print(f"   {i:2d}. {cat}")
                    else:
                        print("\n⚠️  No categories found.")
                    print()
                    continue

                elif user_input.lower().startswith('/category '):
                    category_name = user_input[10:].strip()
                    if not category_name:
                        print(
                            "⚠️  Please specify a category name. Usage: /category <name>")
                        continue

                    print(f"\n🔍 Searching in category: {category_name}")
                    query = input("   Enter your question: ").strip()

                    if query:
                        response = chat_system.search_by_category(
                            query, category_name)
                        format_response(response)
                    continue

                elif user_input.lower() == '/history':
                    history = chat_system.get_conversation_history()
                    if history:
                        print(
                            f"\n📜 Conversation History ({len(history)} items):")
                        print("-" * 50)
                        # Show last 5
                        for i, item in enumerate(history[-5:], 1):
                            print(f"{i}. Q: {item['query'][:50]}...")
                            print(f"   A: {item['answer'][:100]}...")
                            print(f"   Sources: {item['sources_count']}")
                            print()
                    else:
                        print("\n📜 No conversation history yet.")
                    continue

                elif user_input.lower() == '/clear':
                    chat_system.clear_conversation_history()
                    print("🗑️  Conversation history cleared.")
                    continue

                # Process regular chat query
                print("\n🤔 Thinking...")
                response = chat_system.chat(user_input)
                format_response(response)

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break

            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
                continue

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        logger.error(f"Initialization error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
