# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import numpy as np
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import (
    CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY,
    OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY, EMBEDDING_BATCH_SIZE
)
from embedding import KBIMSEmbeddingProcessor


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KBIMSVectorStore:
    """
    Vector database store for KBIMS (Korea Building Information Modeling Standard) documents.

    Provides persistent vector storage using ChromaDB with Korean language optimization
    and integration with the existing KBIMSEmbeddingProcessor pipeline.
    """

    def __init__(self,
                 collection_name: str = CHROMA_COLLECTION_NAME,
                 persist_directory: str = CHROMA_PERSIST_DIRECTORY,
                 embedding_model: str = OPENAI_EMBEDDING_MODEL,
                 api_key: str = None,
                 embedding_processor: KBIMSEmbeddingProcessor = None):
        """
        Initialize KBIMS Vector Store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key
            embedding_processor: Pre-configured embedding processor (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_model_name = embedding_model

        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for vector store operations")

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key
        )

        # Initialize or use provided embedding processor
        if embedding_processor:
            self.embedding_processor = embedding_processor
        else:
            self.embedding_processor = KBIMSEmbeddingProcessor(
                embedding_model=embedding_model,
                api_key=api_key
            )

        # Initialize vector store (will be set in initialize_collection)
        self.vector_store = None
        self._is_initialized = False

        logger.info(
            f"Initialized KBIMSVectorStore with collection: {collection_name}")

    def initialize_collection(self, reset: bool = False) -> None:
        """
        Initialize or load the ChromaDB collection.

        Args:
            reset: If True, delete existing collection and create new one
        """
        try:
            if reset and self.vector_store:
                logger.info(f"Resetting collection: {self.collection_name}")
                self.vector_store.delete_collection()
                self.vector_store = None

            # Initialize ChromaDB vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )

            self._is_initialized = True

            # Log collection info
            try:
                collection_count = self.vector_store._collection.count()
                logger.info(
                    f"Collection '{self.collection_name}' initialized with {collection_count} documents")
            except Exception as e:
                logger.info(
                    f"Collection '{self.collection_name}' initialized (count unavailable: {e})")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _ensure_initialized(self):
        """Ensure the vector store is initialized."""
        if not self._is_initialized or not self.vector_store:
            self.initialize_collection()

    def add_documents(self,
                      documents: List[Document],
                      batch_size: int = EMBEDDING_BATCH_SIZE,
                      show_progress: bool = True) -> List[str]:
        """
        Add documents to the vector store with batch processing.

        Args:
            documents: List of documents to add
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of document IDs
        """
        self._ensure_initialized()

        if not documents:
            logger.warning("No documents provided to add")
            return []

        logger.info(f"Adding {len(documents)} documents to vector store")

        all_ids = []

        # Process in batches to manage memory and API limits
        batches = [documents[i:i + batch_size]
                   for i in range(0, len(documents), batch_size)]

        for batch_idx, batch in enumerate(tqdm(batches, desc="Adding document batches", disable=not show_progress)):
            try:
                # Add batch to vector store
                batch_ids = self.vector_store.add_documents(
                    documents=batch,
                    ids=[f"doc_{batch_idx}_{i}" for i in range(len(batch))]
                )
                all_ids.extend(batch_ids)

            except Exception as e:
                logger.error(f"Failed to add batch {batch_idx + 1}: {e}")
                # Continue with next batch
                continue

        logger.info(
            f"Successfully added {len(all_ids)} documents to vector store")
        return all_ids

    def add_documents_from_embeddings(self,
                                      documents: List[Document],
                                      embeddings: np.ndarray,
                                      batch_size: int = EMBEDDING_BATCH_SIZE) -> List[str]:
        """
        Add documents with pre-computed embeddings.

        Args:
            documents: List of documents
            embeddings: Pre-computed embeddings matrix
            batch_size: Batch size for processing

        Returns:
            List of document IDs
        """
        self._ensure_initialized()

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents count ({len(documents)}) doesn't match embeddings count ({len(embeddings)})")

        logger.info(
            f"Adding {len(documents)} documents with pre-computed embeddings")

        all_ids = []

        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents with embeddings"):
            batch_end = min(i + batch_size, len(documents))
            batch_docs = documents[i:batch_end]
            batch_embeddings = embeddings[i:batch_end].tolist()
            batch_ids = [f"doc_{i}_{j}" for j in range(len(batch_docs))]

            try:
                # Add to ChromaDB with pre-computed embeddings
                self.vector_store._collection.add(
                    embeddings=batch_embeddings,
                    documents=[doc.page_content for doc in batch_docs],
                    metadatas=[doc.metadata for doc in batch_docs],
                    ids=batch_ids
                )
                all_ids.extend(batch_ids)

            except Exception as e:
                logger.error(f"Failed to add batch starting at index {i}: {e}")
                continue

        logger.info(
            f"Successfully added {len(all_ids)} documents with embeddings")
        return all_ids

    def similarity_search(self,
                          query: str,
                          k: int = 5,
                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of similar documents
        """
        self._ensure_initialized()

        try:
            # Perform similarity search
            if filter_metadata:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )

            logger.info(
                f"Found {len(results)} similar documents for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_scores(self,
                                      query: str,
                                      k: int = 5,
                                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with similarity scores.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of (document, score) tuples
        """
        self._ensure_initialized()

        try:
            if filter_metadata:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )

            logger.info(
                f"Found {len(results)} similar documents with scores for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Similarity search with scores failed: {e}")
            return []

    def search_by_metadata(self,
                           metadata_filter: Dict[str, Any],
                           k: Optional[int] = None) -> List[Document]:
        """
        Search documents by metadata filters only.

        Args:
            metadata_filter: Metadata conditions to filter by
            k: Maximum number of results (None for all)

        Returns:
            List of matching documents
        """
        self._ensure_initialized()

        try:
            # Use similarity search with empty query to get metadata-filtered results
            results = self.vector_store.similarity_search(
                query="",
                k=k if k else 100,  # Default limit if not specified
                filter=metadata_filter
            )

            logger.info(
                f"Found {len(results)} documents matching metadata filter")
            return results

        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        self._ensure_initialized()

        try:
            collection = self.vector_store._collection
            count = collection.count()

            # Try to get sample documents to analyze metadata
            sample_size = min(100, count) if count > 0 else 0
            sample_data = collection.get(
                limit=sample_size) if sample_size > 0 else None

            stats = {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': str(self.persist_directory),
                'embedding_model': self.embedding_model_name
            }

            # Analyze metadata if we have sample data
            if sample_data and sample_data.get('metadatas'):
                metadatas = sample_data['metadatas']

                # Count categories
                categories = [meta.get('category', 'unknown')
                              for meta in metadatas if meta]
                category_counts = {cat: categories.count(
                    cat) for cat in set(categories)}

                # Count chunk types
                chunk_types = [meta.get('chunk_type', 'unknown')
                               for meta in metadatas if meta]
                chunk_type_counts = {chunk_type: chunk_types.count(
                    chunk_type) for chunk_type in set(chunk_types)}

                stats.update({
                    'categories': category_counts,
                    'chunk_types': chunk_type_counts,
                    'sample_size': len(metadatas)
                })

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }

    def reset_collection(self, confirm: bool = False) -> bool:
        """
        Reset the collection by deleting all documents.

        Args:
            confirm: Must be True to confirm deletion

        Returns:
            True if successful, False otherwise
        """
        if not confirm:
            logger.warning(
                "Collection reset requires confirmation (confirm=True)")
            return False

        try:
            self._ensure_initialized()

            # Get current count for logging
            current_count = self.vector_store._collection.count()

            # Delete the collection
            self.vector_store.delete_collection()

            # Reinitialize
            self.initialize_collection()

            logger.info(
                f"Successfully reset collection. Deleted {current_count} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        self._ensure_initialized()

        try:
            self.vector_store._collection.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    @classmethod
    def from_documents(cls,
                       documents: List[Document],
                       **kwargs) -> 'KBIMSVectorStore':
        """
        Create a KBIMSVectorStore from a list of documents.

        Args:
            documents: List of documents to add
            **kwargs: Additional arguments for KBIMSVectorStore initialization

        Returns:
            Initialized KBIMSVectorStore with documents added
        """
        # Create store instance
        store = cls(**kwargs)

        # Initialize collection
        store.initialize_collection()

        # Add documents
        store.add_documents(documents)

        return store

    @classmethod
    def from_existing_embeddings(cls,
                                 documents: List[Document],
                                 embeddings: np.ndarray,
                                 **kwargs) -> 'KBIMSVectorStore':
        """
        Create a KBIMSVectorStore from documents with pre-computed embeddings.

        Args:
            documents: List of documents
            embeddings: Pre-computed embeddings matrix
            **kwargs: Additional arguments for KBIMSVectorStore initialization

        Returns:
            Initialized KBIMSVectorStore with documents and embeddings added
        """
        # Create store instance
        store = cls(**kwargs)

        # Initialize collection
        store.initialize_collection()

        # Add documents with embeddings
        store.add_documents_from_embeddings(documents, embeddings)

        return store

    def process_and_store_kbims_documents(self,
                                          documents: Optional[List[Document]] = None,
                                          reset_collection: bool = False) -> Dict[str, Any]:
        """
        Complete pipeline: process KBIMS documents and store in vector database.

        Args:
            documents: Documents to process (if None, loads from KBIMS CSV files)
            reset_collection: Reset collection before adding documents

        Returns:
            Dictionary with processing results
        """
        logger.info(
            "Starting complete KBIMS document processing and storage pipeline")

        # Reset collection if requested
        if reset_collection:
            self.reset_collection(confirm=True)
        else:
            self.initialize_collection()

        # Process documents using embedding processor
        processed_documents, embeddings = self.embedding_processor.process_documents_for_rag(
            documents)

        # Add to vector store
        doc_ids = self.add_documents_from_embeddings(
            processed_documents, embeddings)

        # Get final stats
        stats = self.get_collection_stats()

        result = {
            'processed_documents': len(processed_documents),
            'embeddings_shape': embeddings.shape,
            'added_documents': len(doc_ids),
            'collection_stats': stats
        }

        logger.info(f"Pipeline complete: {result}")
        return result


def create_kbims_vector_store(collection_name: str = CHROMA_COLLECTION_NAME,
                              persist_directory: str = CHROMA_PERSIST_DIRECTORY,
                              reset: bool = False) -> KBIMSVectorStore:
    """
    Convenience function to create and initialize a KBIMS vector store.

    Args:
        collection_name: ChromaDB collection name
        persist_directory: Persistence directory
        reset: Reset existing collection

    Returns:
        Initialized KBIMSVectorStore
    """
    store = KBIMSVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    store.initialize_collection(reset=reset)
    return store


if __name__ == "__main__":
    # Test the vector store
    print("Testing KBIMS Vector Store")
    print("=" * 60)

    try:
        # Create store
        store = create_kbims_vector_store(reset=False)

        # Get initial stats
        stats = store.get_collection_stats()
        print("Initial Collection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # If collection is empty, process some sample documents
        if stats.get('document_count', 0) == 0:
            print("\nCollection is empty. Processing sample KBIMS documents...")
            result = store.process_and_store_kbims_documents(
                reset_collection=True)
            print("Processing complete:", result)

        # Test search
        print("\nTesting similarity search...")
        test_queries = [
            "기초 콘크리트",  # Foundation concrete
            "벽체 구조",      # Wall structure
            "라이브러리"      # Library
        ]

        for query in test_queries:
            results = store.similarity_search(query, k=3)
            print(f"\nQuery: '{query}' - Found {len(results)} results:")
            for i, doc in enumerate(results[:2]):  # Show first 2
                print(f"  {i+1}. Category: {doc.metadata.get('category', 'N/A')}")
                print(f"     Type: {doc.metadata.get('chunk_type', 'N/A')}")
                print(f"     Content: {doc.page_content[:100]}...")

        # Final stats
        final_stats = store.get_collection_stats()
        print("\nFinal Collection Stats:")
        print(f"  Document count: {final_stats.get('document_count', 0)}")
        print(f"  Categories: {len(final_stats.get('categories', {}))}")

        print("\nVector store testing completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


