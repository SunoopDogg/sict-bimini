import numpy as np

from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import tiktoken

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, KOREAN_SEPARATORS,
    OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY, EMBEDDING_BATCH_SIZE
)
from loader import KBIMSCSVLoader, load_kbims_documents


class KBIMSEmbeddingProcessor:
    """
    Specialized embedding processor for KBIMS (Korea Building Information Modeling Standard) documents.
    Handles Korean text splitting, embedding generation, and preprocessing for RAG systems.
    """

    def __init__(self,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 separators: List[str] = None,
                 embedding_model: str = OPENAI_EMBEDDING_MODEL,
                 api_key: str = None,
                 batch_size: int = EMBEDDING_BATCH_SIZE):
        """
        Initialize KBIMS Embedding Processor.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            separators: Korean-optimized text separators
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key
            batch_size: Batch size for embedding generation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or KOREAN_SEPARATORS
        self.batch_size = batch_size

        # Initialize text splitter with Korean separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            add_start_index=True
        )

        # Initialize embeddings
        self.embedding_model_name = embedding_model
        if api_key or OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key or OPENAI_API_KEY
            )
        else:
            print("⚠️  No OpenAI API key found. Embeddings will not be available.")
            self.embeddings = None

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split KBIMS documents using Korean-aware text splitting.

        Args:
            documents: List of KBIMS documents from loader

        Returns:
            List of split documents with enriched metadata
        """
        if not documents:
            return []

        print(f"🔪 Splitting {len(documents)} KBIMS documents...")

        all_splits = []

        for doc in tqdm(documents, desc="Splitting documents"):
            # Split the document
            splits = self.text_splitter.split_documents([doc])

            # Enrich metadata for each split
            for i, split in enumerate(splits):
                # Preserve original metadata
                enriched_metadata = split.metadata.copy()

                # Add chunk-specific metadata
                enriched_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(splits),
                    'chunk_size': len(split.page_content),
                    'chunk_tokens': self._count_tokens(split.page_content),
                    'chunk_type': self._determine_chunk_type(split.page_content),
                    'original_doc_length': len(doc.page_content)
                })

                # Create new document with enriched metadata
                enriched_split = Document(
                    page_content=split.page_content,
                    metadata=enriched_metadata
                )

                all_splits.append(enriched_split)

        print(f"✅ Created {len(all_splits)} document chunks")
        return all_splits

    def _determine_chunk_type(self, content: str) -> str:
        """
        Determine the type of content in a chunk based on Korean KBIMS patterns.

        Args:
            content: Chunk content

        Returns:
            Chunk type classification
        """
        content_lower = content.lower()

        # Check for classification hierarchy
        if "분류-대-공정" in content or "분류-중-재료" in content or "분류-소-객체유형" in content:
            return 'classification'

        # Check for library information
        elif "라이브러리" in content or 'library' in content_lower:
            return 'library'

        # Check for standard codes
        elif "조달청" in content or "표준공사코드" in content:
            return 'standard_code'

        # Check for KBIMS classification
        elif 'kbims' in content_lower or "부위분류" in content:
            return 'kbims_classification'

        # Check for notes/remarks
        elif "비고" in content:
            return 'remarks'

        # General content
        else:
            return 'general'

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Input text

        Returns:
            Token count
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to character count / 4 (rough approximation)
            return len(text) // 4

    def create_embeddings(self, documents: List[Document]) -> Tuple[List[Document], np.ndarray]:
        """
        Generate embeddings for split documents.

        Args:
            documents: List of split documents

        Returns:
            Tuple of (documents, embeddings_matrix)
        """
        if not self.embeddings:
            raise ValueError(
                "OpenAI embeddings not initialized. Check API key.")

        if not documents:
            return documents, np.array([])

        print(
            f"🔗 Generating embeddings for {len(documents)} document chunks...")

        # Extract texts for embedding
        texts = [doc.page_content for doc in documents]

        # Generate embeddings in batches
        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + self.batch_size]

            try:
                # Generate embeddings for batch
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                print(
                    f"⚠️  Error generating embeddings for batch {i//self.batch_size + 1}: {e}")
                # Add zero embeddings for failed batch
                embedding_dim = 1536  # text-embedding-3-small dimension
                zero_embeddings = [[0.0] * embedding_dim] * len(batch_texts)
                all_embeddings.extend(zero_embeddings)

        # Convert to numpy array
        embeddings_matrix = np.array(all_embeddings)

        print(f"✅ Generated embeddings with shape: {embeddings_matrix.shape}")

        return documents, embeddings_matrix

    def process_documents_for_rag(self, documents: List[Document] = None) -> Tuple[List[Document], np.ndarray]:
        """
        Complete preprocessing pipeline for RAG: split documents and create embeddings.

        Args:
            documents: Documents to process (if None, loads from KBIMS CSV files)

        Returns:
            Tuple of (processed_documents, embeddings_matrix)
        """
        # Load documents if not provided
        if documents is None:
            print("📚 Loading KBIMS documents...")
            documents = load_kbims_documents()

        print(f"🚀 Starting RAG preprocessing for {len(documents)} documents")

        # Step 1: Split documents
        split_documents = self.split_documents(documents)

        # Step 2: Generate embeddings
        processed_documents, embeddings = self.create_embeddings(
            split_documents)

        print(f"✅ RAG preprocessing complete!")
        print(f"   - Original documents: {len(documents)}")
        print(f"   - Split chunks: {len(processed_documents)}")
        print(f"   - Embeddings shape: {embeddings.shape}")

        return processed_documents, embeddings

    def get_embedding_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about document chunks and their properties.

        Args:
            documents: List of processed documents

        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {}

        # Collect statistics
        chunk_sizes = [len(doc.page_content) for doc in documents]
        chunk_tokens = [doc.metadata.get('chunk_tokens', 0)
                        for doc in documents]
        chunk_types = [doc.metadata.get(
            'chunk_type', 'unknown') for doc in documents]
        categories = [doc.metadata.get('category', 'unknown')
                      for doc in documents]

        # Calculate statistics
        stats = {
            'total_chunks': len(documents),
            'average_chunk_size': np.mean(chunk_sizes),
            'median_chunk_size': np.median(chunk_sizes),
            'min_chunk_size': np.min(chunk_sizes),
            'max_chunk_size': np.max(chunk_sizes),
            'average_tokens': np.mean(chunk_tokens),
            'total_tokens': np.sum(chunk_tokens),
            'chunk_type_distribution': {chunk_type: chunk_types.count(chunk_type) for chunk_type in set(chunk_types)},
            'category_distribution': {category: categories.count(category) for category in set(categories)},
            'chunks_per_category': {category: categories.count(category) for category in set(categories)}
        }

        return stats

    def print_sample_chunks(self, documents: List[Document], n_samples: int = 3):
        """
        Print sample chunks for inspection.

        Args:
            documents: List of processed documents
            n_samples: Number of samples to print
        """
        if not documents:
            print("No documents to sample")
            return

        print(
            f"\n📄 Sample Document Chunks ({min(n_samples, len(documents))} samples):")
        print("=" * 80)

        for i, doc in enumerate(documents[:n_samples]):
            print(f"\n🔹 Chunk {i+1}:")
            print(f"Category: {doc.metadata.get('category', 'Unknown')}")
            print(f"Type: {doc.metadata.get('chunk_type', 'Unknown')}")
            print(
                f"Size: {doc.metadata.get('chunk_size', 0)} chars, {doc.metadata.get('chunk_tokens', 0)} tokens")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata keys: {list(doc.metadata.keys())}")
            print("-" * 40)


def process_kbims_for_rag(chunk_size: int = None, chunk_overlap: int = None) -> Tuple[List[Document], np.ndarray]:
    """
    Convenience function to process all KBIMS documents for RAG.

    Args:
        chunk_size: Override default chunk size
        chunk_overlap: Override default chunk overlap

    Returns:
        Tuple of (processed_documents, embeddings_matrix)
    """
    # Initialize processor with optional overrides
    kwargs = {}
    if chunk_size is not None:
        kwargs['chunk_size'] = chunk_size
    if chunk_overlap is not None:
        kwargs['chunk_overlap'] = chunk_overlap

    processor = KBIMSEmbeddingProcessor(**kwargs)
    return processor.process_documents_for_rag()


if __name__ == "__main__":
    # Test the embedding processor
    print("🚀 Testing KBIMS Embedding Processor")
    print("=" * 60)

    try:
        # Initialize processor
        processor = KBIMSEmbeddingProcessor()

        # Load a small sample for testing
        loader = KBIMSCSVLoader()
        sample_docs = loader.get_documents_by_category(
            "기초")[:5]  # Just 5 foundation docs for testing

        print(f"📚 Testing with {len(sample_docs)} sample documents")

        # Test splitting
        split_docs = processor.split_documents(sample_docs)
        print(f"🔪 Created {len(split_docs)} chunks")

        # Show sample chunks
        processor.print_sample_chunks(split_docs, n_samples=2)

        # Show statistics
        stats = processor.get_embedding_stats(split_docs)
        print(f"\n📊 Chunk Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                # Show first 5 items
                print(f"  {key}: {dict(list(value.items())[:5])}")
            else:
                print(f"  {key}: {value}")

        # Test embeddings (only if API key available)
        if processor.embeddings:
            print(f"\n🔗 Testing embedding generation...")
            _, embeddings = processor.create_embeddings(
                split_docs[:2])  # Test with just 2 chunks
            print(f"✅ Successfully generated embeddings: {embeddings.shape}")
        else:
            print(f"\n⚠️  Skipping embedding test (no API key)")

        print(f"\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
