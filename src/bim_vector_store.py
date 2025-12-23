import csv
import torch
import logging

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pymilvus import (
    MilvusClient,
    DataType,
)
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MILVUS_DB_PATH = "./milvus_data/milvus.db"
COLLECTION_NAME = "bim_objects"
EMBEDDING_MODEL = "google/embeddinggemma-300m"
VECTOR_DIM = 768
CSV_PATH = "./data/csv/bim_attributes.csv"


@dataclass
class BIMAttribute:
    """Data class for BIM object attributes."""
    ifc_type: str
    category: str
    family_name: str
    kbims_code: str
    family: str
    type: str
    type_id: str

    def to_text(self) -> str:
        """Convert attributes to text for embedding."""
        parts = [
            self.ifc_type,
            self.category,
            self.family_name,
            self.kbims_code,
            self.family,
            self.type,
            self.type_id
        ]
        return " ".join(p for p in parts if p)

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "ifc_type": self.ifc_type,
            "category": self.category,
            "family_name": self.family_name,
            "kbims_code": self.kbims_code,
            "family": self.family,
            "type": self.type,
            "type_id": self.type_id
        }


class BIMVectorStore:
    """
    Vector store for BIM object attributes using Milvus-lite.

    Handles embedding generation, storage, and similarity search.
    """

    def __init__(self,
                 db_path: str = MILVUS_DB_PATH,
                 collection_name: str = COLLECTION_NAME,
                 embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize BIM Vector Store.

        Args:
            db_path: Path to Milvus-lite database file
            collection_name: Name of the collection
            embedding_model: SentenceTransformer model name
        """
        self.db_path = db_path
        self.collection_name = collection_name

        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize Milvus client
        logger.info(f"Connecting to Milvus-lite at {db_path}")
        self.client = MilvusClient(db_path)

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(embedding_model).to(device=device)

        # Get actual embedding dimension from model
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.vector_dim}")

        # Create collection if not exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection with schema if it doesn't exist."""
        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            return

        logger.info(f"Creating collection '{self.collection_name}'")

        # Define collection schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=False,
        )

        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="ifc_type", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="family_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="kbims_code", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="family", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="type_id", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_dim)

        # Create index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 128}
        )

        # Create collection with schema and index
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"Collection '{self.collection_name}' created successfully")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text).tolist()

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts).tolist()

    def load_from_csv(self, csv_path: str = CSV_PATH, batch_size: int = 100) -> int:
        """
        Load BIM attributes from CSV and store in Milvus.

        Args:
            csv_path: Path to CSV file
            batch_size: Batch size for insertion

        Returns:
            Number of records inserted
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading data from {csv_path}")

        # Read CSV
        attributes = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from keys and filter out None keys (from trailing commas)
                row = {k.strip(): v for k, v in row.items() if k is not None}
                attr = BIMAttribute(
                    ifc_type=row.get('ifc_type', ''),
                    category=row.get('category', ''),
                    family_name=row.get('family_name', ''),
                    kbims_code=row.get('kbims_code', ''),
                    family=row.get('family', ''),
                    type=row.get('type', ''),
                    type_id=row.get('type_id', '')
                )
                attributes.append(attr)

        logger.info(f"Loaded {len(attributes)} records from CSV")

        # Generate embeddings and insert in batches
        total_inserted = 0

        for i in range(0, len(attributes), batch_size):
            batch = attributes[i:i + batch_size]

            # Generate embeddings
            texts = [attr.to_text() for attr in batch]
            embeddings = self._generate_embeddings(texts)

            # Prepare data for insertion
            data = []
            for attr, embedding in zip(batch, embeddings):
                record = attr.to_dict()
                record["vector"] = embedding
                data.append(record)

            # Insert batch
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )

            total_inserted += len(batch)
            logger.info(f"Inserted {total_inserted}/{len(attributes)} records")

        logger.info(f"Successfully inserted {total_inserted} records")
        return total_inserted

    def search(self,
               query: str,
               limit: int = 5,
               output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar BIM objects.

        Args:
            query: Search query text
            limit: Maximum number of results
            output_fields: Fields to return (default: all)

        Returns:
            List of search results with scores
        """
        if output_fields is None:
            output_fields = ["ifc_type", "category", "family_name", "kbims_code",
                             "family", "type", "type_id"]

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=output_fields
        )

        # Format results
        formatted_results = []
        for hit in results[0]:
            # MilvusClient nests entity data under "entity" key
            entity = hit.get("entity", {})
            result = {
                "id": hit.get("id"),
                "score": hit.get("distance", 0.0),
                "ifc_type": entity.get("ifc_type", ""),
                "category": entity.get("category", ""),
                "family_name": entity.get("family_name", ""),
                "kbims_code": entity.get("kbims_code", ""),
                "family": entity.get("family", ""),
                "type": entity.get("type", ""),
                "type_id": entity.get("type_id", ""),
            }
            formatted_results.append(result)

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        stats = {
            "collection_name": self.collection_name,
            "db_path": self.db_path,
            "vector_dim": self.vector_dim,
            "embedding_model": EMBEDDING_MODEL,
        }

        # Get collection info
        if self.client.has_collection(self.collection_name):
            collection_stats = self.client.get_collection_stats(self.collection_name)
            stats["row_count"] = collection_stats.get("row_count", 0)
        else:
            stats["row_count"] = 0

        return stats

    def reset(self) -> None:
        """Drop and recreate the collection."""
        if self.client.has_collection(self.collection_name):
            logger.info(f"Dropping collection '{self.collection_name}'")
            self.client.drop_collection(self.collection_name)

        self._ensure_collection()
        logger.info("Collection reset successfully")

    def close(self) -> None:
        """Close the Milvus client connection."""
        self.client.close()
        logger.info("Milvus client closed")


def main():
    """Main function for testing the vector store."""
    import argparse

    parser = argparse.ArgumentParser(description="BIM Vector Store CLI")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV file")
    parser.add_argument("--reset", action="store_true", help="Reset collection before loading")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Search result limit")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")

    args = parser.parse_args()

    # Initialize store
    store = BIMVectorStore()

    try:
        if args.reset:
            store.reset()

        if args.stats:
            stats = store.get_stats()
            print("\nCollection Statistics:")
            print("=" * 50)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print("=" * 50)

        # Load data if CSV provided and collection is empty
        stats = store.get_stats()
        if stats.get("row_count", 0) == 0:
            print(f"\nLoading data from {args.csv}...")
            count = store.load_from_csv(args.csv)
            print(f"Loaded {count} records")

        # Search if query provided
        if args.search:
            print(f"\nSearching for: '{args.search}'")
            print("=" * 50)
            results = store.search(args.search, limit=args.limit)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.get('score', 0):.4f}")
                print(f"   Category: {result.get('category', 'N/A')}")
                print(f"   Family Name: {result.get('family_name', 'N/A')}")
                print(f"   KBIMS Code: {result.get('kbims_code', 'N/A')}")
                print(f"   Family: {result.get('family', 'N/A')}")
                print(f"   Type: {result.get('type', 'N/A')}")

        # Show final stats
        if not args.stats and not args.search:
            stats = store.get_stats()
            print("\nVector Store Ready:")
            print(f"  Collection: {stats['collection_name']}")
            print(f"  Records: {stats['row_count']}")
            print(f"  Dimension: {stats['vector_dim']}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
