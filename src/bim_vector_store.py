import csv
import logging
from pathlib import Path
from typing import Any

import torch
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

from src.utils import BIM_ATTRIBUTE_FIELDS, BIMAttribute, bim_attribute_from_csv_row

logger = logging.getLogger(__name__)

# Constants
MILVUS_DB_PATH = "./milvus_data/milvus.db"
COLLECTION_NAME = "bim_objects"
EMBEDDING_MODEL = "google/embeddinggemma-300m"
CSV_PATH = "./data/csv/bim_attributes.csv"


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
        for field in BIM_ATTRIBUTE_FIELDS:
            max_length = 1024 if field in ("family", "type", "type_id") else 512
            schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=max_length)
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

    def _generate_embeddings(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to embed at once

        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch)
            all_embeddings.extend(embeddings.tolist())
            logger.info(f"Generated embeddings for batch {i // batch_size + 1}")
        return all_embeddings

    def load_from_csv(self, csv_path: str = CSV_PATH, batch_size: int = 100) -> int:
        """
        Load BIM attributes from CSV file and insert into vector store.

        Args:
            csv_path: Path to CSV file
            batch_size: Number of records to insert at once

        Returns:
            Number of records inserted
        """
        logger.info(f"Loading data from {csv_path}")

        # Read CSV file
        attributes: list[BIMAttribute] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                attr = bim_attribute_from_csv_row(row)
                attributes.append(attr)

        logger.info(f"Read {len(attributes)} records from CSV")

        # Generate embeddings for all texts
        embedding_texts = [attr.to_text() for attr in attributes]
        embeddings = self._generate_embeddings(embedding_texts, batch_size=batch_size)

        # Prepare records for insertion
        records = []
        for attr, embedding in zip(attributes, embeddings):
            record = attr.to_dict()
            record["vector"] = embedding
            records.append(record)

        # Insert in batches
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.client.insert(
                collection_name=self.collection_name,
                data=batch
            )
            total_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1} ({total_inserted}/{len(records)})")

        logger.info(f"Successfully loaded {total_inserted} records")
        return total_inserted

    def search(self,
               query: str,
               limit: int = 5,
               output_fields: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Search for similar BIM objects.

        Args:
            query: Search query text
            limit: Maximum number of results
            output_fields: Fields to return (default: all BIM attribute fields)

        Returns:
            List of search results with scores
        """
        if output_fields is None:
            output_fields = list(BIM_ATTRIBUTE_FIELDS)

        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]

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
            entity = hit.get("entity", {})
            result = {
                "id": hit.get("id"),
                "score": hit.get("distance", 0.0),
                **{field: entity.get(field, "") for field in BIM_ATTRIBUTE_FIELDS},
            }
            formatted_results.append(result)

        return formatted_results

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


if __name__ == "__main__":
    store = BIMVectorStore()
    try:
        store.reset()
        store.load_from_csv()
    finally:
        store.close()
