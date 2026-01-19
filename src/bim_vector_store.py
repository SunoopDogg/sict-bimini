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
CSV_PATH = "./data/csv/bim_attributes.csv"


@dataclass
class BIMAttribute:
    """Represents a single BIM object with its attributes."""

    ifc_type: str
    category: str
    family_name: str
    kbims_code: str
    pps_code: str
    family: str
    type: str
    type_id: str

    def to_text(self) -> str:
        """Convert attributes to text for embedding."""
        parts = [
            f"IFC Type: {self.ifc_type}",
            f"Category: {self.category}",
            f"Family Name: {self.family_name}",
            f"KBIMS Code: {self.kbims_code}",
            f"PPS Code: {self.pps_code}",
            f"Family: {self.family}",
            f"Type: {self.type}",
            f"Type ID: {self.type_id}",
        ]
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, str]:
        """Convert attributes to dictionary."""
        return {
            "ifc_type": self.ifc_type,
            "category": self.category,
            "family_name": self.family_name,
            "kbims_code": self.kbims_code,
            "pps_code": self.pps_code,
            "family": self.family,
            "type": self.type,
            "type_id": self.type_id,
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
        schema.add_field(field_name="ifc_type", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="family_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="kbims_code", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="pps_code", datatype=DataType.VARCHAR, max_length=512)
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

    def _generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
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
        attributes: List[BIMAttribute] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                attr = BIMAttribute(
                    ifc_type=row.get("ifc_type", ""),
                    category=row.get("category", ""),
                    family_name=row.get("family_name", ""),
                    kbims_code=row.get("kbims_code", ""),
                    pps_code=row.get("pps_code", ""),
                    family=row.get("family", ""),
                    type=row.get("type", ""),
                    type_id=row.get("type_id", ""),
                )
                attributes.append(attr)

        logger.info(f"Read {len(attributes)} records from CSV")

        # Generate embeddings for all texts
        texts = [attr.to_text() for attr in attributes]
        embeddings = self._generate_embeddings(texts, batch_size=batch_size)

        # Prepare data for insertion
        data = []
        for attr, embedding in zip(attributes, embeddings):
            record = attr.to_dict()
            record["vector"] = embedding
            data.append(record)

        # Insert in batches
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self.client.insert(
                collection_name=self.collection_name,
                data=batch
            )
            total_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1} ({total_inserted}/{len(data)})")

        logger.info(f"Successfully loaded {total_inserted} records")
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
                             "pps_code", "family", "type", "type_id"]

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
                "pps_code": entity.get("pps_code", ""),
                "family": entity.get("family", ""),
                "type": entity.get("type", ""),
                "type_id": entity.get("type_id", ""),
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
