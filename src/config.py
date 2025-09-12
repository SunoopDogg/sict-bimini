import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_DIR = DATA_DIR / "csv"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Model configurations
OLLAMA_MODEL = "gpt-oss:latest"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Text processing settings for Korean
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KOREAN_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]

# ChromaDB settings
CHROMA_COLLECTION_NAME = "kbims_documents"
CHROMA_PERSIST_DIRECTORY = str(CHROMA_DB_DIR)

# CSV processing settings
CSV_FILE_PATTERN = "KBIMS_*.csv"
CSV_ENCODING = "utf-8"

# Embedding batch settings
EMBEDDING_BATCH_SIZE = 100

