from pathlib import Path
from dotenv import load_dotenv
import os

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"

METADATA_DIR = BASE_DIR / "metadata"

CHROMA_DB_DIR = BASE_DIR / "chroma_db"

PROMPTS_DIR = BASE_DIR / "prompts"

LOGS_DIR = BASE_DIR / "logs"

BM25_DIR = BASE_DIR / "bm25"

BM25_INDEX_FILE = BM25_DIR / "bm25_index.pkl"
BM25_DOCUMENTS_FILE = BM25_DIR / "documents.pkl"

# ==========================================================
# LLM Settings
# ==========================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama",
)

LLM_API_URL = os.getenv(
    "LLM_API_URL",
    "http://localhost:11434/v1",
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    "ollama",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2.5:7b-Instruct",
)

LLM_MAX_TOKENS = int(
    os.getenv(
        "LLM_MAX_TOKENS",
        "2048",
    )
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.1",
    )
)

# ==========================================================
# Embedding Settings
# ==========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)

RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "BAAI/bge-reranker-v2-m3",
)

# ==========================================================
# Chroma Settings
# ==========================================================

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "egyptian_laws",
)

# ==========================================================
# Chunking Settings
# ==========================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "500",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "100",
    )
)

# ==========================================================
# Retrieval Settings
# ==========================================================

TOP_K = int(
    os.getenv(
        "TOP_K",
        "1",
    )
)

RERANK_TOP_K = int(
    os.getenv(
        "RERANK_TOP_K",
        "10",
    )
)

# ==========================================================
# Server Settings
# ==========================================================

APP_HOST = os.getenv(
    "APP_HOST",
    "0.0.0.0",
)

APP_PORT = int(
    os.getenv(
        "APP_PORT",
        "8000",
    )
)