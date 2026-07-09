"""
Core settings module.

Re-exports frequently used configuration values
for convenient importing.
"""

from config import (
    BASE_DIR,
    DATA_DIR,
    DATA_RAW_DIR,
    METADATA_DIR,
    CHROMA_DB_DIR,
    PROMPTS_DIR,
    LOGS_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RERANK_TOP_K,
    RERANK_MODEL,
    BM25_DIR,
    BM25_INDEX_FILE,
    BM25_DOCUMENTS_FILE,
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    APP_HOST,
    APP_PORT,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DATA_RAW_DIR",
    "METADATA_DIR",
    "CHROMA_DB_DIR",
    "PROMPTS_DIR",
    "LOGS_DIR",
    "EMBEDDING_MODEL",
    "COLLECTION_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "RERANK_TOP_K",
    "RERANK_MODEL",
    "BM25_DIR",
    "BM25_INDEX_FILE",
    "BM25_DOCUMENTS_FILE",
    "LLM_API_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "APP_HOST",
    "APP_PORT",
]
