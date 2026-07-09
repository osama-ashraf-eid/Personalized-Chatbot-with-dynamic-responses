# ==========================================================
# Base Exception
# ==========================================================


class LegalRAGError(Exception):
    """
    Base exception for the Legal RAG system.
    """
    pass


# ==========================================================
# Ingestion Errors
# ==========================================================


class IngestionError(LegalRAGError):
    """
    Raised when ingestion pipeline fails.
    """
    pass


class MetadataNotFoundError(IngestionError):
    """
    Raised when metadata file is not found.
    """
    pass


class NoLawFilesError(IngestionError):
    """
    Raised when no law files are found in data/raw/.
    """
    pass


# ==========================================================
# Retrieval Errors
# ==========================================================


class RetrievalError(LegalRAGError):
    """
    Raised when retrieval pipeline fails.
    """
    pass


class IndexNotReadyError(RetrievalError):
    """
    Raised when BM25 or vector index is not ready.
    """
    pass


# ==========================================================
# Generation Errors
# ==========================================================


class GenerationError(LegalRAGError):
    """
    Raised when LLM generation fails.
    """
    pass


class LLMConnectionError(GenerationError):
    """
    Raised when LLM API is unreachable.
    """
    pass
