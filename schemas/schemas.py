from pydantic import BaseModel, Field
from typing import List, Optional


# ==========================================================
# Request Models
# ==========================================================


class QuestionRequest(BaseModel):
    """
    Chat / QA request body.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Legal question in Arabic.",
    )


class SearchRequest(BaseModel):
    """
    Retrieval-only request body.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Search query.",
    )

    mode: str = Field(
        default="hybrid",
        description=(
            "Search mode: hybrid, dense, or sparse."
        ),
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results.",
    )

    law_filter: Optional[str] = Field(
        default=None,
        description="Optional law filter.",
    )


class IngestRequest(BaseModel):
    """
    Ingestion trigger request.
    """

    reset: bool = Field(
        default=False,
        description="Reset collection before ingestion.",
    )


# ==========================================================
# Response Models
# ==========================================================


class CitationItem(BaseModel):
    """
    A single legal citation.
    """

    law: str = ""
    law_name: str = ""
    article_label: str = ""
    article_number: str = ""
    chapter: str = ""
    section: str = ""
    book: str = ""


class SourceItem(BaseModel):
    """
    A single retrieved source chunk.
    """

    chunk_id: str = ""
    law: str = ""
    law_name: str = ""
    article_label: str = ""
    chapter: str = ""
    chunk_text: str = ""
    rerank_score: float = 0.0


class AnswerResponse(BaseModel):
    """
    Full QA response.
    """

    question: str
    answer: str
    citations: List[CitationItem] = []
    sources: List[SourceItem] = []


class SearchResult(BaseModel):
    """
    A single search result.
    """

    chunk_id: str = ""
    law: str = ""
    law_name: str = ""
    article_label: str = ""
    chapter: str = ""
    section: str = ""
    chunk_text: str = ""
    embedding_text: str = ""
    score: float = 0.0


class SearchResponse(BaseModel):
    """
    Search results response.
    """

    query: str
    mode: str
    total: int
    results: List[SearchResult] = []


class IngestionResponse(BaseModel):
    """
    Ingestion result.
    """

    status: str
    laws_processed: List[str] = []
    total_chunks: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = "ok"
    version: str = "1.0.0"
    service: str = "Egyptian Legal RAG"