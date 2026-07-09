from fastapi import APIRouter, HTTPException

from schemas.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)

from services.retrieval_service import RetrievalService

from core.logging import get_logger
from core.exceptions import RetrievalError

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Retrieval"],
)

# ==========================================================
# Lazy-initialized service
# ==========================================================

_service: RetrievalService | None = None


def get_service() -> RetrievalService:
    """
    Lazy initialize the RetrievalService.
    """

    global _service

    if _service is None:
        _service = RetrievalService()

    return _service


# ==========================================================
# Search Endpoint
# ==========================================================


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search legal documents",
)
async def search(request: SearchRequest):
    """
    Search Egyptian legal documents using
    hybrid, dense, or sparse retrieval.
    """

    try:

        service = get_service()

        results = service.search_by_mode(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            law_filter=request.law_filter,
        )

        # Format results

        search_results = []

        for r in results:

            score = r.get(
                "rerank_score",
                r.get(
                    "rrf_score",
                    r.get(
                        "dense_score",
                        r.get("bm25_score", 0.0),
                    ),
                ),
            )

            search_results.append(
                SearchResult(
                    chunk_id=r.get("chunk_id", ""),
                    law=r.get("law", ""),
                    law_name=r.get("law_name", ""),
                    article_label=r.get(
                        "article_label", ""
                    ),
                    chapter=r.get("chapter", ""),
                    section=r.get("section", ""),
                    chunk_text=r.get("chunk_text", ""),
                    embedding_text=r.get(
                        "embedding_text", ""
                    ),
                    score=float(score),
                )
            )

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            total=len(search_results),
            results=search_results,
        )

    except RetrievalError as e:

        logger.error(f"Retrieval error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except Exception as e:

        logger.error(f"Unexpected error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )
