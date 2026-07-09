from fastapi import APIRouter, HTTPException

from schemas.schemas import (
    QuestionRequest,
    AnswerResponse,
)

from services.legal_assistant_service import (
    LegalAssistantService,
)

from core.logging import get_logger
from core.exceptions import LegalRAGError

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Chat"],
)

# ==========================================================
# Lazy-initialized service
# ==========================================================

_service: LegalAssistantService | None = None


def get_service() -> LegalAssistantService:
    """
    Lazy initialize the LegalAssistantService.
    """

    global _service

    if _service is None:
        _service = LegalAssistantService()

    return _service


# ==========================================================
# Chat Endpoint
# ==========================================================


@router.post(
    "/chat",
    response_model=AnswerResponse,
    summary="Ask a legal question",
)
async def chat(request: QuestionRequest):
    """
    Ask a legal question and get an answer
    based on Egyptian law.

    The answer is generated only from retrieved
    legal context. Never hallucinates.
    """

    try:

        service = get_service()

        result = service.ask(
            question=request.question,
            top_k=1,
        )

        return AnswerResponse(**result)

    except LegalRAGError as e:

        logger.error(f"RAG error: {e}")

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