from fastapi import APIRouter

from schemas.schemas import HealthResponse

router = APIRouter(tags=["Health"])


# ==========================================================
# Health Check
# ==========================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check():
    """
    Check if the service is running.
    """

    return HealthResponse(
        status="ok",
        version="1.0.0",
        service="Egyptian Legal RAG",
    )
