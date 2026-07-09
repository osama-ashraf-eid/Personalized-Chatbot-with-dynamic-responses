from fastapi import APIRouter, HTTPException

from schemas.schemas import (
    IngestRequest,
    IngestionResponse,
)

from services.ingestion_service import IngestionService

from core.logging import get_logger
from core.exceptions import IngestionError

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Ingestion"],
)


# ==========================================================
# Ingest Endpoint
# ==========================================================


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Trigger data ingestion",
)
async def ingest(request: IngestRequest):
    """
    Trigger the full ingestion pipeline.

    This will:
    1. Generate metadata from raw .txt files
    2. Clean, chunk, embed, and store all documents
    3. Build the BM25 index
    """

    try:

        service = IngestionService()

        result = service.ingest_all(
            reset=request.reset,
        )

        return IngestionResponse(
            status="success",
            laws_processed=result["laws_processed"],
            total_chunks=result["total_chunks"],
            message=(
                f"Ingested {result['total_chunks']} chunks "
                f"from {len(result['laws_processed'])} laws."
            ),
        )

    except IngestionError as e:

        logger.error(f"Ingestion error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except Exception as e:

        logger.error(f"Unexpected error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}",
        )
