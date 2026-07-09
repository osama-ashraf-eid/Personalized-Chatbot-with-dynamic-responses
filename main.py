from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.health import router as health_router
from api.routers.chat import router as chat_router
from api.routers.retrieval import router as retrieval_router
from api.routers.ingestion import router as ingestion_router

from config import APP_HOST, APP_PORT

from core.logging import get_logger

logger = get_logger(__name__)


# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title="Egyptian Legal RAG API",
    description=(
        "A production-ready Retrieval-Augmented Generation "
        "system for Egyptian legal documents. "
        "Powered by Qwen2.5-7B-Instruct and ChromaDB."
    ),
    version="1.0.0",
)


# ==========================================================
# CORS Middleware
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Routers
# ==========================================================

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(retrieval_router)
app.include_router(ingestion_router)


# ==========================================================
# Startup Event
# ==========================================================


@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks.
    """

    logger.info("=" * 60)
    logger.info("Egyptian Legal RAG API starting...")
    logger.info("=" * 60)


# ==========================================================
# Root
# ==========================================================


@app.get("/")
async def root():
    """
    Root endpoint.
    """

    return {
        "service": "Egyptian Legal RAG API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ==========================================================
# Run Server
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
    )
