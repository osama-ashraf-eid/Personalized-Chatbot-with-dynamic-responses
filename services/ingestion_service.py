import subprocess
import sys
from typing import List

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
)

from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from ingestion.pipeline import IngestionPipeline
from ingestion.loader import discover_laws

from retrieval.bm25_indexing import BM25Index

from core.logging import get_logger

logger = get_logger(__name__)


class IngestionService:
    """
    Orchestrates the full ingestion workflow:

    1. Generate metadata from raw .txt files
    2. Run ingestion pipeline (clean → chunk → embed → store)
    3. Build BM25 index
    """

    def __init__(self):

        self.embedder = Embedder()

        self.vector_store = VectorStore(
            persist_directory=str(CHROMA_DB_DIR),
            collection_name=COLLECTION_NAME,
        )

        self.pipeline = IngestionPipeline(
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

    # --------------------------------------------------
    # Generate Metadata
    # --------------------------------------------------

    @staticmethod
    def generate_metadata() -> None:
        """
        Run the metadata generation script.
        """

        logger.info("Generating metadata from raw files...")

        subprocess.run(
            [
                sys.executable,
                "scripts/generate_metadata.py",
            ],
            check=True,
        )

        logger.info("Metadata generation complete.")

    # --------------------------------------------------
    # Build BM25 Index
    # --------------------------------------------------

    def build_bm25_index(self) -> None:
        """
        Build and save BM25 index from all ingested data.
        """

        from ingestion.loader import load_all_metadata
        from ingestion.cleaner import clean_articles
        from ingestion.chunker import chunk_articles
        from ingestion.normalizer import normalize_chunks

        logger.info("Building BM25 index...")

        articles = load_all_metadata()

        if not articles:
            logger.warning(
                "No articles found for BM25 index."
            )
            return

        articles = clean_articles(articles)

        chunks = chunk_articles(articles)

        chunks = normalize_chunks(chunks)

        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save()

        logger.info(
            f"BM25 index built with "
            f"{len(chunks)} documents."
        )

    # --------------------------------------------------
    # Full Ingest
    # --------------------------------------------------

    def ingest_all(
        self,
        reset: bool = True,
    ) -> dict:
        """
        Full ingestion workflow.

        Parameters
        ----------
        reset
            Whether to reset the vector store.

        Returns
        -------
        dict
            {
                "laws_processed": [...],
                "total_chunks": int,
            }
        """

        # Step 1: Generate metadata

        self.generate_metadata()

        # Step 2: Run ingestion pipeline

        laws = discover_laws()

        total_chunks = self.pipeline.run_all(
            reset_collection=reset,
        )

        # Step 3: Build BM25 index

        self.build_bm25_index()

        logger.info(
            f"Ingestion complete: {len(laws)} laws, "
            f"{total_chunks} chunks."
        )

        return {
            "laws_processed": laws,
            "total_chunks": total_chunks,
        }
