from ingestion.loader import load_metadata, discover_laws
from ingestion.cleaner import clean_articles
from ingestion.chunker import chunk_articles
from ingestion.normalizer import normalize_chunks
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore

from core.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Complete ingestion pipeline.

    Stages
    ------
    1. Load metadata
    2. Clean text
    3. Chunk articles
    4. Normalize chunks
    5. Generate embeddings
    6. Store vectors
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder | None = None,
    ):

        self.vector_store = vector_store
        self.embedder = embedder or Embedder()

    # =====================================================
    # Internal
    # =====================================================

    def _process_articles(
        self,
        articles: list[dict],
    ) -> list[dict]:
        """
        Run all preprocessing stages except storage.
        """

        articles = clean_articles(articles)

        chunks = chunk_articles(articles)

        chunks = normalize_chunks(chunks)

        chunks = self.embedder.embed_chunks(chunks)

        return chunks

    # =====================================================
    # Single Law
    # =====================================================

    def run(
        self,
        law_name: str,
        reset_collection: bool = False,
    ) -> int:
        """
        Process one law and store it.
        """

        if reset_collection:
            self.vector_store.reset()

        logger.info(f"Processing law: {law_name}")

        articles = load_metadata(law_name)

        chunks = self._process_articles(
            articles
        )

        self.vector_store.add_chunks(
            chunks
        )

        logger.info(
            f"Ingested {len(chunks)} chunks "
            f"from '{law_name}'"
        )

        return len(chunks)

    # =====================================================
    # Multiple Laws
    # =====================================================

    def run_many(
        self,
        laws: list[str],
        reset_collection: bool = False,
    ) -> int:
        """
        Process selected laws.
        """

        total = 0

        for index, law in enumerate(laws):

            total += self.run(
                law_name=law,
                reset_collection=(
                    reset_collection and index == 0
                ),
            )

        return total

    # =====================================================
    # All Discovered Laws
    # =====================================================

    def run_all(
        self,
        reset_collection: bool = True,
    ) -> int:
        """
        Process every law discovered in data/raw/.

        No hardcoded law names — auto-discovers
        all .txt files.
        """

        laws = discover_laws()

        if not laws:

            logger.warning(
                "No law files found in data/raw/"
            )

            return 0

        logger.info(
            f"Starting ingestion for {len(laws)} laws: "
            f"{laws}"
        )

        return self.run_many(
            laws=laws,
            reset_collection=reset_collection,
        )