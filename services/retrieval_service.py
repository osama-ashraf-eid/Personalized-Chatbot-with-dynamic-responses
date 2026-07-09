from typing import List, Optional

from retrieval.factory import SearchFactory
from retrieval.search_pipeline import SearchPipeline
from routing.router import QueryRouter

from config import TOP_K

from core.logging import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """
    Retrieval service with deduplication
    and metadata filtering.
    """

    def __init__(
        self,
        search_pipeline: SearchPipeline | None = None,
        router: QueryRouter | None = None,
    ):

        self.search = (
            search_pipeline or SearchFactory.build()
        )

        self.router = router or QueryRouter()

    # --------------------------------------------------
    # Deduplicate
    # --------------------------------------------------

    @staticmethod
    def deduplicate(
        results: List[dict],
    ) -> List[dict]:
        """
        Remove duplicate chunks by chunk_id.
        """

        seen = set()

        unique = []

        for item in results:

            chunk_id = item.get("chunk_id", "")

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            unique.append(item)

        return unique

    # --------------------------------------------------
    # Hybrid Search
    # --------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
        law_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Full hybrid search with reranking and dedup.
        """

        results = self.search.search(
            query=query,
            top_k=top_k * 2,
        )

        # Apply law filter

        if law_filter:

            results = [
                r for r in results
                if r.get("law") == law_filter
            ]

        # Deduplicate

        results = self.deduplicate(results)

        return results[:top_k]

    # --------------------------------------------------
    # Dense Search
    # --------------------------------------------------

    def dense_search(
        self,
        query: str,
        top_k: int = TOP_K,
        law_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Dense-only search.
        """

        results = self.search.dense(
            query=query,
            top_k=top_k * 2,
        )

        if law_filter:

            results = [
                r for r in results
                if r.get("law") == law_filter
            ]

        return self.deduplicate(results)[:top_k]

    # --------------------------------------------------
    # Sparse Search
    # --------------------------------------------------

    def sparse_search(
        self,
        query: str,
        top_k: int = TOP_K,
        law_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        BM25-only search.
        """

        results = self.search.sparse(
            query=query,
            top_k=top_k * 2,
        )

        if law_filter:

            results = [
                r for r in results
                if r.get("law") == law_filter
            ]

        return self.deduplicate(results)[:top_k]

    # --------------------------------------------------
    # Search by Mode
    # --------------------------------------------------

    def search_by_mode(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = TOP_K,
        law_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Search using the specified mode.

        Parameters
        ----------
        mode
            "hybrid", "dense", or "sparse"
        """

        if mode == "dense":

            return self.dense_search(
                query=query,
                top_k=top_k,
                law_filter=law_filter,
            )

        elif mode == "sparse":

            return self.sparse_search(
                query=query,
                top_k=top_k,
                law_filter=law_filter,
            )

        else:

            return self.hybrid_search(
                query=query,
                top_k=top_k,
                law_filter=law_filter,
            )
