from typing import List

from config import (
    TOP_K,
    RERANK_TOP_K,
)

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker


class SearchPipeline:
    """
    Complete Retrieval Pipeline.

    Query
        ↓
    Hybrid Retrieval
        ↓
    Cross Encoder Reranking
        ↓
    Final Documents
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ):

        self.hybrid = hybrid_retriever
        self.reranker = reranker

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
    ) -> List[dict]:

        if not query.strip():
            return []

        retrieved = self.hybrid.search(
            query=query,
            top_k=rerank_top_k,
        )

        return self.reranker(
            query=query,
            results=retrieved,
            top_k=top_k,
        )

    # --------------------------------------------------
    # Retrieve Only
    # --------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.hybrid.search(
            query=query,
            top_k=top_k,
        )

    # --------------------------------------------------
    # Sparse Only
    # --------------------------------------------------

    def sparse(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.hybrid.sparse_search(
            query=query,
            top_k=top_k,
        )

    # --------------------------------------------------
    # Dense Only
    # --------------------------------------------------

    def dense(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.hybrid.dense_search(
            query=query,
            top_k=top_k,
        )

    # --------------------------------------------------
    # Hybrid Only
    # --------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.hybrid.search(
            query=query,
            top_k=top_k,
        )

    # --------------------------------------------------
    # Alias
    # --------------------------------------------------

    def __call__(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.search(
            query=query,
            top_k=top_k,
        )