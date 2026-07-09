from typing import List

from config import TOP_K

from retrieval.sparse_retriever import SparseRetriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.rank_fusion import ReciprocalRankFusion


class HybridRetriever:
    """
    Hybrid Retriever.

    Combines:
        • Sparse Retrieval (BM25)
        • Dense Retrieval (Embeddings)

    Then merges results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        rrf: ReciprocalRankFusion | None = None,
    ):

        self.sparse = sparse_retriever

        self.dense = dense_retriever

        self.rrf = rrf or ReciprocalRankFusion()

    # --------------------------------------------------
    # Hybrid Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        sparse_k: int = 20,
        dense_k: int = 20,
    ) -> List[dict]:
        """
        Hybrid retrieval using:

            Sparse
            +
            Dense
            +
            Reciprocal Rank Fusion
        """

        sparse_results = self.sparse.search(
            query=query,
            top_k=sparse_k,
        )

        dense_results = self.dense.search(
            query=query,
            top_k=dense_k,
        )

        fused = self.rrf(
            sparse_results,
            dense_results,
        )

        return fused[:top_k]

    # --------------------------------------------------
    # Sparse Only
    # --------------------------------------------------

    def sparse_search(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.sparse.search(
            query=query,
            top_k=top_k,
        )

    # --------------------------------------------------
    # Dense Only
    # --------------------------------------------------

    def dense_search(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.dense.search(
            query=query,
            top_k=top_k,
        )

    # --------------------------------------------------
    # Alias
    # --------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        return self.search(
            query=query,
            top_k=top_k,
        )