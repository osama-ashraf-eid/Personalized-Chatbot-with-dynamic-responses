from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore

from retrieval.bm25_indexing import BM25Index
from retrieval.sparse_retriever import SparseRetriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.search_pipeline import SearchPipeline

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
)


class SearchFactory:
    """
    Factory responsible for building
    a complete SearchPipeline.
    """

    @staticmethod
    def build() -> SearchPipeline:

        # ---------------------------------------
        # Embedder
        # ---------------------------------------

        embedder = Embedder()

        # ---------------------------------------
        # Vector Store
        # ---------------------------------------

        vector_store = VectorStore(
            persist_directory=str(CHROMA_DB_DIR),
            collection_name=COLLECTION_NAME,
        )

        # ---------------------------------------
        # BM25
        # ---------------------------------------

        bm25 = BM25Index()
        bm25.load()

        # ---------------------------------------
        # Sparse Retriever
        # ---------------------------------------

        sparse = SparseRetriever(
            bm25_index=bm25,
        )

        # ---------------------------------------
        # Dense Retriever
        # ---------------------------------------

        dense = DenseRetriever(
            vector_store=vector_store,
            embedder=embedder,
        )

        # ---------------------------------------
        # Hybrid Retriever
        # ---------------------------------------

        hybrid = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
        )

        # ---------------------------------------
        # Reranker
        # ---------------------------------------

        reranker = Reranker()

        # ---------------------------------------
        # Search Pipeline
        # ---------------------------------------

        pipeline = SearchPipeline(
            hybrid_retriever=hybrid,
            reranker=reranker,
        )

        return pipeline