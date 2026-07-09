from copy import deepcopy
from typing import List

from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore


class DenseRetriever:
    """
    Dense semantic retriever using ChromaDB.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
    ):

        self.vector_store = vector_store
        self.embedder = embedder

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Perform semantic search.
        """

        if not query.strip():
            return []

        query_embedding = self.embedder.embed_text(
            query
        )

        results = self.vector_store.query(
            embedding=query_embedding,
            top_k=top_k,
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):

            item = deepcopy(metadata)

            item["chunk_id"] = chunk_id

            item["embedding_text"] = document

            item["dense_score"] = float(distance)

            retrieved.append(item)

        return retrieved

    # --------------------------------------------------
    # Alias
    # --------------------------------------------------

    def search_with_scores(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:

        return self.search(
            query=query,
            top_k=top_k,
        )