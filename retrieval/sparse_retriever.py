from copy import deepcopy
from typing import List

import numpy as np

from retrieval.bm25_indexing import BM25Index


class SparseRetriever:
    """
    Sparse Retriever using BM25.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
    ):

        if not bm25_index.is_ready:

            raise ValueError(
                "BM25 index is not ready. "
                "Build or load the index first."
            )

        self.index = bm25_index.index
        self.documents = bm25_index.documents

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Search using BM25.
        """

        if not query.strip():
            return []

        query_tokens = query.split()

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for idx in ranked_indices[:top_k]:

            item = deepcopy(
                self.documents[idx]
            )

            item["bm25_score"] = float(
                scores[idx]
            )

            results.append(item)

        return results

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