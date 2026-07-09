from copy import deepcopy
from typing import Dict, List


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Combines multiple ranked retrieval lists into one ranking.

    Reference:
    Cormack et al. (2009)
    """

    def __init__(
        self,
        k: int = 60,
    ):

        self.k = k

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _update_scores(
        self,
        scores: Dict[str, dict],
        results: List[dict],
    ) -> None:

        for rank, item in enumerate(results, start=1):

            chunk_id = item["chunk_id"]

            if chunk_id not in scores:

                scores[chunk_id] = {
                    "score": 0.0,
                    "result": deepcopy(item),
                }

            scores[chunk_id]["score"] += (
                1.0 / (self.k + rank)
            )

    # --------------------------------------------------
    # Fuse
    # --------------------------------------------------

    def fuse(
        self,
        *ranked_lists: List[dict],
    ) -> List[dict]:
        """
        Fuse any number of ranked retrieval lists.
        """

        scores = {}

        for results in ranked_lists:

            if results:

                self._update_scores(
                    scores,
                    results,
                )

        fused = []

        for item in scores.values():

            result = deepcopy(
                item["result"]
            )

            result["rrf_score"] = item["score"]

            fused.append(result)

        fused.sort(
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return fused

    # --------------------------------------------------
    # Alias
    # --------------------------------------------------

    def __call__(
        self,
        *ranked_lists: List[dict],
    ) -> List[dict]:

        return self.fuse(
            *ranked_lists
        )