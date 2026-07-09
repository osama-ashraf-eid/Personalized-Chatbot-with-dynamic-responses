from copy import deepcopy
from typing import List

from sentence_transformers import CrossEncoder

from config import RERANK_MODEL


class Reranker:
    """
    Cross-Encoder based reranker.
    """

    _model = None

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
    ):

        self.model_name = model_name

        if Reranker._model is None:

            print(
                f"\nLoading reranker model: {model_name}"
            )

            Reranker._model = CrossEncoder(
                model_name
            )

        self.model = Reranker._model

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @staticmethod
    def _validate_results(
        results: List[dict],
        text_field: str,
    ):

        if not results:
            return

        for index, item in enumerate(results):

            if text_field not in item:

                raise KeyError(
                    f"Result #{index} "
                    f"does not contain '{text_field}'."
                )

    # --------------------------------------------------
    # Predict Scores
    # --------------------------------------------------

    def score(
        self,
        query: str,
        results: List[dict],
        text_field: str = "embedding_text",
    ) -> List[float]:

        self._validate_results(
            results,
            text_field,
        )

        pairs = [
            (
                query,
                item[text_field],
            )
            for item in results
        ]

        scores = self.model.predict(
            pairs
        )

        return scores.tolist()

    # --------------------------------------------------
    # Rerank
    # --------------------------------------------------

    def rerank(
        self,
        query: str,
        results: List[dict],
        text_field: str = "embedding_text",
        top_k: int | None = None,
    ) -> List[dict]:

        if not results:
            return []

        scores = self.score(
            query=query,
            results=results,
            text_field=text_field,
        )

        ranked = []

        for item, score in zip(
            results,
            scores,
        ):

            doc = deepcopy(item)

            doc["rerank_score"] = float(score)

            ranked.append(doc)

        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        if top_k is not None:

            ranked = ranked[:top_k]

        return ranked

    # --------------------------------------------------
    # Alias
    # --------------------------------------------------

    def __call__(
        self,
        query: str,
        results: List[dict],
        text_field: str = "embedding_text",
        top_k: int | None = None,
    ):

        return self.rerank(
            query=query,
            results=results,
            text_field=text_field,
            top_k=top_k,
        )