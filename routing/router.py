from typing import List, Optional

from routing.law_classifier import LawClassifier

from core.logging import get_logger

logger = get_logger(__name__)


class QueryRouter:
    """
    Routes queries to appropriate law domains.

    Uses keyword classification to determine
    which laws to search.
    """

    def __init__(
        self,
        classifier: LawClassifier | None = None,
    ):

        self.classifier = (
            classifier or LawClassifier()
        )

    # --------------------------------------------------
    # Route
    # --------------------------------------------------

    def route(
        self,
        query: str,
        law_filter: Optional[str] = None,
    ) -> dict:
        """
        Route a query.

        Parameters
        ----------
        query
            User's question.

        law_filter
            Explicit law filter from user.

        Returns
        -------
        dict
            {
                "query": ...,
                "target_laws": [...],
                "filter_applied": bool,
            }
        """

        # Explicit filter takes priority

        if law_filter:

            logger.info(
                f"Explicit law filter: {law_filter}"
            )

            return {
                "query": query,
                "target_laws": [law_filter],
                "filter_applied": True,
            }

        # Keyword classification

        classified_laws = self.classifier.classify(
            query
        )

        return {
            "query": query,
            "target_laws": classified_laws,
            "filter_applied": bool(classified_laws),
        }

    # --------------------------------------------------
    # Build Chroma Filter
    # --------------------------------------------------

    @staticmethod
    def build_chroma_filter(
        target_laws: List[str],
    ) -> Optional[dict]:
        """
        Build a ChromaDB where filter from target laws.

        Parameters
        ----------
        target_laws
            List of law file stems.

        Returns
        -------
        dict or None
            ChromaDB where clause.
        """

        if not target_laws:
            return None

        if len(target_laws) == 1:

            return {
                "law": target_laws[0],
            }

        return {
            "$or": [
                {"law": law}
                for law in target_laws
            ],
        }
