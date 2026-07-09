from typing import Optional

from routing.router import QueryRouter

from core.logging import get_logger

logger = get_logger(__name__)


class RoutingPipeline:
    """
    End-to-end routing pipeline.

    Query
        ↓
    Classify / Filter
        ↓
    Route Result
    """

    def __init__(
        self,
        router: QueryRouter | None = None,
    ):

        self.router = router or QueryRouter()

    # --------------------------------------------------
    # Route
    # --------------------------------------------------

    def route(
        self,
        query: str,
        law_filter: Optional[str] = None,
    ) -> dict:
        """
        Route a query through the pipeline.

        Parameters
        ----------
        query
            User's question.

        law_filter
            Optional explicit law filter.

        Returns
        -------
        dict
            Routing result with target laws.
        """

        result = self.router.route(
            query=query,
            law_filter=law_filter,
        )

        logger.info(
            f"Routed query → "
            f"laws={result['target_laws']}, "
            f"filtered={result['filter_applied']}"
        )

        return result
