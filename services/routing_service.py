from typing import Optional

from routing.routing_pipeline import RoutingPipeline

from core.logging import get_logger

logger = get_logger(__name__)


class RoutingService:
    """
    Wraps the RoutingPipeline for service-level use.
    """

    def __init__(
        self,
        pipeline: RoutingPipeline | None = None,
    ):

        self.pipeline = (
            pipeline or RoutingPipeline()
        )

    # --------------------------------------------------
    # Route Query
    # --------------------------------------------------

    def route(
        self,
        query: str,
        law_filter: Optional[str] = None,
    ) -> dict:
        """
        Route a query to target laws.
        """

        return self.pipeline.route(
            query=query,
            law_filter=law_filter,
        )
