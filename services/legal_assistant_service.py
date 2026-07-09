from typing import Optional

from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService
from services.routing_service import RoutingService

from core.logging import get_logger

logger = get_logger(__name__)


class LegalAssistantService:
    """
    End-to-end Legal Assistant Service.

    Query
        ↓
    Route (classify law domain)
        ↓
    Retrieve (hybrid search + rerank + dedup)
        ↓
    Generate (LLM answer + citations)
        ↓
    Structured Response
    """

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
        routing_service: RoutingService | None = None,
    ):

        self.retrieval = (
            retrieval_service or RetrievalService()
        )

        self.generation = (
            generation_service or GenerationService()
        )

        self.routing = (
            routing_service or RoutingService()
        )

    # --------------------------------------------------
    # Ask
    # --------------------------------------------------

    def ask(
        self,
        question: str,
        law_filter: Optional[str] = None,
        top_k: int = 1,
    ) -> dict:
        """
        Answer a legal question end-to-end.

        Parameters
        ----------
        question
            User's legal question in Arabic.

        law_filter
            Optional explicit law filter.

        top_k
            Number of chunks to retrieve.

        Returns
        -------
        dict
            {
                "question": ...,
                "answer": ...,
                "citations": [...],
                "sources": [...],
            }
        """

        logger.info(
            f"Processing question: {question[:80]}..."
        )

        # Step 1: Route

        route_result = self.routing.route(
            query=question,
            law_filter=law_filter,
        )

        # Use routing result as filter

        effective_filter = None

        if route_result["filter_applied"]:

            target_laws = route_result["target_laws"]

            if len(target_laws) == 1:
                effective_filter = target_laws[0]

        # Override with explicit filter

        if law_filter:
            effective_filter = law_filter

        # Step 2: Retrieve

        retrieved = self.retrieval.hybrid_search(
            query=question,
            top_k=top_k,
            law_filter=effective_filter,
        )

        logger.info(
            f"Retrieved {len(retrieved)} chunks"
        )

        # Step 3: Generate

        response = self.generation.generate_answer(
            question=question,
            retrieved_chunks=retrieved,
        )

        return response
