from typing import List

from generation.generation_pipeline import GenerationPipeline

from core.logging import get_logger

logger = get_logger(__name__)


class GenerationService:
    """
    Wraps the GenerationPipeline for service-level use.
    """

    def __init__(
        self,
        pipeline: GenerationPipeline | None = None,
    ):

        self.pipeline = (
            pipeline or GenerationPipeline()
        )

    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[dict],
    ) -> dict:
        """
        Generate an answer from retrieved chunks.

        Parameters
        ----------
        question
            User's question.

        retrieved_chunks
            Retrieved legal chunks.

        Returns
        -------
        dict
            Structured response.
        """

        return self.pipeline.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
