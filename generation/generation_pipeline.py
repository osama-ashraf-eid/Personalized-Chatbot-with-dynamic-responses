from typing import List

from generation.llm import LLMClient
from generation.prompt_builder import PromptBuilder
from generation.citation import CitationExtractor
from generation.answer_formatter import AnswerFormatter

from core.logging import get_logger

logger = get_logger(__name__)


class GenerationPipeline:
    """
    Complete Generation Pipeline.

    Retrieved Chunks
        ↓
    Build Context
        ↓
    Build Prompt
        ↓
    LLM Generation
        ↓
    Extract Citations
        ↓
    Format Answer
        ↓
    Structured Response
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        citation_extractor: CitationExtractor | None = None,
        answer_formatter: AnswerFormatter | None = None,
    ):

        self.llm = llm or LLMClient()

        self.prompt_builder = (
            prompt_builder or PromptBuilder()
        )

        self.citation_extractor = (
            citation_extractor or CitationExtractor()
        )

        self.answer_formatter = (
            answer_formatter or AnswerFormatter()
        )

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    def generate(
        self,
        question: str,
        retrieved_chunks: List[dict],
    ) -> dict:
        """
        Generate an answer from retrieved chunks.

        Parameters
        ----------
        question
            User's legal question.

        retrieved_chunks
            List of retrieved chunk dicts.

        Returns
        -------
        dict
            Structured response with answer, citations,
            and sources.
        """

        # No context available

        if not retrieved_chunks:

            logger.info(
                "No chunks retrieved — returning fallback"
            )

            return self.answer_formatter.build_response(
                question=question,
                answer=self.answer_formatter.fallback(),
                citations=[],
                sources=[],
            )

        # Build context

        context = self.prompt_builder.build_context(
            retrieved_chunks
        )

        # Build prompts

        system_prompt = self.prompt_builder.build_system_prompt()

        user_prompt = self.prompt_builder.build_legal_prompt(
            context=context,
            question=question,
        )

        # Generate answer

        logger.info(
            f"Generating answer for: {question[:80]}..."
        )

        raw_answer = self.llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Extract citations

        citations = self.citation_extractor.extract(
            retrieved_chunks
        )

        citations_text = self.citation_extractor.format_citations(
            citations
        )

        # Format final answer

        formatted_answer = self.answer_formatter.format(
            answer=raw_answer,
            citations_text=citations_text,
        )

        # Build response

        return self.answer_formatter.build_response(
            question=question,
            answer=formatted_answer,
            citations=citations,
            sources=retrieved_chunks,
        )

    # --------------------------------------------------
    # Alias
    # --------------------------------------------------

    def __call__(
        self,
        question: str,
        retrieved_chunks: List[dict],
    ) -> dict:

        return self.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
