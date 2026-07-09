from typing import List

from core.constants import FALLBACK_MESSAGE
from core.logging import get_logger

logger = get_logger(__name__)


class AnswerFormatter:
    """
    Formats the final answer combining LLM output
    and citations.
    """

    # --------------------------------------------------
    # Format Answer
    # --------------------------------------------------

    @staticmethod
    def format(
        answer: str,
        citations_text: str = "",
    ) -> str:
        """
        Combine LLM answer with citations.

        Parameters
        ----------
        answer
            Raw LLM response.

        citations_text
            Formatted citation string.

        Returns
        -------
        str
            Final formatted answer.
        """

        if not answer or not answer.strip():
            return FALLBACK_MESSAGE

        parts = [answer.strip()]

        if citations_text:
            parts.append("")
            parts.append(citations_text)

        return "\n".join(parts)

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    @staticmethod
    def fallback() -> str:
        """
        Return the fallback message.
        """

        return FALLBACK_MESSAGE

    # --------------------------------------------------
    # Build Response Dict
    # --------------------------------------------------

    @staticmethod
    def build_response(
        question: str,
        answer: str,
        citations: List[dict],
        sources: List[dict],
    ) -> dict:
        """
        Build a structured response dictionary.

        Parameters
        ----------
        question
            Original question.

        answer
            Formatted answer text.

        citations
            List of citation dicts.

        sources
            List of retrieved chunk dicts.

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

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "sources": [
                {
                    "chunk_id": s.get("chunk_id", ""),
                    "law": s.get("law", ""),
                    "law_name": s.get("law_name", ""),
                    "article_label": s.get(
                        "article_label", ""
                    ),
                    "chapter": s.get("chapter", ""),
                    "chunk_text": s.get(
                        "chunk_text", ""
                    ),
                    "rerank_score": s.get(
                        "rerank_score",
                        s.get("rrf_score", 0.0),
                    ),
                }
                for s in sources
            ],
        }
