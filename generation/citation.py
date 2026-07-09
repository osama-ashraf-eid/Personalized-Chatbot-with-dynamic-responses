from typing import List

from core.logging import get_logger

logger = get_logger(__name__)


class CitationExtractor:
    """
    Extract structured citations from retrieved chunks.

    Creates a list of citation references with
    law name, article number, and chapter information.
    """

    # --------------------------------------------------
    # Extract Citations
    # --------------------------------------------------

    @staticmethod
    def extract(
        chunks: List[dict],
    ) -> List[dict]:
        """
        Extract unique citations from chunks.

        Parameters
        ----------
        chunks
            Retrieved legal chunks with metadata.

        Returns
        -------
        list[dict]
            List of citation dicts:
            {
                "law": "criminal_code",
                "law_name": "قانون العقوبات",
                "article_label": "المادة 234",
                "chapter": "...",
                "section": "...",
            }
        """

        if not chunks:
            return []

        seen = set()

        citations = []

        for chunk in chunks:

            law = chunk.get("law", "")
            article_label = chunk.get(
                "article_label", ""
            )

            # Deduplicate by law + article

            key = f"{law}_{article_label}"

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                {
                    "law": law,
                    "law_name": chunk.get(
                        "law_name", ""
                    ),
                    "article_label": article_label,
                    "article_number": chunk.get(
                        "article_number", ""
                    ),
                    "chapter": chunk.get(
                        "chapter", ""
                    ),
                    "section": chunk.get(
                        "section", ""
                    ),
                    "book": chunk.get(
                        "book", ""
                    ),
                }
            )

        return citations

    # --------------------------------------------------
    # Format Citations
    # --------------------------------------------------

    @staticmethod
    def format_citations(
        citations: List[dict],
    ) -> str:
        """
        Format citations as a readable Arabic string.
        """

        if not citations:
            return ""

        lines = ["المراجع القانونية:"]

        for i, citation in enumerate(
            citations, start=1
        ):

            parts = []

            if citation.get("law_name"):
                parts.append(citation["law_name"])

            if citation.get("article_label"):
                parts.append(citation["article_label"])

            if citation.get("chapter"):
                parts.append(citation["chapter"])

            line = " - ".join(parts)

            lines.append(f"  {i}. {line}")

        return "\n".join(lines)
