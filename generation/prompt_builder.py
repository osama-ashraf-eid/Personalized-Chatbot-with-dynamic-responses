from pathlib import Path
from typing import List

from config import PROMPTS_DIR

from core.logging import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """
    Builds prompts from template files.

    Templates are loaded from the prompts/ directory
    and filled with dynamic context.
    """

    def __init__(
        self,
        prompts_dir: Path = PROMPTS_DIR,
    ):

        self.prompts_dir = prompts_dir

        # Cache loaded templates

        self._cache = {}

    # --------------------------------------------------
    # Load Template
    # --------------------------------------------------

    def _load_template(
        self,
        filename: str,
    ) -> str:
        """
        Load a prompt template file.
        """

        if filename in self._cache:
            return self._cache[filename]

        filepath = self.prompts_dir / filename

        if not filepath.exists():

            raise FileNotFoundError(
                f"Prompt template not found: {filepath}"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            template = f.read()

        self._cache[filename] = template

        return template

    # --------------------------------------------------
    # System Prompt
    # --------------------------------------------------

    def build_system_prompt(self) -> str:
        """
        Load the system prompt.
        """

        return self._load_template(
            "system_prompt.txt"
        )

    # --------------------------------------------------
    # Legal Prompt
    # --------------------------------------------------

    def build_legal_prompt(
        self,
        context: str,
        question: str,
    ) -> str:
        """
        Build the legal QA prompt with context.

        Parameters
        ----------
        context
            Retrieved legal text chunks.

        question
            User's question.

        Returns
        -------
        str
            Formatted prompt.
        """

        template = self._load_template(
            "legal_prompt.txt"
        )

        return template.format(
            context=context,
            question=question,
        )

    # --------------------------------------------------
    # Fallback Prompt
    # --------------------------------------------------

    def build_fallback_prompt(self) -> str:
        """
        Load the fallback response.
        """

        return self._load_template(
            "fallback_prompt.txt"
        )

    # --------------------------------------------------
    # Context Builder
    # --------------------------------------------------

    @staticmethod
    def build_context(
        chunks: List[dict],
        text_field: str = "chunk_text",
    ) -> str:
        """
        Build context string from retrieved chunks.

        Each chunk is labeled with its source metadata.
        """

        if not chunks:
            return ""

        context_parts = []

        for i, chunk in enumerate(chunks, start=1):

            # Build source label

            law_name = chunk.get("law_name", "")
            article_label = chunk.get("article_label", "")
            chapter = chunk.get("chapter", "")

            source_parts = []

            if law_name:
                source_parts.append(law_name)

            if chapter:
                source_parts.append(chapter)

            if article_label:
                source_parts.append(article_label)

            source = " - ".join(source_parts)

            text = chunk.get(text_field, "")

            context_parts.append(
                f"[{i}] ({source})\n{text}"
            )

        return "\n\n".join(context_parts)
