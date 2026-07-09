from pathlib import Path

from config import PROMPTS_DIR

from ingestion.loader import discover_laws, load_metadata

from core.logging import get_logger

logger = get_logger(__name__)


class RouterPrompt:
    """
    Dynamic router prompt builder.

    Discovers available laws from metadata
    and builds prompts without hardcoded names.
    """

    def __init__(
        self,
        prompts_dir: Path = PROMPTS_DIR,
    ):

        self.prompts_dir = prompts_dir

    # --------------------------------------------------
    # Get Available Laws
    # --------------------------------------------------

    @staticmethod
    def get_available_laws() -> list[dict]:
        """
        Get info about all available laws.
        """

        laws = discover_laws()

        law_info = []

        for law in laws:

            try:
                articles = load_metadata(law)

                if articles:

                    law_info.append(
                        {
                            "key": law,
                            "name": articles[0].get(
                                "law_name", law
                            ),
                            "article_count": len(articles),
                        }
                    )

            except FileNotFoundError:
                continue

        return law_info

    # --------------------------------------------------
    # Build Prompt
    # --------------------------------------------------

    def build(
        self,
        question: str,
    ) -> str:
        """
        Build a routing prompt with available laws.
        """

        template_path = (
            self.prompts_dir / "router_prompt.txt"
        )

        if not template_path.exists():
            return ""

        with open(
            template_path, "r", encoding="utf-8"
        ) as f:
            template = f.read()

        # Build law list

        laws_info = self.get_available_laws()

        laws_text = "\n".join(
            f"- {info['name']} ({info['key']})"
            for info in laws_info
        )

        return template.format(
            available_laws=laws_text,
            question=question,
        )
