import json
from pathlib import Path

from config import METADATA_DIR, DATA_RAW_DIR

from core.logging import get_logger

logger = get_logger(__name__)


# ==========================================================
# Auto-Discover Laws
# ==========================================================


def discover_laws() -> list[str]:
    """
    Auto-discover all law files in data/raw/.

    Returns
    -------
    list[str]
        List of law names (file stems).

    Example
    -------
        ["criminal_code", "criminal_procedure", "education_code"]
    """

    if not DATA_RAW_DIR.exists():

        logger.warning(
            f"Data directory not found: {DATA_RAW_DIR}"
        )

        return []

    law_files = sorted(DATA_RAW_DIR.glob("*.txt"))

    laws = [f.stem for f in law_files]

    logger.info(
        f"Discovered {len(laws)} law files: {laws}"
    )

    return laws


# ==========================================================
# Load Metadata
# ==========================================================


def load_metadata(law_name: str) -> list[dict]:
    """
    Load metadata for a single law.

    Example:
        load_metadata("criminal_code")
        load_metadata("criminal_procedure")
        load_metadata("education_code")
    """

    metadata_file = METADATA_DIR / f"{law_name}_metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}"
        )

    with open(metadata_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

        if not isinstance(articles, list):
            raise ValueError(
                f"{metadata_file.name} must contain a list of articles."
            )

    return articles


# ==========================================================
# Load All Metadata
# ==========================================================


def load_all_metadata() -> list[dict]:
    """
    Load metadata for all discovered laws.

    Returns
    -------
    list[dict]
        Combined list of all articles from all laws.
    """

    laws = discover_laws()

    all_articles = []

    for law in laws:

        try:

            articles = load_metadata(law)

            all_articles.extend(articles)

            logger.info(
                f"Loaded {len(articles)} articles "
                f"from '{law}'"
            )

        except FileNotFoundError:

            logger.warning(
                f"Metadata not found for '{law}'. "
                f"Run scripts/generate_metadata.py first."
            )

    return all_articles