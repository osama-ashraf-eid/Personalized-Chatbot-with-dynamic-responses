from typing import List, Optional

from ingestion.loader import discover_laws, load_metadata

from core.logging import get_logger

logger = get_logger(__name__)


# ==========================================================
# Keyword Maps (Auto-built from metadata)
# ==========================================================


def _build_keyword_map() -> dict:
    """
    Build a keyword map from discovered law metadata.

    Maps law names (Arabic) to their file stems.
    """

    keyword_map = {}

    laws = discover_laws()

    for law in laws:

        try:
            articles = load_metadata(law)

            if articles:

                law_name = articles[0].get(
                    "law_name", ""
                )

                if law_name:
                    keyword_map[law_name] = law

        except FileNotFoundError:
            continue

    return keyword_map


# ==========================================================
# Common Legal Keywords
# ==========================================================

LEGAL_KEYWORDS = {

    # Criminal law

    "عقوبات": ["criminal_code"],
    "عقوبة": ["criminal_code"],
    "جريمة": ["criminal_code"],
    "جرائم": ["criminal_code"],
    "جناية": ["criminal_code"],
    "جنحة": ["criminal_code"],
    "سرقة": ["criminal_code"],
    "قتل": ["criminal_code"],
    "ضرب": ["criminal_code"],
    "حبس": ["criminal_code"],
    "سجن": ["criminal_code"],
    "غرامة": ["criminal_code"],
    "إعدام": ["criminal_code"],

    # Criminal procedure

    "إجراءات جنائية": ["criminal_procedure"],
    "تحقيق": ["criminal_procedure"],
    "نيابة": ["criminal_procedure"],
    "قبض": ["criminal_procedure"],
    "تفتيش": ["criminal_procedure"],
    "محاكمة": ["criminal_procedure"],
    "استئناف": ["criminal_procedure"],
    "نقض": ["criminal_procedure"],
    "حكم": ["criminal_procedure"],
    "طعن": ["criminal_procedure"],
    "دعوى": ["criminal_procedure"],

    # Education

    "تعليم": ["education_code"],
    "مدرسة": ["education_code"],
    "مدارس": ["education_code"],
    "طالب": ["education_code"],
    "طلاب": ["education_code"],
    "تلميذ": ["education_code"],
    "تلاميذ": ["education_code"],
    "امتحان": ["education_code"],
    "امتحانات": ["education_code"],
    "منهج": ["education_code"],
    "مناهج": ["education_code"],
}


class LawClassifier:
    """
    Keyword-based law classifier.

    Maps user queries to relevant law domains
    without requiring LLM inference.
    """

    def __init__(self):

        self.keyword_map = LEGAL_KEYWORDS

    # --------------------------------------------------
    # Classify
    # --------------------------------------------------

    def classify(
        self,
        query: str,
    ) -> List[str]:
        """
        Classify a query to relevant law domains.

        Parameters
        ----------
        query
            User's question.

        Returns
        -------
        list[str]
            List of matching law file stems.
            Empty list means search all laws.
        """

        if not query.strip():
            return []

        matched_laws = set()

        query_lower = query.strip()

        for keyword, laws in self.keyword_map.items():

            if keyword in query_lower:

                matched_laws.update(laws)

        result = list(matched_laws)

        if result:

            logger.info(
                f"Classified query to laws: {result}"
            )

        return result

    # --------------------------------------------------
    # Has Match
    # --------------------------------------------------

    def has_match(
        self,
        query: str,
    ) -> bool:
        """
        Check if query matches any known law keywords.
        """

        return len(self.classify(query)) > 0
