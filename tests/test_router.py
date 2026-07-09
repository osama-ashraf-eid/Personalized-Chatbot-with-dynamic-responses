"""
Tests for the routing components.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from routing.law_classifier import LawClassifier
from routing.router import QueryRouter


# ==========================================================
# Law Classifier Tests
# ==========================================================


class TestLawClassifier:

    def setup_method(self):
        self.classifier = LawClassifier()

    def test_classify_empty(self):
        result = self.classifier.classify("")
        assert result == []

    def test_classify_criminal_code(self):
        result = self.classifier.classify(
            "ما هي عقوبة السرقة؟"
        )
        assert "criminal_code" in result

    def test_classify_criminal_procedure(self):
        result = self.classifier.classify(
            "ما هي إجراءات التحقيق؟"
        )
        assert "criminal_procedure" in result

    def test_classify_education(self):
        result = self.classifier.classify(
            "ما هي شروط القبول في المدارس؟"
        )
        assert "education_code" in result

    def test_classify_unknown(self):
        result = self.classifier.classify(
            "ما هو الطقس اليوم؟"
        )
        assert result == []

    def test_has_match_true(self):
        assert self.classifier.has_match(
            "عقوبة القتل"
        )

    def test_has_match_false(self):
        assert not self.classifier.has_match(
            "كيف الحال"
        )


# ==========================================================
# Query Router Tests
# ==========================================================


class TestQueryRouter:

    def setup_method(self):
        self.router = QueryRouter()

    def test_route_with_explicit_filter(self):
        result = self.router.route(
            query="أي سؤال",
            law_filter="criminal_code",
        )
        assert result["filter_applied"] is True
        assert result["target_laws"] == ["criminal_code"]

    def test_route_with_keyword_match(self):
        result = self.router.route(
            query="ما هي عقوبة السرقة؟",
        )
        assert result["filter_applied"] is True
        assert "criminal_code" in result["target_laws"]

    def test_route_no_match(self):
        result = self.router.route(
            query="سؤال عام",
        )
        assert result["filter_applied"] is False
        assert result["target_laws"] == []

    def test_build_chroma_filter_none(self):
        result = QueryRouter.build_chroma_filter([])
        assert result is None

    def test_build_chroma_filter_single(self):
        result = QueryRouter.build_chroma_filter(
            ["criminal_code"]
        )
        assert result == {"law": "criminal_code"}

    def test_build_chroma_filter_multiple(self):
        result = QueryRouter.build_chroma_filter(
            ["criminal_code", "education_code"]
        )
        assert "$or" in result
        assert len(result["$or"]) == 2
