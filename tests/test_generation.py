"""
Tests for the generation components.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from generation.citation import CitationExtractor
from generation.answer_formatter import AnswerFormatter
from generation.prompt_builder import PromptBuilder
from core.constants import FALLBACK_MESSAGE


# ==========================================================
# Citation Tests
# ==========================================================


class TestCitationExtractor:

    def test_extract_empty(self):
        result = CitationExtractor.extract([])
        assert result == []

    def test_extract_single_chunk(self):
        chunks = [
            {
                "law": "criminal_code",
                "law_name": "قانون العقوبات",
                "article_label": "المادة 234",
                "article_number": "234",
                "chapter": "الباب الأول",
                "section": "",
                "book": "",
            }
        ]
        result = CitationExtractor.extract(chunks)
        assert len(result) == 1
        assert result[0]["law"] == "criminal_code"
        assert result[0]["article_label"] == "المادة 234"

    def test_extract_deduplicates(self):
        chunks = [
            {
                "law": "criminal_code",
                "article_label": "المادة 234",
            },
            {
                "law": "criminal_code",
                "article_label": "المادة 234",
            },
        ]
        result = CitationExtractor.extract(chunks)
        assert len(result) == 1

    def test_extract_multiple_articles(self):
        chunks = [
            {
                "law": "criminal_code",
                "article_label": "المادة 234",
            },
            {
                "law": "criminal_code",
                "article_label": "المادة 235",
            },
        ]
        result = CitationExtractor.extract(chunks)
        assert len(result) == 2

    def test_format_citations_empty(self):
        result = CitationExtractor.format_citations([])
        assert result == ""

    def test_format_citations(self):
        citations = [
            {
                "law_name": "قانون العقوبات",
                "article_label": "المادة 234",
                "chapter": "الباب الأول",
            },
        ]
        result = CitationExtractor.format_citations(
            citations
        )
        assert "قانون العقوبات" in result
        assert "المادة 234" in result


# ==========================================================
# Answer Formatter Tests
# ==========================================================


class TestAnswerFormatter:

    def test_format_with_answer(self):
        result = AnswerFormatter.format(
            answer="الإجابة هنا",
        )
        assert "الإجابة هنا" in result

    def test_format_empty_answer(self):
        result = AnswerFormatter.format(
            answer="",
        )
        assert result == FALLBACK_MESSAGE

    def test_format_with_citations(self):
        result = AnswerFormatter.format(
            answer="الإجابة",
            citations_text="المراجع: ...",
        )
        assert "الإجابة" in result
        assert "المراجع" in result

    def test_fallback(self):
        result = AnswerFormatter.fallback()
        assert result == FALLBACK_MESSAGE

    def test_build_response(self):
        result = AnswerFormatter.build_response(
            question="ما هي العقوبة؟",
            answer="الإجابة",
            citations=[],
            sources=[],
        )
        assert result["question"] == "ما هي العقوبة؟"
        assert result["answer"] == "الإجابة"
        assert isinstance(result["citations"], list)
        assert isinstance(result["sources"], list)


# ==========================================================
# Prompt Builder Tests
# ==========================================================


class TestPromptBuilder:

    def test_build_context_empty(self):
        result = PromptBuilder.build_context([])
        assert result == ""

    def test_build_context_single_chunk(self):
        chunks = [
            {
                "law_name": "قانون العقوبات",
                "article_label": "المادة 1",
                "chapter": "الباب الأول",
                "chunk_text": "نص المادة الأولى",
            }
        ]
        result = PromptBuilder.build_context(chunks)
        assert "قانون العقوبات" in result
        assert "نص المادة الأولى" in result
        assert "[1]" in result

    def test_build_context_multiple_chunks(self):
        chunks = [
            {
                "law_name": "قانون العقوبات",
                "article_label": "المادة 1",
                "chunk_text": "نص أول",
            },
            {
                "law_name": "قانون العقوبات",
                "article_label": "المادة 2",
                "chunk_text": "نص ثاني",
            },
        ]
        result = PromptBuilder.build_context(chunks)
        assert "[1]" in result
        assert "[2]" in result
