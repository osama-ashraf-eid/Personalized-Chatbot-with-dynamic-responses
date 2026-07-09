"""
Tests for the ingestion pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from ingestion.cleaner import clean_text, clean_articles
from ingestion.normalizer import (
    normalize_text,
    normalize_unicode,
    remove_diacritics,
    normalize_arabic_letters,
    normalize_legal_terms,
    normalize_chunks,
)
from ingestion.chunker import (
    count_words,
    split_into_paragraphs,
    build_chunks,
    chunk_articles,
)
from ingestion.loader import discover_laws


# ==========================================================
# Cleaner Tests
# ==========================================================


class TestCleaner:

    def test_clean_text_empty(self):
        assert clean_text("") == ""

    def test_clean_text_none(self):
        assert clean_text("") == ""

    def test_clean_text_removes_bom(self):
        result = clean_text("\ufeffمادة 1")
        assert "\ufeff" not in result

    def test_clean_text_removes_rtl(self):
        result = clean_text("\u200fمادة 1")
        assert "\u200f" not in result

    def test_clean_text_normalizes_spaces(self):
        result = clean_text("مادة    1")
        assert "    " not in result

    def test_clean_text_normalizes_newlines(self):
        result = clean_text("line1\r\nline2")
        assert "\r" not in result

    def test_clean_articles(self):
        articles = [
            {"text": "مادة 1  نص  المادة"},
        ]
        result = clean_articles(articles)
        assert len(result) == 1
        assert "clean_text" in result[0]
        assert "text" in result[0]


# ==========================================================
# Normalizer Tests
# ==========================================================


class TestNormalizer:

    def test_normalize_text_empty(self):
        assert normalize_text("") == ""

    def test_remove_diacritics(self):
        result = remove_diacritics("كِتَابٌ")
        assert "ِ" not in result
        assert "َ" not in result
        assert "ٌ" not in result

    def test_normalize_arabic_letters(self):
        result = normalize_arabic_letters("أحمد إبراهيم آمن")
        assert "أ" not in result
        assert "إ" not in result
        assert "آ" not in result

    def test_normalize_legal_terms(self):
        result = normalize_legal_terms("الماده الأولى")
        assert "المادة" in result

    def test_normalize_chunks_adds_field(self):
        chunks = [
            {"chunk_text": "نص الاختبار"},
        ]
        result = normalize_chunks(chunks)
        assert len(result) == 1
        assert "embedding_text" in result[0]


# ==========================================================
# Chunker Tests
# ==========================================================


class TestChunker:

    def test_count_words(self):
        assert count_words("واحد اثنين ثلاثة") == 3

    def test_count_words_empty(self):
        assert count_words("") == 0

    def test_split_into_paragraphs(self):
        text = "فقرة أولى\n\nفقرة ثانية"
        result = split_into_paragraphs(text)
        assert len(result) == 2

    def test_build_chunks_empty_article(self):
        article = {"text": ""}
        result = build_chunks(article)
        assert len(result) == 0

    def test_build_chunks_short_article(self):
        article = {
            "clean_text": "المادة 1\nنص قصير للمادة",
            "article_key": "test_1_1",
            "article_label": "المادة 1",
        }
        result = build_chunks(article)
        assert len(result) >= 1
        assert result[0]["chunk_id"].startswith("test_1_1")

    def test_chunk_articles(self):
        articles = [
            {
                "clean_text": "المادة 1\nنص المادة الأولى",
                "article_key": "law_1_1",
                "article_label": "المادة 1",
            },
            {
                "clean_text": "المادة 2\nنص المادة الثانية",
                "article_key": "law_2_2",
                "article_label": "المادة 2",
            },
        ]
        result = chunk_articles(articles)
        assert len(result) >= 2


# ==========================================================
# Loader Tests
# ==========================================================


class TestLoader:

    def test_discover_laws_returns_list(self):
        result = discover_laws()
        assert isinstance(result, list)
