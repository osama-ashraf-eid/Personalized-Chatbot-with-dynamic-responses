import re
from copy import deepcopy

# ==========================================================
# Chunk Configuration
# ==========================================================

MAX_CHUNK_WORDS = 350
MIN_CHUNK_WORDS = 120


# ==========================================================
# Helpers
# ==========================================================

def count_words(text: str) -> int:
    """
    Count words in text.
    """
    return len(text.split())


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split article into logical paragraphs.

    Empty paragraphs are ignored.
    """

    paragraphs = []

    for paragraph in text.split("\n\n"):

        paragraph = paragraph.strip()

        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def split_long_paragraph(paragraph: str) -> list[str]:
    """
    Split a long paragraph into sentence groups.
    """

    sentences = re.split(
        r'(?<=[.!؟])\s+',
        paragraph
    )

    chunks = []

    current = []

    current_words = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        words = count_words(sentence)

        if (
            current
            and current_words + words > MAX_CHUNK_WORDS
        ):

            chunks.append(
                " ".join(current)
            )

            current = [sentence]

            current_words = words

        else:

            current.append(sentence)

            current_words += words

    if current:

        chunks.append(
            " ".join(current)
        )

    return chunks


# ==========================================================
# Chunk Builder
# ==========================================================

def build_chunks(article: dict) -> list[dict]:
    """
    Build logical chunks from one legal article.
    """

    text = article.get(
        "clean_text",
        article.get("text", "")
    )

    if not text:
        return []

    paragraphs = split_into_paragraphs(text)

    # ------------------------------------------------------
    # Single paragraph article
    # ------------------------------------------------------

    if len(paragraphs) <= 1:

        if count_words(text) <= MAX_CHUNK_WORDS:

            paragraphs = [text]

        else:

            paragraphs = split_long_paragraph(text)

    current_chunk = []

    current_words = 0

    chunk_texts = []

    # ------------------------------------------------------
    # Build chunks
    # ------------------------------------------------------

    for paragraph in paragraphs:

        words = count_words(paragraph)

        # Huge paragraph

        if words > MAX_CHUNK_WORDS:

            if current_chunk:

                chunk_texts.append(
                    "\n\n".join(current_chunk)
                )

                current_chunk = []

                current_words = 0

            chunk_texts.extend(
                split_long_paragraph(paragraph)
            )

            continue

        # Normal paragraph

        if current_words + words <= MAX_CHUNK_WORDS:

            current_chunk.append(paragraph)

            current_words += words

        else:

            chunk_texts.append(
                "\n\n".join(current_chunk)
            )

            current_chunk = [paragraph]

            current_words = words

    if current_chunk:

        chunk_texts.append(
            "\n\n".join(current_chunk)
        )

    # ------------------------------------------------------
    # Merge very small last chunk
    # ------------------------------------------------------

    if (
        len(chunk_texts) > 1
        and count_words(chunk_texts[-1]) < MIN_CHUNK_WORDS
    ):
        chunk_texts[-2] += "\n\n" + chunk_texts[-1]
        chunk_texts.pop()

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    total_chunks = len(chunk_texts)

    chunks = []

    header = (
        article.get("article_label")
        or article.get("title")
        or ""
    )

    for index, chunk in enumerate(
        chunk_texts,
        start=1
    ):

        item = deepcopy(article)

        # Repeat article title in every chunk
        if (
            header
            and not chunk.startswith(header)
        ):
            chunk = f"{header}\n{chunk}"

        item["chunk_id"] = (
            f"{article['article_key']}_{index}"
        )

        item["chunk_index"] = index

        item["total_chunks"] = total_chunks

        item["chunk_word_count"] = count_words(chunk)

        item["chunk_char_count"] = len(chunk)

        # Display text only
        item["chunk_text"] = chunk

        chunks.append(item)

    return chunks


# ==========================================================
# Public API
# ==========================================================

def chunk_articles(
    articles: list[dict]
) -> list[dict]:
    """
    Build chunks for a list of legal articles.
    """

    all_chunks = []

    for article in articles:

        all_chunks.extend(
            build_chunks(article)
        )

    return all_chunks