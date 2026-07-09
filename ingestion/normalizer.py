import re
import unicodedata
from copy import deepcopy

# ==========================================================
# Constants
# ==========================================================

SEARCH_MODE = "search"
DISPLAY_MODE = "display"

ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06ED]"
)

ARABIC_LETTER_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ٲ": "ا",
    "ٳ": "ا",
    "ٵ": "ا",

    "ى": "ي",

    "ـ": "",
}

LEGAL_TERM_MAP = {

    "الماده": "المادة",
    "ماده": "المادة",

    "مكرراً": "مكرر",
    "مكررًا": "مكرر",
    "مكررا": "مكرر",
    "مكررة": "مكرر",

    "اولا": "أولاً",
    "اولاً": "أولاً",
    "أولا": "أولاً",
}

# ==========================================================
# Unicode
# ==========================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode representation.
    """

    return unicodedata.normalize(
        "NFKC",
        text
    )


# ==========================================================
# Invisible Characters
# ==========================================================

def remove_invisible_chars(text: str) -> str:
    """
    Remove hidden unicode characters while preserving
    new lines and tabs.
    """

    cleaned = []

    for ch in text:

        if ch in ("\n", "\t"):
            cleaned.append(ch)
            continue

        if unicodedata.category(ch).startswith("C"):
            continue

        cleaned.append(ch)

    return "".join(cleaned)


# ==========================================================
# Arabic Letters
# ==========================================================

def normalize_arabic_letters(text: str) -> str:
    """
    Safe Arabic normalization.

    Preserve:
        ة
        ؤ
        ئ
    """

    for old, new in ARABIC_LETTER_MAP.items():

        text = text.replace(old, new)

    return text


# ==========================================================
# Remove Diacritics
# ==========================================================

def remove_diacritics(text: str) -> str:

    return ARABIC_DIACRITICS.sub(
        "",
        text
    )


# ==========================================================
# Legal Terms
# ==========================================================

def normalize_legal_terms(text: str) -> str:
    """
    Normalize common OCR / legal writing variations.
    """

    for old, new in LEGAL_TERM_MAP.items():

        text = text.replace(old, new)

    return text


# ==========================================================
# Punctuation
# ==========================================================

def collapse_punctuation(text: str) -> str:

    text = re.sub(r"\.{2,}", ".", text)

    text = re.sub(r"،{2,}", "،", text)

    text = re.sub(r"؛{2,}", "؛", text)

    text = re.sub(r"؟{2,}", "؟", text)

    text = re.sub(r"!{2,}", "!", text)

    text = re.sub(r"-{2,}", "-", text)

    return text


# ==========================================================
# Whitespace
# ==========================================================

def normalize_whitespace(text: str) -> str:

    text = text.replace("\u00A0", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    return "\n".join(lines).strip()


# ==========================================================
# Normalize Single Text
# ==========================================================

def normalize_text(
    text: str,
    mode: str = SEARCH_MODE,
) -> str:
    """
    Normalize a single text.

    Parameters
    ----------
    display
        Minimal normalization.

    search
        Strong normalization for embeddings
        and retrieval.
    """

    if not text:
        return ""

    text = normalize_unicode(text)

    text = remove_invisible_chars(text)

    text = remove_diacritics(text)

    text = normalize_arabic_letters(text)

    text = collapse_punctuation(text)

    if mode == SEARCH_MODE:

        text = normalize_legal_terms(text)

    text = normalize_whitespace(text)

    return text


# ==========================================================
# Normalize Chunks
# ==========================================================

def normalize_chunks(
    chunks: list[dict],
) -> list[dict]:
    """
    Add embedding_text to every chunk.

    Original chunk_text is preserved.

    Pipeline:

        chunk_text
             │
             ▼
      normalize_text()
             │
             ▼
      embedding_text
    """

    normalized_chunks = []

    for chunk in chunks:

        item = deepcopy(chunk)

        item["embedding_text"] = normalize_text(
            item.get("chunk_text", ""),
            mode=SEARCH_MODE,
        )

        normalized_chunks.append(item)

    return normalized_chunks