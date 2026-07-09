import re
from copy import deepcopy


def clean_text(text: str) -> str:
    """
    Clean article text before chunking and embedding.
    """

    if not text:
        return ""

    # Remove invisible Unicode characters
    
    text = (
        text
        .replace("\ufeff", "")   # BOM
        .replace("\u200f", "")   # RTL Mark
        .replace("\u200e", "")   # LTR Mark
        .replace("\u2066", "")   # LRI
        .replace("\u2067", "")   # RLI
        .replace("\u2068", "")   # FSI
        .replace("\u2069", "")   # PDI
    )

    # Normalize line endings

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")


    # Normalize spaces
    
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around new lines
    text = re.sub(r" *\n *", "\n", text)

    
    # Trim every line

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    
    # Remove excessive blank lines
    
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def clean_articles(articles):
    """
    Add clean_text field to every article.

    The original metadata remains unchanged.
    """

    cleaned_articles = []

    for article in articles:

        item = deepcopy(article)

        item["clean_text"] = clean_text(
            item.get("text", "")
        )

        cleaned_articles.append(item)

    return cleaned_articles