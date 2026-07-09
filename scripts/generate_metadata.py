import json
import re
from pathlib import Path

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
METADATA_DIR = BASE_DIR / "metadata"

METADATA_DIR.mkdir(exist_ok=True)

# Regex Patterns
BOOK_PATTERN = re.compile(
    r"^الكتاب\s+.+"
)

CHAPTER_PATTERN = re.compile(
    r"^الباب\s+.+"
)

SECTION_PATTERN = re.compile(
    r"^الفصل\s+.+"
)

ARTICLE_PATTERN = re.compile(
    r"^المادة\s+([0-9]+(?:\s*مكرر(?:اً)?)*(?:\s*\([^)]+\))?)",
    re.UNICODE
)
# Helper Functions


def clean_line(text: str) -> str:
    """
    Remove extra spaces.
    """
    return re.sub(r"\s+", " ", text.strip())


def extract_law_name(lines):
    """
    Search first lines for the law title.
    """
    for line in lines[:40]:

        line = clean_line(line)

        if line.startswith("قانون"):
            return line

    return ""


def is_empty(line):
    return len(clean_line(line)) == 0


def normalize_text(text: str):

    text = text.replace("\u200f", "")
    text = text.replace("\u200e", "")
    text = text.replace("\ufeff", "")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def word_count(text):

    return len(text.split())


# Metadata Generator
def generate_metadata(txt_path: Path):

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    lines = [line.rstrip("\n") for line in raw_lines]

    law_name = extract_law_name(lines)

    metadata = []

    current_book = ""
    current_chapter = ""
    current_section = ""

    article_id = 1

    article_number = None
    article_title = None
    article_start = None
    article_text = []

    started_real_law = False



    for i, raw in enumerate(lines):

        line = clean_line(raw)

        if is_empty(line):
            continue

        # Skip law introduction until first Book/Chapter/Section
        if not started_real_law:

            if (
                BOOK_PATTERN.match(line)
                or CHAPTER_PATTERN.match(line)
                or SECTION_PATTERN.match(line)
            ):
                started_real_law = True
            else:
                continue

        # Book

        if BOOK_PATTERN.match(line):

            current_book = line

            # New Book -> reset chapter & section
            current_chapter = ""
            current_section = ""

            continue

        # Chapter
        if CHAPTER_PATTERN.match(line):

            current_chapter = line

            # New Chapter -> reset section
            current_section = ""

            continue

        # Section
        if SECTION_PATTERN.match(line):

            current_section = line

            continue

        # Article Detection
        article_match = ARTICLE_PATTERN.match(line)

        if article_match:

            # Save previous article
            if article_number is not None:

                text = normalize_text(
                    "\n".join(article_text)
                )

                metadata.append(
                    {
                        "id": article_id,
                        "article_key": (
                                        f"{txt_path.stem}_{article_number}_{article_id}"
                                        ),

                        "law": txt_path.stem,
                        "law_name": law_name,
                        "source": txt_path.name,

                        "book": current_book,
                        "chapter": current_chapter,
                        "section": current_section,

                        "article_number": article_number,
                        "article_label": f"المادة {article_number}",
                        "title": article_title,

                        "start_line": article_start + 1,
                        "end_line": i,

                        "word_count": word_count(text),
                        "text_length": len(text),

                        "text": text
                    }
                )

                article_id += 1

            article_number = article_match.group(1).strip()
            article_title = line
            article_start = i
            article_text = [line]

            continue

        # Collect article text
        if article_number is not None:

            article_text.append(line)

    # Save Last Articl
    if article_number is not None:

        text = normalize_text(
            "\n".join(article_text)
        )

        metadata.append(
            {
                "id": article_id,
                "article_key": (
                                  f"{txt_path.stem}_{article_number}_{article_id}"
                               ),

                "law": txt_path.stem,
                "law_name": law_name,
                "source": txt_path.name,
                "book": current_book,
                "chapter": current_chapter,
                "section": current_section,

                "article_number": article_number,
                "article_label": f"المادة {article_number}",
                "title": article_title,

                "start_line": article_start + 1,
                "end_line": len(lines),

                "word_count": word_count(text),
                "text_length": len(text),

                "text": text
            }
        )

    # Sort Article
    metadata.sort(key=lambda x: x["id"])

    # Save JSO
    output_path = METADATA_DIR / f"{txt_path.stem}_metadata.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("=" * 60)
    print(f"✓ {txt_path.name}")
    print(f"Law Name  : {law_name}")
    print(f"Articles  : {len(metadata)}")
    print(f"Saved To  : {output_path}")
    print("=" * 60)
    print()


# Main

def main():

    txt_files = sorted(DATA_DIR.glob("*.txt"))

    if not txt_files:
        print("No TXT files found inside /data")
        return

    print()
    print("=" * 70)
    print("Generating Metadata Files")
    print("=" * 70)
    print()

    for txt_file in txt_files:

        try:
            generate_metadata(txt_file)

        except Exception as e:
            print(f"❌ Error processing {txt_file.name}")
            print(e)
            print()

    print("=" * 70)
    print("Finished Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()