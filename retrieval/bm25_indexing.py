from pathlib import Path
import pickle
from typing import List

from rank_bm25 import BM25Okapi

from config import (
    BM25_INDEX_FILE,
    BM25_DOCUMENTS_FILE,
)


class BM25Index:
    """
    BM25 Index Manager.

    Responsible for:
        • Building BM25 index
        • Saving index
        • Loading index
        • Managing associated documents

    It does NOT perform retrieval.
    Retrieval is handled by SparseRetriever.
    """

    def __init__(self):

        self.index: BM25Okapi | None = None

        self.documents: List[dict] = []

    # --------------------------------------------------
    # Build
    # --------------------------------------------------

    def build(
        self,
        chunks: List[dict],
        text_field: str = "embedding_text",
    ) -> None:
        """
        Build BM25 index from chunks.
        """

        if not chunks:
            raise ValueError(
                "Chunks list is empty."
            )

        corpus = []

        self.documents = chunks

        for chunk in chunks:

            if text_field not in chunk:

                raise KeyError(
                    f"Chunk '{chunk.get('chunk_id')}' "
                    f"does not contain '{text_field}'."
                )

            corpus.append(
                chunk[text_field].split()
            )

        self.index = BM25Okapi(corpus)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save(self) -> None:
        """
        Save BM25 index and documents.
        """

        if self.index is None:

            raise ValueError(
                "BM25 index has not been built."
            )

        BM25_INDEX_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            BM25_INDEX_FILE,
            "wb",
        ) as f:

            pickle.dump(
                self.index,
                f,
            )

        with open(
            BM25_DOCUMENTS_FILE,
            "wb",
        ) as f:

            pickle.dump(
                self.documents,
                f,
            )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    def load(self) -> None:
        """
        Load BM25 index and documents.
        """

        if not self.exists():

            raise FileNotFoundError(
                "BM25 index files were not found."
            )

        with open(
            BM25_INDEX_FILE,
            "rb",
        ) as f:

            self.index = pickle.load(f)

        with open(
            BM25_DOCUMENTS_FILE,
            "rb",
        ) as f:

            self.documents = pickle.load(f)

    # --------------------------------------------------
    # Exists
    # --------------------------------------------------

    @staticmethod
    def exists() -> bool:
        """
        Check whether BM25 files exist.
        """

        return (
            BM25_INDEX_FILE.exists()
            and BM25_DOCUMENTS_FILE.exists()
        )

    # --------------------------------------------------
    # Build Or Load
    # --------------------------------------------------

    def build_or_load(
        self,
        chunks: List[dict],
        text_field: str = "embedding_text",
    ) -> None:
        """
        Load an existing BM25 index if available.
        Otherwise build a new one.
        """

        if self.exists():

            self.load()

            return

        self.build(
            chunks=chunks,
            text_field=text_field,
        )

        self.save()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """
        Check whether BM25 index is ready.
        """

        return (
            self.index is not None
            and len(self.documents) > 0
        )