from copy import deepcopy
from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


# Embedder

class Embedder:
    
    _model = None

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):

        self.model_name = model_name

        if Embedder._model is None:

            print(
                f"\nLoading embedding model: {model_name}"
            )

            Embedder._model = SentenceTransformer(
                model_name
            )

        self.model = Embedder._model

    # Validation

    @staticmethod
    def _validate_chunks(
        chunks: List[dict],
        text_field: str,
    ):

        if not chunks:
            raise ValueError(
                "Chunk list is empty."
            )

        for index, chunk in enumerate(chunks):

            if text_field not in chunk:

                raise KeyError(
                    f"Chunk #{index} "
                    f"does not contain '{text_field}'."
                )

    # Single Text

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate embedding for one text.
        """

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vector.tolist()

    # Batch

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

        return vectors.tolist()

    # Chunks

    def embed_chunks(
        self,
        chunks: List[dict],
        text_field: str = "embedding_text",
        batch_size: int = 32,
    ) -> List[dict]:
        """
        Add embeddings to chunks.

        Parameters
        ----------
        chunks
            List of chunk dictionaries.

        text_field
            Which field should be embedded.

            Examples
            --------
            embedding_text
            chunk_text

        batch_size
            Embedding batch size.
        """

        self._validate_chunks(
            chunks,
            text_field,
        )

        texts = [
            chunk[text_field]
            for chunk in chunks
        ]

        vectors = self.embed_texts(
            texts=texts,
            batch_size=batch_size,
        )

        embedded_chunks = []

        for chunk, vector in zip(chunks, vectors):

            item = deepcopy(chunk)

            item["embedding"] = vector

            embedded_chunks.append(item)

        return embedded_chunks

    # Utility

    def embedding_dimension(self) -> int:
        """
        Return embedding vector size.
        """

        return len(
            self.embed_text("اختبار")
        )