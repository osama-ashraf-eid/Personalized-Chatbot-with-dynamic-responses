from typing import List

import chromadb

from core.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    ChromaDB Vector Store wrapper.

    Handles:
        • Collection management
        • Adding embedded chunks
        • Querying by embedding
        • Reset / count operations
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    # --------------------------------------------------
    # Add Chunks
    # --------------------------------------------------

    def add_chunks(
        self,
        chunks: List[dict],
        batch_size: int = 100,
    ) -> int:
        """
        Add embedded chunks to ChromaDB.

        Each chunk must have:
            • chunk_id
            • embedding_text
            • embedding
        """

        if not chunks:
            return 0

        total = len(chunks)
        added = 0

        for start in range(0, total, batch_size):

            batch = chunks[start:start + batch_size]

            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for chunk in batch:

                chunk_id = chunk["chunk_id"]

                ids.append(chunk_id)

                documents.append(
                    chunk.get("embedding_text", "")
                )

                embeddings.append(
                    chunk["embedding"]
                )

                # Store metadata (exclude large fields)

                metadata = {
                    key: value
                    for key, value in chunk.items()
                    if key not in (
                        "embedding",
                        "embedding_text",
                        "clean_text",
                        "text",
                    )
                    and isinstance(value, (str, int, float, bool))
                }

                metadatas.append(metadata)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            added += len(batch)

        logger.info(
            f"Added {added} chunks to collection "
            f"'{self.collection_name}'"
        )

        return added

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> dict:
        """
        Query ChromaDB by embedding vector.

        Parameters
        ----------
        embedding
            Query embedding vector.

        top_k
            Number of results.

        where
            Optional metadata filter.

        Returns
        -------
        dict
            ChromaDB query results with keys:
            ids, documents, metadatas, distances.
        """

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self) -> None:
        """
        Delete and recreate the collection.
        """

        self.client.delete_collection(
            self.collection_name
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

        logger.info(
            f"Reset collection '{self.collection_name}'"
        )

    # --------------------------------------------------
    # Count
    # --------------------------------------------------

    def count(self) -> int:
        """
        Return number of documents in the collection.
        """

        return self.collection.count()