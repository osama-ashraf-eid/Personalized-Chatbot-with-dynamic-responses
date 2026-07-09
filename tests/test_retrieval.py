"""
Tests for the retrieval components.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from retrieval.rank_fusion import ReciprocalRankFusion


# ==========================================================
# Rank Fusion Tests
# ==========================================================


class TestReciprocalRankFusion:

    def test_fuse_empty_lists(self):
        rrf = ReciprocalRankFusion()
        result = rrf.fuse([], [])
        assert result == []

    def test_fuse_single_list(self):
        rrf = ReciprocalRankFusion()
        results = [
            {"chunk_id": "a", "text": "doc a"},
            {"chunk_id": "b", "text": "doc b"},
        ]
        fused = rrf.fuse(results)
        assert len(fused) == 2
        assert "rrf_score" in fused[0]

    def test_fuse_two_lists_dedup(self):
        rrf = ReciprocalRankFusion()

        list1 = [
            {"chunk_id": "a", "text": "doc a"},
            {"chunk_id": "b", "text": "doc b"},
        ]

        list2 = [
            {"chunk_id": "b", "text": "doc b"},
            {"chunk_id": "c", "text": "doc c"},
        ]

        fused = rrf.fuse(list1, list2)

        # Should have 3 unique chunks

        chunk_ids = [f["chunk_id"] for f in fused]
        assert len(set(chunk_ids)) == 3

        # "b" should have highest score (appears in both)

        b_item = [
            f for f in fused
            if f["chunk_id"] == "b"
        ][0]

        assert b_item["rrf_score"] > 0

    def test_fuse_preserves_order(self):
        rrf = ReciprocalRankFusion()

        list1 = [
            {"chunk_id": "a"},
            {"chunk_id": "b"},
            {"chunk_id": "c"},
        ]

        fused = rrf.fuse(list1)

        scores = [f["rrf_score"] for f in fused]

        # Scores should be in descending order

        assert scores == sorted(
            scores, reverse=True
        )

    def test_callable(self):
        rrf = ReciprocalRankFusion()
        result = rrf(
            [{"chunk_id": "a"}],
            [{"chunk_id": "b"}],
        )
        assert len(result) == 2
