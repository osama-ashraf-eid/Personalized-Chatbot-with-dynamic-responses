"""
Tests for the FastAPI application.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from fastapi.testclient import TestClient

from main import app


# ==========================================================
# Test Client
# ==========================================================

client = TestClient(app)


# ==========================================================
# Health Tests
# ==========================================================


class TestHealth:

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "Egyptian Legal RAG"

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data


# ==========================================================
# API Schema Tests
# ==========================================================


class TestSchemas:

    def test_chat_empty_question(self):
        """
        Sending empty question should return 422.
        """
        response = client.post(
            "/api/chat",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_search_empty_query(self):
        """
        Sending empty query should return 422.
        """
        response = client.post(
            "/api/search",
            json={"query": ""},
        )
        assert response.status_code == 422
