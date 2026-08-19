"""Shared configuration loaded from environment."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIMS = 768
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
RETRIEVAL_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", "50"))
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
CRAG_ENABLED = os.getenv("CRAG_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
