"""Ingest graph state."""

from __future__ import annotations

from typing import Any, TypedDict


class IngestState(TypedDict, total=False):
    file_id: str
    user_id: str
    file_url: str
    file_name: str
    mime_or_type: str | None

    ext: str
    file_bytes: bytes
    raw_docs: list[Any]
    prepared: list[tuple[int, str, dict[str, Any]]]
    vectors: list[list[float]]
    chunk_count: int
    error: str | None
