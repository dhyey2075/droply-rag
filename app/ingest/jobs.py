"""Load a file row, run the ingest graph, and update indexing status."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import psycopg
from fastapi import HTTPException

from app.clients import get_db_url
from app.ingest.graph import ingest_graph

INDEXABLE_EXTENSIONS = {
    "pdf",
    "docx",
    "doc",
    "txt",
    "md",
    "markdown",
    "csv",
    "rtf",
}
INDEXABLE_MIME_SNIPPETS = (
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/rtf",
    "text/rtf",
)
IMAGE_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "svg",
    "bmp",
    "ico",
    "heic",
    "heif",
    "tiff",
    "tif",
    "avif",
}


def _extension(name: str | None) -> str | None:
    if not name or "." not in name:
        return None
    return name.rsplit(".", 1)[-1].lower()


def is_indexable_document(
    name: str | None, type_: str | None, is_folder: bool = False
) -> bool:
    if is_folder:
        return False
    ext = _extension(name)
    if ext and ext in IMAGE_EXTENSIONS:
        return False
    if ext and ext in INDEXABLE_EXTENSIONS:
        return True
    kind = (type_ or "").lower()
    if not kind or kind == "folder" or kind == "image" or kind.startswith("image/"):
        return False
    if kind in INDEXABLE_EXTENSIONS:
        return True
    return any(snippet in kind for snippet in INDEXABLE_MIME_SNIPPETS)


def exception_message(exc: BaseException) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            return detail
        return str(detail)
    return str(exc) or exc.__class__.__name__


def _connect():
    return psycopg.connect(get_db_url())


def load_file(file_id: str, user_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id::text,
                    user_id,
                    name,
                    type,
                    file_url,
                    is_trash,
                    is_folder,
                    indexing_status,
                    index_attempts
                FROM files
                WHERE id = %s::uuid AND user_id = %s
                """,
                (file_id, user_id),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "user_id": row[1],
        "name": row[2],
        "type": row[3],
        "file_url": row[4],
        "is_trash": row[5],
        "is_folder": row[6],
        "indexing_status": row[7],
        "index_attempts": row[8],
    }


def update_status(
    file_id: str,
    user_id: str,
    *,
    indexing_status: str,
    index_error: str | None = None,
    increment_attempts: bool = False,
    chunk_count: int | None = None,
    indexed_at: datetime | None = None,
) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            if increment_attempts:
                cur.execute(
                    """
                    UPDATE files
                    SET
                        indexing_status = %s,
                        index_error = %s,
                        index_attempts = COALESCE(index_attempts, 0) + 1,
                        updated_at = NOW()
                    WHERE id = %s::uuid AND user_id = %s
                    """,
                    (indexing_status, index_error, file_id, user_id),
                )
            elif chunk_count is not None:
                cur.execute(
                    """
                    UPDATE files
                    SET
                        indexing_status = %s,
                        index_error = %s,
                        chunk_count = %s,
                        indexed_at = %s,
                        updated_at = NOW()
                    WHERE id = %s::uuid AND user_id = %s
                    """,
                    (
                        indexing_status,
                        index_error,
                        chunk_count,
                        indexed_at,
                        file_id,
                        user_id,
                    ),
                )
            else:
                cur.execute(
                    """
                    UPDATE files
                    SET
                        indexing_status = %s,
                        index_error = %s,
                        updated_at = NOW()
                    WHERE id = %s::uuid AND user_id = %s
                    """,
                    (indexing_status, index_error, file_id, user_id),
                )
        conn.commit()


def emit_status(
    *,
    file_id: str,
    user_id: str,
    indexing_status: str,
    index_error: str | None = None,
    chunk_count: int | None = None,
    indexed_at: datetime | None = None,
) -> None:
    import os

    app_url = (
        os.getenv("APP_URL")
        or os.getenv("NEXT_PUBLIC_APP_URL")
        or "http://localhost:3000"
    ).rstrip("/")
    rag_key = os.getenv("RAG_INTERNAL_KEY")
    if not rag_key:
        return

    payload: dict[str, Any] = {
        "fileId": file_id,
        "userId": user_id,
        "indexingStatus": indexing_status,
        "indexError": index_error,
        "chunkCount": chunk_count,
        "indexedAt": indexed_at.isoformat() if indexed_at else None,
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{app_url}/api/files/indexing-stream",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {rag_key}",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        print(f"Failed to publish indexing status: {exc}")


def run_ingest(row: dict[str, Any]) -> int:
    result = ingest_graph.invoke(
        {
            "file_id": row["id"],
            "user_id": row["user_id"],
            "file_url": row["file_url"],
            "file_name": row["name"],
            "mime_or_type": row["type"],
        }
    )
    return int(result.get("chunk_count") or 0)
