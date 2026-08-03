"""pgvector retrieve + upsert."""

from __future__ import annotations

import json
from typing import Any

import psycopg
from fastapi import HTTPException

from app.clients import get_db_url


def retrieve_chunks(
    user_id: str,
    query_vector: list[float],
    k: int,
    file_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    vector_literal = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
    scoped_ids = [fid for fid in (file_ids or []) if fid and fid.strip()]
    db_url = get_db_url()
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            if scoped_ids:
                cur.execute(
                    """
                    SELECT
                        id::text,
                        file_id::text,
                        chunk_index,
                        content,
                        metadata,
                        embedding <=> %s::vector AS distance
                    FROM document_chunks
                    WHERE user_id = %s
                      AND file_id = ANY(%s::uuid[])
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, user_id, scoped_ids, vector_literal, k),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id::text,
                        file_id::text,
                        chunk_index,
                        content,
                        metadata,
                        embedding <=> %s::vector AS distance
                    FROM document_chunks
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, user_id, vector_literal, k),
                )
            rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        meta = row[4] if isinstance(row[4], dict) else {}
        results.append(
            {
                "id": row[0],
                "file_id": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "metadata": meta,
                "distance": float(row[5]) if row[5] is not None else None,
                "file_name": meta.get("source") or "document",
            }
        )
    return results


def upsert_chunks(
    user_id: str,
    file_id: str,
    prepared: list[tuple[int, str, dict[str, Any]]],
    vectors: list[list[float]],
) -> None:
    if len(vectors) != len(prepared):
        raise HTTPException(status_code=500, detail="Embedding count mismatch")

    db_url = get_db_url()
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE file_id = %s::uuid AND user_id = %s
                    """,
                    (file_id, user_id),
                )
                for (chunk_index, content, metadata), vector in zip(
                    prepared, vectors, strict=True
                ):
                    vector_literal = "[" + ",".join(str(float(x)) for x in vector) + "]"
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            id,
                            user_id,
                            file_id,
                            chunk_index,
                            content,
                            embedding,
                            metadata
                        ) VALUES (
                            gen_random_uuid(),
                            %s,
                            %s::uuid,
                            %s,
                            %s,
                            %s::vector,
                            %s::jsonb
                        )
                        """,
                        (
                            user_id,
                            file_id,
                            chunk_index,
                            content,
                            vector_literal,
                            json.dumps(metadata),
                        ),
                    )
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upsert vectors: {exc}",
        ) from exc
