"""Source badge builders for document and web modes."""

from __future__ import annotations

from typing import Any

import psycopg

from app.clients import get_db_url


def build_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("file_name") or "document"
        blocks.append(f"[{i}] Source: {source}\n{chunk['content']}")
    return "\n\n".join(blocks)


def enrich_file_urls(user_id: str, file_ids: list[str]) -> dict[str, dict[str, str]]:
    if not file_ids:
        return {}
    db_url = get_db_url()
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, name, file_url
                FROM files
                WHERE user_id = %s AND id = ANY(%s::uuid[])
                """,
                (user_id, file_ids),
            )
            rows = cur.fetchall()
    return {
        row[0]: {"fileName": row[1] or "document", "fileUrl": row[2] or ""}
        for row in rows
    }


def build_sources(user_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One clickable source badge per file (deduped)."""
    file_ids = list({chunk["file_id"] for chunk in chunks if chunk.get("file_id")})
    file_meta = enrich_file_urls(user_id, file_ids)

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        file_id = chunk["file_id"]
        if file_id in seen:
            continue
        seen.add(file_id)
        meta = file_meta.get(file_id, {})
        file_name = meta.get("fileName") or chunk.get("file_name") or "document"
        file_url = meta.get("fileUrl") or ""
        snippet = (chunk["content"] or "").strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        sources.append(
            {
                "fileId": file_id,
                "fileName": file_name,
                "fileUrl": file_url,
                "snippet": snippet,
            }
        )
    return sources


def build_web_context(results: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {item['title']}\nURL: {item['href']}\n{item['body']}"
        )
    return "\n\n".join(blocks)


def build_web_sources(results: list[dict[str, str]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for i, item in enumerate(results, start=1):
        snippet = item["body"].replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        sources.append(
            {
                "fileId": f"web-{i}",
                "fileName": item["title"] or f"Web result {i}",
                "fileUrl": item["href"],
                "snippet": snippet,
            }
        )
    return sources
