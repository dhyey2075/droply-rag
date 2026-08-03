"""LangGraph nodes for document ingest."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.clients import get_gemini_client
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.ingest.loaders import extension_of, load_documents
from app.ingest.state import IngestState
from app.retrieval.embeddings import embed_texts
from app.retrieval.vectorstore import upsert_chunks


def validate(state: IngestState) -> dict[str, Any]:
    if not state.get("user_id") or not state.get("file_id"):
        raise HTTPException(
            status_code=400,
            detail="user_id and file_id are required",
        )
    if not state.get("file_url"):
        raise HTTPException(status_code=400, detail="file_url is required")

    ext = extension_of(state["file_name"], state.get("mime_or_type"))
    if not ext:
        raise HTTPException(status_code=400, detail="Could not determine file type")

    parsed = urlparse(state["file_url"])
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Invalid file_url")

    return {"ext": ext, "error": None}


def download(state: IngestState) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(state["file_url"])
            response.raise_for_status()
            return {"file_bytes": response.content}
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download file: {exc}",
        ) from exc


def parse(state: IngestState) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / f"upload.{state['ext']}"
        dest.write_bytes(state["file_bytes"])
        try:
            raw_docs = load_documents(dest, state["ext"], state["file_name"])
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse document: {exc}",
            ) from exc
    return {"raw_docs": raw_docs}


def chunk(state: IngestState) -> dict[str, Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(state["raw_docs"])
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks")

    prepared: list[tuple[int, str, dict[str, Any]]] = []
    for idx, doc in enumerate(chunks):
        metadata: dict[str, Any] = {
            **(doc.metadata or {}),
            "user_id": state["user_id"],
            "file_id": state["file_id"],
            "source": state["file_name"],
            "chunk_index": idx,
        }
        if not metadata.get("user_id") or not metadata.get("file_id"):
            raise HTTPException(
                status_code=500,
                detail="Refusing to store chunk without user_id and file_id metadata",
            )
        prepared.append((idx, doc.page_content, metadata))

    return {"prepared": prepared, "chunk_count": len(prepared)}


def embed_docs(state: IngestState) -> dict[str, Any]:
    texts = [content for _, content, _ in state["prepared"]]
    try:
        client = get_gemini_client()
        vectors = embed_texts(client, texts, task_type="RETRIEVAL_DOCUMENT")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Embedding failed: {exc}",
        ) from exc
    return {"vectors": vectors}


def upsert(state: IngestState) -> dict[str, Any]:
    upsert_chunks(
        state["user_id"],
        state["file_id"],
        state["prepared"],
        state["vectors"],
    )
    return {"chunk_count": len(state["prepared"])}
