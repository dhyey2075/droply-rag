"""Droply RAG: ingest + library-wide chat (user-scoped retrieve → generate)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from urllib.parse import urlparse

import httpx
import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from groq import Groq
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Droply RAG", version="0.2.0")

EMBEDDING_DIMS = 768
EMBEDDING_MODEL = "gemini-embedding-001"
# Chat via Groq (Gemini free-tier generate quota is exhausted)
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
RETRIEVAL_TOP_K = int(os.getenv("RAG_TOP_K", "8"))


class IngestRequest(BaseModel):
    file_id: str
    user_id: str
    file_url: str
    file_name: str
    mime_or_type: str | None = None


class IngestResponse(BaseModel):
    status: str = "ok"
    chunk_count: int


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    user_id: str
    question: str = Field(min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)


def require_internal_key(
    authorization: str | None = Header(default=None),
) -> None:
    expected = os.getenv("RAG_INTERNAL_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="RAG_INTERNAL_KEY not configured")
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured",
        )
    return api_key


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured",
        )
    return api_key


def get_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    return url


def extension_of(file_name: str, mime_or_type: str | None) -> str:
    suffix = Path(file_name).suffix.lower().lstrip(".")
    if suffix:
        return suffix
    mime = (mime_or_type or "").lower()
    if "pdf" in mime:
        return "pdf"
    if "wordprocessingml" in mime or mime.endswith("docx"):
        return "docx"
    if "msword" in mime:
        return "doc"
    if "markdown" in mime:
        return "md"
    if "csv" in mime:
        return "csv"
    if "rtf" in mime:
        return "rtf"
    if "text/plain" in mime or mime == "txt":
        return "txt"
    return ""


def load_documents(path: Path, ext: str, source_name: str) -> list[Document]:
    if ext == "pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    elif ext in {"docx", "doc"}:
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
    elif ext in {"txt", "md", "markdown", "csv", "rtf"}:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {ext or 'unknown'}",
        )

    for doc in docs:
        doc.metadata = {**(doc.metadata or {}), "source": source_name}
    return docs


def embed_texts(
    client: genai.Client,
    texts: list[str],
    *,
    task_type: str,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMS,
                task_type=task_type,
            ),
        )
        values = result.embeddings[0].values
        if not values or len(values) != EMBEDDING_DIMS:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Unexpected embedding size "
                    f"{0 if not values else len(values)}, expected {EMBEDDING_DIMS}"
                ),
            )
        vectors.append(list(values))
    return vectors


def retrieve_chunks(user_id: str, query_vector: list[float], k: int) -> list[dict[str, Any]]:
    vector_literal = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
    db_url = get_db_url()
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
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


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    payload: IngestRequest,
    _: None = Depends(require_internal_key),
) -> IngestResponse:
    if not payload.user_id or not payload.file_id:
        raise HTTPException(
            status_code=400,
            detail="user_id and file_id are required",
        )
    if not payload.file_url:
        raise HTTPException(status_code=400, detail="file_url is required")

    ext = extension_of(payload.file_name, payload.mime_or_type)
    if not ext:
        raise HTTPException(status_code=400, detail="Could not determine file type")

    parsed = urlparse(payload.file_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Invalid file_url")

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / f"upload.{ext}"
        try:
            with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                response = client.get(payload.file_url)
                response.raise_for_status()
                dest.write_bytes(response.content)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download file: {exc}",
            ) from exc

        try:
            raw_docs = load_documents(dest, ext, payload.file_name)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse document: {exc}",
            ) from exc

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(raw_docs)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no text chunks")

        prepared: list[tuple[int, str, dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            metadata: dict[str, Any] = {
                **(chunk.metadata or {}),
                "user_id": payload.user_id,
                "file_id": payload.file_id,
                "source": payload.file_name,
                "chunk_index": idx,
            }
            if not metadata.get("user_id") or not metadata.get("file_id"):
                raise HTTPException(
                    status_code=500,
                    detail="Refusing to store chunk without user_id and file_id metadata",
                )
            prepared.append((idx, chunk.page_content, metadata))

        api_key = get_gemini_api_key()
        texts = [content for _, content, _ in prepared]
        try:
            client = genai.Client(api_key=api_key)
            vectors = embed_texts(client, texts, task_type="RETRIEVAL_DOCUMENT")
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"Embedding failed: {exc}",
            ) from exc

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
                        (payload.file_id, payload.user_id),
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
                                payload.user_id,
                                payload.file_id,
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

    return IngestResponse(status="ok", chunk_count=len(prepared))


@app.post("/chat")
async def chat(
    payload: ChatRequest,
    _: None = Depends(require_internal_key),
) -> StreamingResponse:
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    async def event_stream() -> AsyncIterator[str]:
        try:
            gemini_client = genai.Client(api_key=get_gemini_api_key())
            groq_client = Groq(api_key=get_groq_api_key())

            query_vec = embed_texts(
                gemini_client,
                [question],
                task_type="RETRIEVAL_QUERY",
            )[0]
            chunks = retrieve_chunks(payload.user_id, query_vec, RETRIEVAL_TOP_K)
            sources = build_sources(payload.user_id, chunks)
            yield sse("meta", {"sources": sources})

            if not chunks:
                fallback = (
                    "I couldn't find relevant information in your indexed documents "
                    "for that question."
                )
                yield sse("token", {"text": fallback})
                yield sse("done", {"sources": sources})
                return

            context = build_context(chunks)
            system_instruction = (
                "You are Droply's document assistant. Answer using ONLY the provided "
                "context from the user's documents. If the context is insufficient, "
                "say you don't know based on the available documents. Be concise. "
                "When citing, use only the human-readable filename (e.g. Resume.pdf). "
                "Never mention file_id, chunk numbers, UUIDs, or internal IDs."
            )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_instruction},
            ]
            for turn in payload.history[-6:]:
                messages.append(
                    {
                        "role": "user" if turn.role == "user" else "assistant",
                        "content": turn.content,
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Answer:"
                    ),
                }
            )

            stream = groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield sse("token", {"text": delta})

            yield sse("done", {"sources": sources})
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Chat failed"
            yield sse("error", {"message": detail})
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"message": f"Chat failed: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
