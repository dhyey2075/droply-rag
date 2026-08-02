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
# CRAG: if every chunk scores below this (%), fall back to web search
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", "50"))
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
CRAG_ENABLED = os.getenv("CRAG_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


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
    # Optional scope: when set, retrieve only from these file_ids (still user-scoped).
    # Omit / empty = search all of the user's indexed documents.
    file_ids: list[str] = Field(default_factory=list)


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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def grade_chunk_relevance(
    groq_client: Groq,
    question: str,
    chunks: list[dict[str, Any]],
) -> list[float]:
    """Score each chunk 0–100 for relevance to the question (Corrective RAG)."""
    if not chunks:
        return []

    numbered: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = (chunk.get("content") or "").strip().replace("\n", " ")
        if len(text) > 500:
            text = text[:497] + "..."
        source = chunk.get("file_name") or "document"
        numbered.append(f"[{i}] ({source}) {text}")

    prompt = (
        "You are a retrieval relevance grader for Corrective RAG (CRAG).\n"
        "Score how relevant EACH document chunk is to answering the question.\n"
        "Return ONLY valid JSON with this exact shape:\n"
        '{"scores":[number, number, ...]}\n'
        "Each score must be from 0 to 100 (percent). "
        f"Provide exactly {len(chunks)} scores in the same order as the chunks.\n"
        "Guidelines: 80–100 strongly answers the question; 50–79 partially useful; "
        "0–49 mostly irrelevant or off-topic.\n\n"
        f"Question: {question}\n\n"
        "Chunks:\n"
        + "\n".join(numbered)
    )

    completion = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You output only compact JSON. No markdown fences.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = completion.choices[0].message.content or ""
    parsed = _extract_json_object(content)
    if not parsed or "scores" not in parsed:
        raise ValueError(f"Relevance grader returned unusable JSON: {content[:200]}")

    raw_scores = parsed["scores"]
    if not isinstance(raw_scores, list) or len(raw_scores) != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} scores, got "
            f"{len(raw_scores) if isinstance(raw_scores, list) else type(raw_scores)}"
        )

    scores: list[float] = []
    for value in raw_scores:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid relevance score: {value}") from exc
        scores.append(max(0.0, min(100.0, score)))
    return scores


def web_search(query: str, max_results: int = WEB_SEARCH_RESULTS) -> list[dict[str, str]]:
    """CRAG web fallback via DDGS (no API key required)."""
    try:
        from ddgs import DDGS
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail="Web search package 'ddgs' is not installed",
        ) from exc

    try:
        results = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Web search failed: {exc}",
        ) from exc

    cleaned: list[dict[str, str]] = []
    for item in results:
        title = (item.get("title") or "Web result").strip()
        href = (item.get("href") or item.get("link") or "").strip()
        body = (item.get("body") or item.get("snippet") or "").strip()
        if not href and not body:
            continue
        cleaned.append({"title": title, "href": href, "body": body})
    return cleaned


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
            scoped_file_ids = [
                fid.strip() for fid in payload.file_ids if fid and fid.strip()
            ]
            chunks = retrieve_chunks(
                payload.user_id,
                query_vec,
                RETRIEVAL_TOP_K,
                file_ids=scoped_file_ids or None,
            )

            mode: Literal["documents", "web"] = "documents"
            relevance: list[dict[str, Any]] = []
            sources: list[dict[str, Any]] = []
            context = ""
            system_instruction = ""

            # --- Corrective RAG (CRAG) ---
            # Grade retrieved chunks; if all < threshold, answer via web search.
            use_web = False
            relevant_chunks = chunks

            if CRAG_ENABLED:
                if not chunks:
                    use_web = True
                else:
                    try:
                        scores = grade_chunk_relevance(groq_client, question, chunks)
                        relevance = [
                            {
                                "fileName": c.get("file_name") or "document",
                                "chunkIndex": c.get("chunk_index"),
                                "score": score,
                            }
                            for c, score in zip(chunks, scores, strict=True)
                        ]
                        relevant_chunks = [
                            c
                            for c, score in zip(chunks, scores, strict=True)
                            if score >= CRAG_RELEVANCE_THRESHOLD
                        ]
                        use_web = len(relevant_chunks) == 0
                    except Exception as grade_exc:  # noqa: BLE001
                        # Fail open to document RAG if the grader errors
                        print(f"CRAG grading failed, using documents: {grade_exc}")
                        use_web = False
                        relevant_chunks = chunks

            if use_web:
                mode = "web"
                yield sse(
                    "meta",
                    {
                        "mode": mode,
                        "sources": [],
                        "relevance": relevance,
                        "message": "Documents were not relevant enough; searching the web…",
                    },
                )
                web_results = web_search(question, max_results=WEB_SEARCH_RESULTS)
                sources = build_web_sources(web_results)
                yield sse(
                    "meta",
                    {
                        "mode": mode,
                        "sources": sources,
                        "relevance": relevance,
                    },
                )

                if not web_results:
                    fallback = (
                        "I couldn't find relevant information in your documents "
                        "or on the web for that question."
                    )
                    yield sse("token", {"text": fallback})
                    yield sse("done", {"mode": mode, "sources": sources, "relevance": relevance})
                    return

                context = build_web_context(web_results)
                system_instruction = (
                    "You are Droply's assistant in web-fallback mode (Corrective RAG). "
                    "The user's documents were not relevant enough, so answer using ONLY "
                    "the web search results below. Be concise and factual. "
                    "Cite sources by page title. If results are insufficient, say so."
                )
            else:
                mode = "documents"
                sources = build_sources(payload.user_id, relevant_chunks)
                yield sse(
                    "meta",
                    {
                        "mode": mode,
                        "sources": sources,
                        "relevance": relevance,
                    },
                )

                if not relevant_chunks:
                    fallback = (
                        "I couldn't find relevant information in your indexed documents "
                        "for that question."
                    )
                    yield sse("token", {"text": fallback})
                    yield sse("done", {"mode": mode, "sources": sources, "relevance": relevance})
                    return

                context = build_context(relevant_chunks)
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

            yield sse(
                "done",
                {"mode": mode, "sources": sources, "relevance": relevance},
            )
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
