"""Droply RAG FastAPI shell — LangGraph pipelines live under app/."""

from __future__ import annotations

from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.auth import require_internal_key
from app.chat.graph import chat_graph
from app.clients import get_groq_client
from app.config import CHAT_MODEL
from app.ingest.graph import ingest_graph
from app.models import ChatRequest, IngestRequest, IngestResponse
from app.sse import sse

app = FastAPI(title="Droply RAG", version="0.3.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    payload: IngestRequest,
    _: None = Depends(require_internal_key),
) -> IngestResponse:
    result = ingest_graph.invoke(
        {
            "file_id": payload.file_id,
            "user_id": payload.user_id,
            "file_url": payload.file_url,
            "file_name": payload.file_name,
            "mime_or_type": payload.mime_or_type,
        }
    )
    return IngestResponse(status="ok", chunk_count=int(result.get("chunk_count") or 0))


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
            history = [
                {"role": turn.role, "content": turn.content}
                for turn in payload.history
            ]
            # Run CRAG graph through build_prompt (control flow + context prep).
            final: dict[str, Any] = await chat_graph.ainvoke(
                {
                    "user_id": payload.user_id,
                    "question": question,
                    "history": history,
                    "file_ids": payload.file_ids or [],
                }
            )

            mode = final.get("mode") or "documents"
            sources = final.get("sources") or []
            relevance = final.get("relevance") or []
            meta_message = final.get("meta_message")

            # Mirror prior UX: announce web fallback before sources when needed.
            if mode == "web" and meta_message:
                yield sse(
                    "meta",
                    {
                        "mode": mode,
                        "sources": [],
                        "relevance": relevance,
                        "message": meta_message,
                    },
                )

            yield sse(
                "meta",
                {
                    "mode": mode,
                    "sources": sources,
                    "relevance": relevance,
                    **({"message": meta_message} if meta_message and mode != "web" else {}),
                },
            )

            fallback = final.get("fallback_text")
            if fallback:
                yield sse("token", {"text": fallback})
                yield sse(
                    "done",
                    {"mode": mode, "sources": sources, "relevance": relevance},
                )
                return

            messages = final.get("messages") or []
            if not messages:
                yield sse(
                    "token",
                    {"text": "I couldn't generate an answer right now."},
                )
                yield sse(
                    "done",
                    {"mode": mode, "sources": sources, "relevance": relevance},
                )
                return

            groq_client = get_groq_client()
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
