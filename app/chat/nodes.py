"""LangGraph nodes for Corrective RAG chat."""

from __future__ import annotations

from typing import Any

from app.chat.intent import classify_intent, heuristic_intent
from app.chat.state import ChatState
from app.clients import get_gemini_client, get_groq_client
from app.config import CHAT_MODEL, CRAG_ENABLED, CRAG_RELEVANCE_THRESHOLD, RETRIEVAL_TOP_K
from app.grading.relevance import filter_relevant_chunks, grade_chunk_relevance
from app.retrieval.embeddings import embed_texts
from app.retrieval.sources import (
    build_context,
    build_sources,
    build_web_context,
    build_web_sources,
)
from app.retrieval.vectorstore import retrieve_chunks
from app.search.web import web_search


def route_intent(state: ChatState) -> dict[str, Any]:
    """Skip retrieval for greetings, capability questions, and other chit-chat."""
    history = state.get("history") or []
    guessed = heuristic_intent(state["question"], history)
    if guessed is not None:
        return {"intent": guessed}
    try:
        intent = classify_intent(
            get_groq_client(),
            state["question"],
            history,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Intent classify failed, retrieving: {exc}")
        intent = "retrieve"
    return {"intent": intent}


def prepare_direct(_state: ChatState) -> dict[str, Any]:
    return {
        "mode": "chat",
        "sources": [],
        "context": "",
        "system_instruction": (
            "You are Droply, a file-library assistant. You can answer questions "
            "about the user's uploaded documents and, if those are not relevant, "
            "search the web. This turn does not use documents or web search — "
            "reply from conversation only. Be friendly and concise. "
            "If they ask what you can do, explain that they can ask about their "
            "files and you will search documents first, then the web if needed. "
            "Do not invent contents of their files. "
            "Format in Markdown when listing capabilities. "
            "Do not wrap the whole answer in a code fence."
        ),
        "fallback_text": None,
    }


def embed_query(state: ChatState) -> dict[str, Any]:
    client = get_gemini_client()
    vector = embed_texts(
        client,
        [state["question"]],
        task_type="RETRIEVAL_QUERY",
    )[0]
    return {"query_vector": vector, "error": None}


def retrieve(state: ChatState) -> dict[str, Any]:
    scoped = [fid.strip() for fid in state.get("file_ids") or [] if fid and fid.strip()]
    chunks = retrieve_chunks(
        state["user_id"],
        state["query_vector"],
        RETRIEVAL_TOP_K,
        file_ids=scoped or None,
    )
    return {"chunks": chunks}


def grade_documents(state: ChatState) -> dict[str, Any]:
    chunks = state.get("chunks") or []

    if not CRAG_ENABLED:
        return {
            "use_web": False,
            "relevant_chunks": chunks,
            "relevance": [],
            "meta_message": None,
        }

    if not chunks:
        return {
            "use_web": True,
            "relevant_chunks": [],
            "relevance": [],
            "meta_message": "Documents were not relevant enough; refining query for web search…",
        }

    try:
        groq_client = get_groq_client()
        scores = grade_chunk_relevance(groq_client, state["question"], chunks)
        relevance = [
            {
                "fileName": c.get("file_name") or "document",
                "chunkIndex": c.get("chunk_index"),
                "score": score,
            }
            for c, score in zip(chunks, scores, strict=True)
        ]
        relevant = filter_relevant_chunks(chunks, scores, CRAG_RELEVANCE_THRESHOLD)
        use_web = len(relevant) == 0
        return {
            "relevance": relevance,
            "relevant_chunks": relevant,
            "use_web": use_web,
            "meta_message": (
                "Documents were not relevant enough; refining query for web search…"
                if use_web
                else None
            ),
        }
    except Exception as grade_exc:  # noqa: BLE001
        # Fail open to document RAG if the grader errors
        print(f"CRAG grading failed, using documents: {grade_exc}")
        return {
            "use_web": False,
            "relevant_chunks": chunks,
            "relevance": [],
            "meta_message": None,
        }


def prepare_documents(state: ChatState) -> dict[str, Any]:
    relevant = state.get("relevant_chunks") or []
    sources = build_sources(state["user_id"], relevant)
    if not relevant:
        return {
            "mode": "documents",
            "sources": sources,
            "context": "",
            "system_instruction": "",
            "fallback_text": (
                "I couldn't find relevant information in your indexed documents "
                "for that question."
            ),
        }

    return {
        "mode": "documents",
        "sources": sources,
        "context": build_context(relevant),
        "system_instruction": (
            "You are Droply's document assistant. Answer using ONLY the provided "
            "context from the user's documents. If the context is insufficient, "
            "say you don't know based on the available documents. Be concise. "
            "When citing, use only the human-readable filename (e.g. Resume.pdf). "
            "Never mention file_id, chunk numbers, UUIDs, or internal IDs. "
            "Format in Markdown: headings, **bold**, and bullet/numbered lists. "
            "Do not wrap the whole answer in a code fence."
        ),
        "fallback_text": None,
    }


def refine_query(state: ChatState) -> dict[str, Any]:
    """Rewrite the user question into a self-contained, search-engine-friendly query."""
    question = (state.get("question") or "").strip()
    history = state.get("history") or []

    history_lines: list[str] = []
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "(none)"

    prompt = (
        "Rewrite the user's question into a single web search query.\n"
        "Requirements:\n"
        "- Be self-contained: resolve pronouns/ellipsis using recent chat history when useful\n"
        "- Prefer concrete keywords, entities, dates, and product/doc names\n"
        "- Keep it concise (about 5–16 words); no quotes, no markdown, no explanation\n"
        "- Do NOT invent facts that are not implied by the question or history\n"
        "- Output ONLY the refined search query text\n\n"
        f"Recent chat history:\n{history_block}\n\n"
        f"Current question: {question}\n\n"
        "Refined search query:"
    )

    try:
        groq_client = get_groq_client()
        completion = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write excellent web search queries. "
                        "Respond with only the query string."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        refined = (completion.choices[0].message.content or "").strip()
        refined = refined.strip().strip('"').strip("'")
        # Drop accidental labels like "Query: ..."
        for prefix in ("refined search query:", "search query:", "query:"):
            if refined.lower().startswith(prefix):
                refined = refined[len(prefix) :].strip()
        if not refined:
            refined = question
    except Exception as exc:  # noqa: BLE001
        print(f"Query refine failed, using original question: {exc}")
        refined = question

    return {
        "refined_query": refined or question,
        "meta_message": "Searching the web with a refined query…",
    }


def web_search_node(state: ChatState) -> dict[str, Any]:
    query = (state.get("refined_query") or state.get("question") or "").strip()
    results = web_search(query)
    return {"web_results": results, "mode": "web"}


def prepare_web(state: ChatState) -> dict[str, Any]:
    results = state.get("web_results") or []
    sources = build_web_sources(results)
    if not results:
        return {
            "mode": "web",
            "sources": sources,
            "context": "",
            "system_instruction": "",
            "fallback_text": (
                "I couldn't find relevant information in your documents "
                "or on the web for that question."
            ),
        }

    return {
        "mode": "web",
        "sources": sources,
        "context": build_web_context(results),
        "system_instruction": (
            "You are Droply's assistant in web-fallback mode (Corrective RAG). "
            "The user's documents were not relevant enough, so answer using ONLY "
            "the web search results below. Be concise and factual. "
            "Cite sources by page title. If results are insufficient, say so. "
            "Format in Markdown: headings, **bold**, and bullet/numbered lists. "
            "Do not wrap the whole answer in a code fence."
        ),
        "fallback_text": None,
    }


def build_prompt(state: ChatState) -> dict[str, Any]:
    if state.get("fallback_text"):
        return {"messages": []}

    messages: list[dict[str, str]] = [
        {"role": "system", "content": state.get("system_instruction") or ""},
    ]
    for turn in (state.get("history") or [])[-6:]:
        messages.append(
            {
                "role": "user" if turn["role"] == "user" else "assistant",
                "content": turn["content"],
            }
        )
    question = state["question"]
    context = (state.get("context") or "").strip()
    user_content = (
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        if context
        else f"Question: {question}\n\nAnswer:"
    )
    messages.append({"role": "user", "content": user_content})
    return {"messages": messages}
