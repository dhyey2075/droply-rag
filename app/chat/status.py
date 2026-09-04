"""User-facing SSE status copy for chat graph steps."""

from __future__ import annotations

from typing import Any

_STEP_MESSAGES: dict[str, str] = {
    "start": "Understanding your question…",
    "route_intent": "Checking whether your documents are needed…",
    "embed_query": "Searching your documents…",
    "retrieve": "Checking which passages matter…",
    "grade_documents": "Preparing an answer from your files…",
    "prepare_documents": "Writing an answer…",
    "refine_query": "Your documents were not a close match — searching the web…",
    "web_search": "Reading web results…",
    "prepare_web": "Writing an answer…",
    "build_prompt": "Finished... just putting cherry on the answer...",
    "generate": "Writing an answer…",
}


GRAPH_STATUS_STEPS = frozenset(
    {
        "route_intent",
        "embed_query",
        "retrieve",
        "grade_documents",
        "prepare_documents",
        "refine_query",
        "web_search",
        "prepare_web",
        "build_prompt",
    }
)


def status_message_for(step: str, state: dict[str, Any] | None = None) -> str:
    if step == "route_intent" and state and state.get("intent") == "direct":
        return "Replying…"
    if step == "grade_documents" and state and state.get("use_web"):
        custom = state.get("meta_message")
        if isinstance(custom, str) and custom.strip():
            return custom.strip()
        return _STEP_MESSAGES["refine_query"]
    return _STEP_MESSAGES.get(step, "Working on your answer…")
