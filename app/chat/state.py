"""Chat / CRAG graph state."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class HistoryTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ChatState(TypedDict, total=False):
    # Inputs
    user_id: str
    question: str
    history: list[HistoryTurn]
    file_ids: list[str]

    # Working
    intent: Literal["direct", "retrieve"]
    query_vector: list[float]
    chunks: list[dict[str, Any]]
    relevance: list[dict[str, Any]]
    relevant_chunks: list[dict[str, Any]]
    use_web: bool
    refined_query: str
    web_results: list[dict[str, str]]
    mode: Literal["documents", "web", "chat"]
    sources: list[dict[str, Any]]
    context: str
    system_instruction: str
    messages: list[dict[str, str]]

    # Control / output
    fallback_text: str | None
    error: str | None
    meta_message: str | None
