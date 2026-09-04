"""HTTP request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    user_id: str
    question: str = Field(min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    # Optional scope: when set, retrieve only from these file_ids (still user-scoped).
    file_ids: list[str] = Field(default_factory=list)
