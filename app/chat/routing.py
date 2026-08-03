"""Conditional edges for the CRAG chat graph."""

from __future__ import annotations

from typing import Literal

from app.chat.state import ChatState


def route_after_grade(state: ChatState) -> Literal["refine_query", "prepare_documents"]:
    if state.get("use_web"):
        return "refine_query"
    return "prepare_documents"
