"""Decide whether a question needs document/web retrieval."""

from __future__ import annotations

import re
from typing import Literal

from groq import Groq

from app.chat.state import HistoryTurn
from app.config import CHAT_MODEL
from app.grading.relevance import extract_json_object

Intent = Literal["direct", "retrieve"]

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[.!?,'\"]+$")

# Whole-message greetings / thanks — not "hello, summarize my resume".
_GREETING = re.compile(
    r"^(hi+|hello|hey+|heya|hiya|yo|howdy|sup|"
    r"good\s+(morning|afternoon|evening|night)|"
    r"thanks|thank\s+you|thx|ty|"
    r"ok|okay|k|"
    r"bye|goodbye|see\s+ya|cheers)$",
    re.I,
)

_SMALLTALK = re.compile(
    r"^(how\s+are\s+you|hows\s+it\s+going|how\s+is\s+it\s+going|"
    r"whats?\s+up|what\s+is\s+up)$",
    re.I,
)

_CAPABILITY = re.compile(
    r"^(what\s+can\s+you\s+do|what\s+do\s+you\s+do|who\s+are\s+you|"
    r"what\s+are\s+you|how\s+do\s+you\s+work|how\s+does\s+(this|droply)\s+work|"
    r"what\s+is\s+droply|help|help\s+me)$",
    re.I,
)

_FOLLOW_UP = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|please|go\s+ahead|continue|do\s+it)$",
    re.I,
)


def _normalize(text: str) -> str:
    cleaned = _WS.sub(" ", (text or "").strip().lower())
    cleaned = _PUNCT.sub("", cleaned).strip()
    return cleaned


def heuristic_intent(question: str, history: list[HistoryTurn] | None = None) -> Intent | None:
    """Return a route when obvious; None means use the LLM classifier."""
    q = _normalize(question)
    if not q:
        return "direct"

    if history and _FOLLOW_UP.match(q):
        return "retrieve"

    if _GREETING.match(q) or _SMALLTALK.match(q) or _CAPABILITY.match(q):
        return "direct"

    return None


def classify_intent(
    groq_client: Groq,
    question: str,
    history: list[HistoryTurn] | None = None,
) -> Intent:
    guessed = heuristic_intent(question, history)
    if guessed is not None:
        return guessed

    history_lines: list[str] = []
    for turn in (history or [])[-4:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content[:400]}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    prompt = (
        "Classify whether answering the user requires searching their documents "
        "or the public web.\n"
        'Return ONLY JSON: {"intent":"direct"} or {"intent":"retrieve"}\n'
        "direct: greetings, thanks, small talk, questions about you/Droply "
        "(what you can do, how you work), or anything you can answer from "
        "general knowledge without looking anything up.\n"
        "retrieve: needs facts from the user's files, a specific document, "
        "current/web information, or a follow-up that continues a document/web answer.\n"
        "If unsure, use retrieve.\n\n"
        f"Recent chat:\n{history_block}\n\n"
        f"Current message: {question.strip()}\n"
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
    parsed = extract_json_object(completion.choices[0].message.content or "")
    intent = ""
    if parsed:
        raw = parsed.get("intent") or parsed.get("route") or ""
        intent = str(raw).strip().lower()
    if intent in {"direct", "chat", "skip"}:
        return "direct"
    return "retrieve"
