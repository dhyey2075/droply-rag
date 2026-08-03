"""CRAG relevance grading."""

from __future__ import annotations

import json
from typing import Any

from groq import Groq

from app.config import CHAT_MODEL, CRAG_RELEVANCE_THRESHOLD


def extract_json_object(text: str) -> dict[str, Any] | None:
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
    parsed = extract_json_object(content)
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


def filter_relevant_chunks(
    chunks: list[dict[str, Any]],
    scores: list[float],
    threshold: float = CRAG_RELEVANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    return [
        c
        for c, score in zip(chunks, scores, strict=True)
        if score >= threshold
    ]
