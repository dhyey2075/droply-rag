"""Embedding helpers (Gemini)."""

from __future__ import annotations

from fastapi import HTTPException
from google import genai
from google.genai import types

from app.config import EMBEDDING_DIMS, EMBEDDING_MODEL


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
