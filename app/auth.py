"""Internal API auth."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_internal_key(
    authorization: str | None = Header(default=None),
) -> None:
    expected = os.getenv("RAG_INTERNAL_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="RAG_INTERNAL_KEY not configured")
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")
