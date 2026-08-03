"""Web search fallback (DDGS)."""

from __future__ import annotations

from fastapi import HTTPException

from app.config import WEB_SEARCH_RESULTS


def web_search(query: str, max_results: int = WEB_SEARCH_RESULTS) -> list[dict[str, str]]:
    """CRAG web fallback via DDGS (no API key required)."""
    try:
        from ddgs import DDGS
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail="Web search package 'ddgs' is not installed",
        ) from exc

    try:
        results = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Web search failed: {exc}",
        ) from exc

    cleaned: list[dict[str, str]] = []
    for item in results:
        title = (item.get("title") or "Web result").strip()
        href = (item.get("href") or item.get("link") or "").strip()
        body = (item.get("body") or item.get("snippet") or "").strip()
        if not href and not body:
            continue
        cleaned.append({"title": title, "href": href, "body": body})
    return cleaned
