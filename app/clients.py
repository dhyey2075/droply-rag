"""API clients and env accessors."""

from __future__ import annotations

import os

from fastapi import HTTPException
from google import genai
from groq import Groq


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured",
        )
    return api_key


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured",
        )
    return api_key


def get_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    return url


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_gemini_api_key())


def get_groq_client() -> Groq:
    return Groq(api_key=get_groq_api_key())
