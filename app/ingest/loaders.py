"""Document loading helpers for ingest."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document


def extension_of(file_name: str, mime_or_type: str | None) -> str:
    suffix = Path(file_name).suffix.lower().lstrip(".")
    if suffix:
        return suffix
    mime = (mime_or_type or "").lower()
    if "pdf" in mime:
        return "pdf"
    if "wordprocessingml" in mime or mime.endswith("docx"):
        return "docx"
    if "msword" in mime:
        return "doc"
    if "markdown" in mime:
        return "md"
    if "csv" in mime:
        return "csv"
    if "rtf" in mime:
        return "rtf"
    if "text/plain" in mime or mime == "txt":
        return "txt"
    return ""


def load_documents(path: Path, ext: str, source_name: str) -> list[Document]:
    if ext == "pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    elif ext in {"docx", "doc"}:
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
    elif ext in {"txt", "md", "markdown", "csv", "rtf"}:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {ext or 'unknown'}",
        )

    for doc in docs:
        doc.metadata = {**(doc.metadata or {}), "source": source_name}
    return docs
