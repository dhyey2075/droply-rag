"""Export Droply RAG LangGraph diagrams as PNG images.

Usage (from droply-rag root, with venv active):

    python export_graphs.py

Writes:
    diagrams/chat_crag_graph.png
    diagrams/ingest_graph.png
"""

from __future__ import annotations

from pathlib import Path

from app.chat.graph import chat_graph
from app.ingest.graph import ingest_graph

OUT_DIR = Path(__file__).resolve().parent / "diagrams"


def export_png(compiled_graph, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    png_bytes = compiled_graph.get_graph().draw_mermaid_png()
    path.write_bytes(png_bytes)
    return path


def export_mermaid(compiled_graph, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_text(compiled_graph.get_graph().draw_mermaid(), encoding="utf-8")
    return path


def main() -> None:
    chat_png = export_png(chat_graph, "chat_crag_graph.png")
    ingest_png = export_png(ingest_graph, "ingest_graph.png")

    # Also save Mermaid source for docs / GitHub rendering
    chat_mmd = export_mermaid(chat_graph, "chat_crag_graph.mmd")
    ingest_mmd = export_mermaid(ingest_graph, "ingest_graph.mmd")

    print(f"Saved {chat_png}")
    print(f"Saved {ingest_png}")
    print(f"Saved {chat_mmd}")
    print(f"Saved {ingest_mmd}")


if __name__ == "__main__":
    main()
