"""Compiled linear ingest graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.ingest.nodes import chunk, download, embed_docs, parse, upsert, validate
from app.ingest.state import IngestState


def build_ingest_graph():
    graph = StateGraph(IngestState)

    graph.add_node("validate", validate)
    graph.add_node("download", download)
    graph.add_node("parse", parse)
    graph.add_node("chunk", chunk)
    graph.add_node("embed_docs", embed_docs)
    graph.add_node("upsert", upsert)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "download")
    graph.add_edge("download", "parse")
    graph.add_edge("parse", "chunk")
    graph.add_edge("chunk", "embed_docs")
    graph.add_edge("embed_docs", "upsert")
    graph.add_edge("upsert", END)

    return graph.compile()


ingest_graph = build_ingest_graph()
