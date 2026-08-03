"""Compiled Corrective RAG chat graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.chat.nodes import (
    build_prompt,
    embed_query,
    grade_documents,
    prepare_documents,
    prepare_web,
    refine_query,
    retrieve,
    web_search_node,
)
from app.chat.routing import route_after_grade
from app.chat.state import ChatState


def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("embed_query", embed_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("prepare_documents", prepare_documents)
    graph.add_node("refine_query", refine_query)
    graph.add_node("web_search", web_search_node)
    graph.add_node("prepare_web", prepare_web)
    graph.add_node("build_prompt", build_prompt)

    graph.add_edge(START, "embed_query")
    graph.add_edge("embed_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {
            "refine_query": "refine_query",
            "prepare_documents": "prepare_documents",
        },
    )
    graph.add_edge("refine_query", "web_search")
    graph.add_edge("web_search", "prepare_web")
    graph.add_edge("prepare_web", "build_prompt")
    graph.add_edge("prepare_documents", "build_prompt")
    graph.add_edge("build_prompt", END)

    return graph.compile()


chat_graph = build_chat_graph()
