"""
Loaders for document graph and knowledge graph from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx


def load_document_graph(path: Path):
    """
    Load DocumentGraph from JSON file.
    Returns a DocumentGraph instance (from vectorizerE).
    """
    from vectorizerE import DocumentGraph

    doc_graph = DocumentGraph()
    doc_graph.load(path)
    return doc_graph


def load_knowledge_graph(path: Path) -> Optional[nx.MultiDiGraph]:
    """
    Load knowledge graph from JSON file.
    Returns nx.MultiDiGraph or None if file does not exist.
    """
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    graph = nx.MultiDiGraph()
    for node in payload.get("nodes", []):
        node_id = node.pop("id")
        graph.add_node(node_id, **node)
    for edge in payload.get("edges", []):
        source = edge.pop("source")
        target = edge.pop("target")
        graph.add_edge(source, target, **edge)
    return graph
