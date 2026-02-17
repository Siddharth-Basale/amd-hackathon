"""
Utilities for building a knowledge graph from extracted triples.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "networkx is required for knowledge graph construction. "
        "Install it with `pip install networkx`."
    ) from exc


def _entity_key(name: str) -> str:
    return name.strip().lower()


def build_graph(document_id: str, chunk_records: Iterable[Dict]) -> nx.MultiDiGraph:
    """Build a MultiDiGraph from chunk-level extraction records."""
    graph = nx.MultiDiGraph()
    entity_nodes: Dict[str, Dict] = {}

    for record in chunk_records:
        chunk_id = record.get("chunk_id")
        heading = record.get("heading")
        section_path = record.get("section_path")
        chunk_node_id = f"chunk:{chunk_id}"
        graph.add_node(
            chunk_node_id,
            type="chunk",
            chunk_id=chunk_id,
            heading=heading,
            section_path=section_path,
        )

        for entity in record.get("entities", []):
            name = entity.get("name")
            if not name:
                continue
            key = _entity_key(name)
            node_id = f"entity:{key}"
            if node_id not in entity_nodes:
                entity_nodes[node_id] = {
                    "type": "entity",
                    "name": name,
                    "entity_type": entity.get("type", "OTHER"),
                    "description": entity.get("description", ""),
                    "aliases": entity.get("aliases", []),
                    "documents": set([document_id]),
                    "source_chunks": set(),
                }
            entity_nodes[node_id]["source_chunks"].add(chunk_id)
            graph.add_edge(node_id, chunk_node_id, relation="mentions")

        for relation in record.get("relations", []):
            source = relation.get("source")
            target = relation.get("target")
            rel_type = relation.get("relation")
            if not source or not target or not rel_type:
                continue
            source_node = f"entity:{_entity_key(source)}"
            target_node = f"entity:{_entity_key(target)}"
            if source_node not in entity_nodes:
                entity_nodes[source_node] = {
                    "type": "entity",
                    "name": source,
                    "entity_type": "OTHER",
                    "description": "",
                    "aliases": [],
                    "documents": set([document_id]),
                    "source_chunks": set(),
                }
            if target_node not in entity_nodes:
                entity_nodes[target_node] = {
                    "type": "entity",
                    "name": target,
                    "entity_type": "OTHER",
                    "description": "",
                    "aliases": [],
                    "documents": set([document_id]),
                    "source_chunks": set(),
                }
            graph.add_edge(
                source_node,
                target_node,
                relation=rel_type,
                evidence=relation.get("evidence", ""),
                chunk_id=chunk_id,
            )

    for node_id, data in entity_nodes.items():
        graph.add_node(
            node_id,
            type="entity",
            name=data["name"],
            entity_type=data["entity_type"],
            description=data["description"],
            aliases=list(data["aliases"]),
            documents=list(data["documents"]),
            source_chunks=list(data["source_chunks"]),
        )

    return graph


def graph_to_dict(graph: nx.MultiDiGraph) -> Dict:
    """Serialize graph to a dict suitable for JSON export."""
    nodes = []
    for node_id, attributes in graph.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                **attributes,
            }
        )

    edges = []
    for u, v, attributes in graph.edges(data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                **attributes,
            }
        )

    return {"nodes": nodes, "edges": edges}


def aggregate_entities(graph: nx.MultiDiGraph) -> List[Dict]:
    """Create a flat list of entity metadata with aggregated references."""
    entities = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "entity":
            continue
        entities.append(
            {
                "id": node_id,
                "name": data.get("name"),
                "type": data.get("entity_type"),
                "description": data.get("description"),
                "aliases": data.get("aliases", []),
                "source_chunks": data.get("source_chunks", []),
                "documents": data.get("documents", []),
            }
        )
    return entities


def aggregate_relations(graph: nx.MultiDiGraph) -> List[Dict]:
    """Create a deduplicated list of relations from the graph edges."""
    relation_map: Dict[Tuple[str, str, str], Dict] = {}
    for source, target, data in graph.edges(data=True):
        relation = data.get("relation")
        if not relation:
            continue
        key = (source, relation, target)
        if key not in relation_map:
            relation_map[key] = {
                "source": source,
                "relation": relation,
                "target": target,
                "evidence": [],
                "chunk_ids": [],
            }
        evidence = data.get("evidence")
        chunk_id = data.get("chunk_id")
        if evidence:
            relation_map[key]["evidence"].append(evidence)
        if chunk_id:
            relation_map[key]["chunk_ids"].append(chunk_id)
    return list(relation_map.values())


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    """Persist the knowledge graph as JSON."""
    payload = graph_to_dict(graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
