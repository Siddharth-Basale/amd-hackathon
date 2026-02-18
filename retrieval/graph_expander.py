"""
Graph expansion for RAG: expand seed chunks via document graph and knowledge graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx

from retrieval.chunk_utils import doc_id_to_kg, kg_to_doc_id, normalize_to_doc_id, normalize_to_kg


@dataclass
class ExpandedChunk:
    """A chunk from graph expansion with provenance."""

    chunk_id: str
    source: str  # "document" or "knowledge"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def expand_from_chunk_kg(
    kg: nx.MultiDiGraph,
    chunk_id: str,
    *,
    doc_stem: Optional[str] = None,
    mention_relation: str = "mentions",
    entity_node_type: str = "entity",
    max_hops: int = 2,
    max_chunks: int = 20,
) -> List[ExpandedChunk]:
    """
    Expand from a seed chunk in the knowledge graph.

    1. Get entities that mention this chunk (reverse mentions).
    2. Get related entities via entity-entity edges.
    3. Get chunks those entities mention.
    4. Deduplicate and cap.
    """
    chunk_node_id = f"chunk:{chunk_id}"
    if chunk_node_id not in kg:
        return []

    results: Dict[str, ExpandedChunk] = {}
    seen_entities: set = set()
    entity_queue: deque = deque()
    hop = 0

    # Step 1: Get entities that mention this chunk (predecessors with relation=mentions)
    for pred, _, edge_data in kg.in_edges(chunk_node_id, data=True):
        if edge_data.get("relation") == mention_relation and kg.nodes[pred].get("type") == entity_node_type:
            entity_name = kg.nodes[pred].get("name", pred)
            entity_queue.append((pred, 0, f"entity:{entity_name}"))
            seen_entities.add(pred)

    while entity_queue and len(results) < max_chunks and hop <= max_hops:
        entity_id, depth, reason_prefix = entity_queue.popleft()
        if depth > max_hops:
            continue
        hop = max(hop, depth)

        # Step 2 & 3: From this entity, get chunks via mentions (out_edges to chunk nodes)
        for _, target, edge_data in kg.out_edges(entity_id, data=True):
            if edge_data.get("relation") != mention_relation:
                continue
            target_data = kg.nodes.get(target, {})
            if target_data.get("type") != "chunk":
                continue
            target_chunk_id = target_data.get("chunk_id", target)
            if target_chunk_id == chunk_id:
                continue  # Skip seed
            if target_chunk_id not in results:
                entity_name = kg.nodes[entity_id].get("name", entity_id)
                results[target_chunk_id] = ExpandedChunk(
                    chunk_id=target_chunk_id,
                    source="knowledge",
                    reason=f"entity:{entity_name}",
                    metadata={"entity_id": entity_id},
                )

        # Step 2: Get related entities (entity-entity edges) for next hop
        if depth < max_hops:
            for _, target, edge_data in kg.out_edges(entity_id, data=True):
                rel = edge_data.get("relation")
                if rel == mention_relation:
                    continue
                if kg.nodes.get(target, {}).get("type") == entity_node_type and target not in seen_entities:
                    seen_entities.add(target)
                    entity_name = kg.nodes[entity_id].get("name", entity_id)
                    target_name = kg.nodes[target].get("name", target)
                    entity_queue.append((target, depth + 1, f"{reason_prefix} -> {target_name}"))
            for pred, _, edge_data in kg.in_edges(entity_id, data=True):
                if edge_data.get("relation") == mention_relation:
                    continue
                if kg.nodes.get(pred, {}).get("type") == entity_node_type and pred not in seen_entities:
                    seen_entities.add(pred)
                    pred_name = kg.nodes[pred].get("name", pred)
                    entity_queue.append((pred, depth + 1, f"{reason_prefix} <- {pred_name}"))

    return list(results.values())[:max_chunks]


class GraphExpander:
    """
    Expand seed chunks via document graph and/or knowledge graph.
    """

    def __init__(
        self,
        document_graph: Optional[Any],
        knowledge_graph: Optional[nx.MultiDiGraph],
        doc_stem: str,
        *,
        max_expansion: int = 25,
        max_kg_chunks_per_seed: int = 20,
    ) -> None:
        self.document_graph = document_graph
        self.knowledge_graph = knowledge_graph
        self.doc_stem = doc_stem
        self.max_expansion = max_expansion
        self.max_kg_chunks_per_seed = max_kg_chunks_per_seed

    def expand(
        self,
        seed_chunk_ids: List[Union[int, str]],
        *,
        use_document_graph: bool = True,
        use_knowledge_graph: bool = True,
    ) -> List[ExpandedChunk]:
        """
        Expand seed chunks using both graphs.
        Returns deduplicated list with canonical chunk_id format (kg string).
        """
        all_chunk_ids: Dict[str, ExpandedChunk] = {}

        # Normalize seeds to doc IDs and kg IDs
        doc_ids: List[int] = []
        kg_ids: List[str] = []
        for sid in seed_chunk_ids:
            doc_id = normalize_to_doc_id(sid, self.doc_stem)
            kg_id = normalize_to_kg(sid, self.doc_stem)
            if doc_id is not None:
                doc_ids.append(doc_id)
            kg_ids.append(kg_id)
            all_chunk_ids[kg_id] = ExpandedChunk(
                chunk_id=kg_id,
                source="seed",
                reason="seed",
            )

        # Document graph expansion
        if use_document_graph and self.document_graph and doc_ids:
            try:
                expanded_doc_ids = self.document_graph.expand_from_chunks(
                    doc_ids, max_expansion=self.max_expansion
                )
                for eid in expanded_doc_ids:
                    kg_id = doc_id_to_kg(self.doc_stem, eid)
                    if kg_id not in all_chunk_ids:
                        all_chunk_ids[kg_id] = ExpandedChunk(
                            chunk_id=kg_id,
                            source="document",
                            reason="section|adjacent|similar",
                        )
            except Exception:
                pass

        # Knowledge graph expansion
        if use_knowledge_graph and self.knowledge_graph and kg_ids:
            for kg_id in kg_ids:
                kg_expanded = expand_from_chunk_kg(
                    self.knowledge_graph,
                    kg_id,
                    max_hops=2,
                    max_chunks=self.max_kg_chunks_per_seed,
                )
                for ec in kg_expanded:
                    if ec.chunk_id not in all_chunk_ids:
                        all_chunk_ids[ec.chunk_id] = ec

        return list(all_chunk_ids.values())
