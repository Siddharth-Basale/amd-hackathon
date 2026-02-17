"""
Hybrid retrieval planner combining vector search with knowledge-graph traversal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


VectorRetriever = Callable[[str, int], Sequence[Tuple[str, float, Dict]]]


class RetrievalMode(str, Enum):
    DOCUMENT = "document"
    ENTITY = "entity"
    HYBRID = "hybrid"


@dataclass
class RetrievalCandidate:
    chunk_id: str
    score: float
    reason: str
    source: str  # "vector" or "knowledge"
    metadata: Dict


@dataclass
class RetrievalPlan:
    mode: RetrievalMode
    candidates: List[RetrievalCandidate]
    contributing_entities: List[str]


class QueryClassifier:
    """Simple heuristic classifier for query intent."""

    entity_patterns = [
        re.compile(r"\brelationship between\b", re.IGNORECASE),
        re.compile(r"\bhow.*related\b", re.IGNORECASE),
        re.compile(r"\bcompare\b", re.IGNORECASE),
    ]

    document_patterns = [
        re.compile(r"\bpage\b", re.IGNORECASE),
        re.compile(r"\bsection\b", re.IGNORECASE),
        re.compile(r"\bsummary\b", re.IGNORECASE),
    ]

    def classify(self, query: str) -> RetrievalMode:
        query_lc = query.lower()
        if any(pattern.search(query_lc) for pattern in self.entity_patterns):
            if any(pattern.search(query_lc) for pattern in self.document_patterns):
                return RetrievalMode.HYBRID
            return RetrievalMode.ENTITY
        if any(pattern.search(query_lc) for pattern in self.document_patterns):
            return RetrievalMode.DOCUMENT
        return RetrievalMode.HYBRID


class HybridRetrievalPlanner:
    """Planner that merges vector retrieval and knowledge graph traversal."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        knowledge_graph: Optional[nx.MultiDiGraph] = None,
        *,
        classifier: Optional[QueryClassifier] = None,
        entity_node_type: str = "entity",
        mention_relation: str = "mentions",
    ) -> None:
        self.vector_retriever = vector_retriever
        self.knowledge_graph = knowledge_graph
        self.classifier = classifier or QueryClassifier()
        self.entity_node_type = entity_node_type
        self.mention_relation = mention_relation

    def plan(self, query: str, *, top_k: int = 8) -> RetrievalPlan:
        mode = self.classifier.classify(query)
        vector_candidates = self._run_vector_search(query, top_k)
        knowledge_candidates, entities = self._run_knowledge_search(query, top_k)

        if mode == RetrievalMode.DOCUMENT:
            candidates = vector_candidates
        elif mode == RetrievalMode.ENTITY:
            candidates = knowledge_candidates or vector_candidates
        else:
            merged: Dict[str, RetrievalCandidate] = {}
            for candidate in vector_candidates + knowledge_candidates:
                existing = merged.get(candidate.chunk_id)
                if existing is None or candidate.score > existing.score:
                    merged[candidate.chunk_id] = candidate
            candidates = sorted(merged.values(), key=lambda item: item.score, reverse=True)
            candidates = candidates[:top_k]

        return RetrievalPlan(mode=mode, candidates=candidates, contributing_entities=entities)

    def _run_vector_search(self, query: str, top_k: int) -> List[RetrievalCandidate]:
        results: List[RetrievalCandidate] = []
        try:
            for chunk_id, score, metadata in self.vector_retriever(query, top_k):
                results.append(
                    RetrievalCandidate(
                        chunk_id=chunk_id,
                        score=score,
                        reason="semantic similarity",
                        source="vector",
                        metadata=metadata,
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                RetrievalCandidate(
                    chunk_id="__error__",
                    score=0.0,
                    reason=f"vector retriever failed: {exc}",
                    source="vector",
                    metadata={},
                )
            )
        return results

    def _run_knowledge_search(
        self, query: str, top_k: int
    ) -> Tuple[List[RetrievalCandidate], List[str]]:
        if not self.knowledge_graph:
            return [], []

        query_tokens = {token.lower() for token in re.findall(r"\w+", query) if len(token) > 2}
        matched_entities: Dict[str, float] = {}

        for node_id, data in self.knowledge_graph.nodes(data=True):
            if data.get("type") != self.entity_node_type:
                continue
            name = (data.get("name") or "").lower()
            aliases = [alias.lower() for alias in data.get("aliases", [])]
            if name in query_tokens or any(alias in query_tokens for alias in aliases):
                matched_entities[node_id] = 1.0
            else:
                overlap = query_tokens.intersection(set(name.split()))
                if overlap:
                    matched_entities[node_id] = max(matched_entities.get(node_id, 0.0), len(overlap) / len(query_tokens))

        candidates: Dict[str, RetrievalCandidate] = {}
        contributing_entities: List[str] = []

        for entity_id, entity_score in matched_entities.items():
            contributing_entities.append(self.knowledge_graph.nodes[entity_id].get("name", entity_id))
            for _, chunk_node, data in self.knowledge_graph.out_edges(entity_id, data=True):
                if data.get("relation") != self.mention_relation:
                    continue
                chunk_id = self.knowledge_graph.nodes[chunk_node].get("chunk_id", chunk_node)
                score = entity_score
                if data.get("evidence"):
                    score += 0.05
                candidate = candidates.get(chunk_id)
                if candidate is None or score > candidate.score:
                    candidates[chunk_id] = RetrievalCandidate(
                        chunk_id=chunk_id,
                        score=score,
                        reason=f"entity match: {self.knowledge_graph.nodes[entity_id].get('name')}",
                        source="knowledge",
                        metadata={
                            "entity_id": entity_id,
                            "evidence": data.get("evidence"),
                        },
                    )

        ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k], contributing_entities


def load_knowledge_graph(path: Path) -> Optional[nx.MultiDiGraph]:
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
