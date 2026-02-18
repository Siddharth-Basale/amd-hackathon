"""
Full debug analysis for every retrieval. Saves a JSON payload and a readable .txt
report under the retrieval folder so each run can be inspected.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval.hybrid_planner import RetrievalPlan, RetrievalCandidate
from retrieval.graph_expander import ExpandedChunk


def _run_id(query: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    h = hashlib.sha256(query.encode()).hexdigest()[:8]
    return f"{ts}_{h}"


def _graph_stats(doc_graph: Any, knowledge_graph: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"document_graph": None, "knowledge_graph": None}
    if doc_graph is not None and getattr(doc_graph, "graph", None) is not None:
        g = doc_graph.graph
        stats["document_graph"] = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
    if knowledge_graph is not None:
        stats["knowledge_graph"] = {
            "nodes": knowledge_graph.number_of_nodes(),
            "edges": knowledge_graph.number_of_edges(),
        }
    return stats


def build_debug_payload(
    query: str,
    top_k: int,
    plan: RetrievalPlan,
    expanded: List[ExpandedChunk],
    *,
    doc_graph: Any = None,
    knowledge_graph: Any = None,
    use_document_graph: bool = True,
    use_knowledge_graph: bool = True,
    max_expansion: int = 25,
    doc_stem: str = "",
    timings_ms: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable dict with full retrieval debug info."""
    candidates_debug = [
        {
            "chunk_id": c.chunk_id,
            "score": c.score,
            "reason": c.reason,
            "source": c.source,
            "metadata": c.metadata,
        }
        for c in plan.candidates
    ]
    expanded_debug = [
        {
            "chunk_id": ec.chunk_id,
            "source": ec.source,
            "reason": ec.reason,
            "metadata": ec.metadata,
        }
        for ec in expanded
    ]
    payload = {
        "query": query,
        "top_k": top_k,
        "plan": {
            "mode": plan.mode.value,
            "candidates": candidates_debug,
            "contributing_entities": plan.contributing_entities,
        },
        "expanded": expanded_debug,
        "expansion_config": {
            "use_document_graph": use_document_graph,
            "use_knowledge_graph": use_knowledge_graph,
            "max_expansion": max_expansion,
            "doc_stem": doc_stem,
        },
        "graph_stats": _graph_stats(doc_graph, knowledge_graph),
        "counts": {
            "seed_candidates": len(plan.candidates),
            "expanded_chunks": len(expanded),
            "by_source": _count_by_source(expanded),
        },
    }
    if timings_ms:
        payload["timings_ms"] = timings_ms
    if extra:
        payload["extra"] = extra
    return payload


def _count_by_source(expanded: List[ExpandedChunk]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ec in expanded:
        counts[ec.source] = counts.get(ec.source, 0) + 1
    return counts


def _payload_to_text(payload: Dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "RETRIEVAL DEBUG ANALYSIS",
        "=" * 60,
        f"Query: {payload.get('query', '')!r}",
        f"Top-K: {payload.get('top_k')}",
        "",
        "--- Plan ---",
        f"Mode: {payload.get('plan', {}).get('mode')}",
        f"Contributing entities: {payload.get('plan', {}).get('contributing_entities')}",
        "",
        "Seed candidates:",
    ]
    for c in payload.get("plan", {}).get("candidates", []):
        lines.append(f"  - {c.get('chunk_id')}  score={c.get('score')}  source={c.get('source')}  reason={c.get('reason')}")
    lines.extend([
        "",
        "--- Expansion config ---",
        json.dumps(payload.get("expansion_config", {}), indent=2),
        "",
        "--- Counts ---",
        json.dumps(payload.get("counts", {}), indent=2),
        "",
        "--- Graph stats ---",
        json.dumps(payload.get("graph_stats", {}), indent=2),
        "",
        "--- Expanded chunks ---",
    ])
    for ec in payload.get("expanded", []):
        lines.append(f"  - {ec.get('chunk_id')}  source={ec.get('source')}  reason={ec.get('reason', '')[:60]}")
    if payload.get("timings_ms"):
        lines.extend(["", "--- Timings (ms) ---", json.dumps(payload["timings_ms"], indent=2)])
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def write_retrieval_debug(
    debug_dir: Path,
    payload: Dict[str, Any],
    run_id: str,
) -> List[Path]:
    """
    Write full debug analysis for one retrieval under debug_dir.
    Creates debug_dir if needed. Returns paths written (e.g. [json_path, txt_path]).
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    json_path = debug_dir / f"retrieval_{run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    written.append(json_path)

    txt_path = debug_dir / f"retrieval_{run_id}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_payload_to_text(payload))
    written.append(txt_path)

    return written


def default_debug_dir() -> Path:
    """Default directory for retrieval debug output (under retrieval folder)."""
    return Path(__file__).resolve().parent / "debug"
