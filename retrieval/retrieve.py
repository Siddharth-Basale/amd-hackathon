"""
Single entry point for retrieval: plan + expand + save full debug analysis
to the retrieval folder for every run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

from retrieval.debug_analysis import (
    build_debug_payload,
    default_debug_dir,
    write_retrieval_debug,
    _run_id,
)
from retrieval.graph_expander import ExpandedChunk, GraphExpander
from retrieval.hybrid_planner import HybridRetrievalPlanner, RetrievalPlan, expand_candidates


def retrieve(
    query: str,
    planner: HybridRetrievalPlanner,
    expander: GraphExpander,
    *,
    top_k: int = 8,
    use_document_graph: bool = True,
    use_knowledge_graph: bool = True,
    debug_dir: Optional[Path] = None,
) -> tuple[List[ExpandedChunk], RetrievalPlan, List[Path]]:
    """
    Run one retrieval: plan, expand, then save full debug analysis under
    the retrieval folder. Returns (expanded_chunks, plan, debug_file_paths).
    """
    if debug_dir is None:
        debug_dir = default_debug_dir()

    t0 = time.perf_counter()
    plan = planner.plan(query, top_k=top_k)
    t_plan = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    expanded = expand_candidates(
        plan,
        expander,
        use_document_graph=use_document_graph,
        use_knowledge_graph=use_knowledge_graph,
    )
    t_expand = (time.perf_counter() - t1) * 1000

    timings_ms = {"plan_ms": round(t_plan, 2), "expand_ms": round(t_expand, 2)}

    payload = build_debug_payload(
        query=query,
        top_k=top_k,
        plan=plan,
        expanded=expanded,
        doc_graph=getattr(expander, "document_graph", None),
        knowledge_graph=getattr(expander, "knowledge_graph", None),
        use_document_graph=use_document_graph,
        use_knowledge_graph=use_knowledge_graph,
        max_expansion=getattr(expander, "max_expansion", 25),
        doc_stem=getattr(expander, "doc_stem", ""),
        timings_ms=timings_ms,
    )
    run_id = _run_id(query)
    written = write_retrieval_debug(debug_dir, payload, run_id)

    return expanded, plan, written
