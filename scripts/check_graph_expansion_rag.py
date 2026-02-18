"""
Check graph expansion and RAG retrieval: load document + knowledge graphs,
run HybridRetrievalPlanner, GraphExpander, and expand_candidates.
Uses a mock vector retriever so Chroma is not required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mock_vector_retriever(doc_stem: str, chunk_ids: list[str]):
    """Build a vector retriever that returns the given chunk IDs with dummy scores."""

    def retriever(query: str, k: int):
        return [(cid, 1.0 - i * 0.05, {}) for i, cid in enumerate(chunk_ids[: k])]

    return retriever


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check graph expansion and RAG: planner + expander on existing outputs."
    )
    parser.add_argument(
        "--output",
        default="knowledge_package",
        help="Output folder name under output/ (e.g. knowledge_package or file). Default: knowledge_package",
    )
    parser.add_argument(
        "--query",
        default="How does entity extraction work?",
        help="Query string for the retrieval planner.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of seed candidates from planner (default: 5).",
    )
    parser.add_argument(
        "--max-expansion",
        type=int,
        default=25,
        help="Max expanded chunks (default: 25).",
    )
    parser.add_argument(
        "--no-doc-graph",
        action="store_true",
        help="Disable document graph expansion.",
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Disable knowledge graph expansion.",
    )
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    output_dir = root / "output" / args.output
    doc_graph_path = output_dir / f"{args.output}_document_graph.json"
    kg_path = output_dir / "knowledge" / f"{args.output}_knowledge_graph.json"

    if not doc_graph_path.exists():
        print(f"Document graph not found: {doc_graph_path}")
        return 1
    if not kg_path.exists():
        print(f"Knowledge graph not found: {kg_path}")
        return 1

    # Load graphs
    from retrieval.loaders import load_document_graph, load_knowledge_graph
    from retrieval.graph_expander import GraphExpander
    from retrieval.hybrid_planner import HybridRetrievalPlanner
    from retrieval.retrieve import retrieve
    from retrieval.debug_analysis import default_debug_dir
    from retrieval.chunk_utils import doc_id_to_kg, kg_to_doc_id

    print("Loading graphs...")
    doc_graph = load_document_graph(doc_graph_path)
    kg = load_knowledge_graph(kg_path)
    if kg is None:
        print("Failed to load knowledge graph.")
        return 1

    # Chunk IDs from knowledge graph for mock retriever (chunk nodes have chunk_id in attr)
    chunk_ids_from_kg = []
    for nid, data in kg.nodes(data=True):
        if data.get("type") == "chunk":
            cid = data.get("chunk_id", nid)
            if isinstance(cid, str) and cid not in chunk_ids_from_kg:
                chunk_ids_from_kg.append(cid)
    chunk_ids_from_kg.sort(key=lambda x: (x,))
    if not chunk_ids_from_kg:
        print("No chunk nodes found in knowledge graph.")
        return 1

    mock_retriever = _mock_vector_retriever(args.output, chunk_ids_from_kg)
    planner = HybridRetrievalPlanner(mock_retriever, kg)
    expander = GraphExpander(
        doc_graph,
        kg,
        doc_stem=args.output,
        max_expansion=args.max_expansion,
    )

    use_doc = not args.no_doc_graph
    use_kg = not args.no_kg
    debug_dir = root / "retrieval" / "debug"

    # Run retrieval (plan + expand + save full debug to retrieval folder)
    print(f"\nQuery: {args.query!r}")
    expanded, plan, debug_paths = retrieve(
        args.query,
        planner,
        expander,
        top_k=args.top_k,
        use_document_graph=use_doc,
        use_knowledge_graph=use_kg,
        debug_dir=debug_dir,
    )
    seed_ids = [c.chunk_id for c in plan.candidates]
    print(f"Plan mode: {plan.mode.value}")
    print(f"Seed candidates: {len(seed_ids)}")
    for c in plan.candidates:
        print(f"  - {c.chunk_id} (score={c.score:.2f}, source={c.source})")
    print(f"\nExpanded chunks: {len(expanded)} (doc_graph={use_doc}, kg={use_kg})")
    for ec in expanded:
        reason = (ec.reason[:40] + "…") if len(ec.reason or "") > 40 else (ec.reason or "")
        print(f"  - {ec.chunk_id}  source={ec.source}  reason={reason}")
    print(f"\nDebug analysis saved in retrieval folder:")
    for p in debug_paths:
        print(f"  {p}")

    # Quick sanity check: chunk_utils
    print("\nChunk ID utils:")
    stem = args.output
    kg_id = doc_id_to_kg(stem, 0)
    back = kg_to_doc_id(kg_id)
    print(f"  doc_id_to_kg({stem!r}, 0) = {kg_id!r}, kg_to_doc_id(...) = {back}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
