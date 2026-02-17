## Graph-Based Vectorization with Knowledge Overlay

This project builds a document understanding pipeline that:

- Parses Markdown documents into structure-aware chunks (`vectorizerE.py`)
- Generates embeddings and a section/sequence document graph
- Extracts entities and relations for each chunk and stores knowledge triples
- Exports the combined data in Microsoft GraphRAG parquet format (`graphrag_bridge.py`)
- Provides visualization for both document and knowledge layers (`visualizeGraphE.py`)
- Ships a hybrid retrieval planner that can mix vector search with knowledge-graph traversal

### Repository layout

| Path | Purpose |
| --- | --- |
| `vectorizerE.py` | Primary ingestion pipeline. Produces chunk embeddings, document graph, and knowledge triples/graph. |
| `knowledge/entity_extractor.py` | LLM-backed entity & relation extractor used during vectorization. |
| `knowledge/graph_builder.py` | Converts chunk-level triples into a knowledge graph + aggregated entity/relationship tables. |
| `graphrag_bridge.py` | Bundles Plan-E outputs into GraphRAG-compatible parquet exports (sections, chunks, knowledge entities, relations). |
| `visualizeGraphE.py` | Generates interactive/static graphs with optional knowledge overlay. |
| `retrieval/hybrid_planner.py` | Helper utilities to blend vector similarity and knowledge traversal for query answering. |
| `scripts/demo_pipeline.py` | Smoke-test / walkthrough script for running the full pipeline end-to-end. |

---

## Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`
- Running Ollama instance (defaults to `http://localhost:11434`) with:
  - Embedding model: `nomic-embed-text:v1.5`
  - LLM for extraction: e.g. `llama3.1:8b` (configurable via `PLAN_E_ENTITY_MODEL`)
- Optional: `graphrag` CLI (`pip install graphrag`)

### Environment toggles

| Variable | Default | Description |
| --- | --- | --- |
| `PLAN_E_ENABLE_ENTITY_EXTRACTION` | `1` | Toggle per-chunk entity extraction. Set to `0` to skip knowledge generation. |
| `PLAN_E_ENTITY_MODEL` | `gpt-4o-mini` | Chat model used for entity extraction (OpenAI key required). |
| `PLAN_E_ENABLE_ENTITY_EXTRACTION` | `1` | Set to `"0"` to disable knowledge graph generation. |

---

## Quickstart

1. **Vectorize documents**
   ```bash
   python vectorizerE.py path/to/folder-or-markdown
   ```
   Outputs land under `output/<doc>/`:
   - `vector_db/` – Chroma persistence
   - `<doc>_document_graph.json` – section/sequence graph
   - `knowledge/<doc>_chunk_knowledge.json` – per-chunk extraction
   - `knowledge/<doc>_knowledge_graph.json` – aggregated knowledge graph

2. **Export GraphRAG tables**
   ```bash
   python graphrag_bridge.py output/<doc> --dest output/graphrag --collection demo
   ```
   Generates parquet tables `entities.parquet`, `relationships.parquet`, `text_units.parquet`, `documents.parquet` (with knowledge entity/edge augmentation) plus a starter `settings.template.yaml`.

3. **Index with GraphRAG**
   ```bash
   cd graphrag-project
   graphrag init --root . -m gpt-4o-mini -e nomic-embed-text   # non-interactive, uses your defaults
   graphrag index --root .
   ```
   Ensure `.env` has `OPENAI_API_KEY` and Ollama is serving embeddings.

4. **Ask questions**
   ```bash
   graphrag query --root graphrag-project --mode global "How does Project X relate to onboarding?"
   ```

---

## Visualization

Produce document + knowledge overlays:

```bash
python visualizeGraphE.py output/<doc> \
  --knowledge-graph output/<doc>/knowledge/<doc>_knowledge_graph.json
```

- `*_interactive.html` merges the section/chunk graph with knowledge entities.
- `*_knowledge_only.html` focuses just on the entity graph.
- Static PNGs now include optional legend entries for knowledge entities.

---

## Hybrid retrieval planner

`retrieval/hybrid_planner.py` exposes:

```python
from retrieval.hybrid_planner import HybridRetrievalPlanner, RetrievalMode

planner = HybridRetrievalPlanner(vector_retriever, knowledge_graph)
plan = planner.plan("How is Vendor X related to renewal delays?")
for candidate in plan.candidates:
    print(candidate.chunk_id, candidate.score, candidate.reason)
```

- `vector_retriever(query, k)` should return `(chunk_id, score, metadata)` tuples.
- Knowledge traversal matches entity names/aliases and follows `mentions` edges back to chunks.

---

## Demo pipeline script

```
python scripts/demo_pipeline.py \
  --markdown docs/sample.md \
  --collection demo \
  --graphrag-root graphrag-project
```

What it does:
1. Runs `vectorizerE.py` on the supplied markdown folder or file.
2. Invokes `graphrag_bridge.py` to produce parquet exports in `output/graphrag/<collection>`.
3. Optionally calls `graphrag index --root <graphrag-root>` when the CLI is available (can be disabled with `--skip-index`).
4. Prints final `graphrag query` commands for manual exploration.

Use this script in CI or as a reproducible smoke test after adding new documents.

---

## Notes

- Knowledge graph JSON mirrors the document graph structure (node/edge payloads) and can be merged or visualized independently.
- To speed extraction or reduce cost, tweak `PLAN_E_ENTITY_MODEL` (e.g. to a distilled model) or disable extraction entirely.
- `knowledge/entity_extractor.py` is intentionally modular; swap in OpenAI or Azure chat models by supplying a custom `llm` argument.
