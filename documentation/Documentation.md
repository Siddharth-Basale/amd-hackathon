# Graph-Based Document Understanding Pipeline — Documentation

## Overview

This project provides a **graph-based document understanding pipeline** for Markdown documents. It combines:

1. **Structure-aware chunking** — Parses Markdown with section hierarchy awareness
2. **Vector embedding** — Chroma + Ollama embeddings (nomic-embed-text)
3. **Document graph** — Section/chunk structure with `follows`, `contains`, `similar_to` edges
4. **Knowledge graph** — LLM-extracted entities and relations per chunk
5. **Hybrid retrieval** — Vector similarity + knowledge-graph traversal

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE OVERVIEW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  Markdown (.md)  ──►  vectorizerE.py  ──►  output/<doc>/
        │                      │
        │                      ├── vector_db/          (Chroma persistence)
        │                      ├── *_document_graph.json   (section/chunk graph)
        │                      ├── *_vector_mapping.json   (chunk metadata)
        │                      └── knowledge/
        │                            ├── *_chunk_knowledge.json
        │                            └── *_knowledge_graph.json
        │
        ▼
  visualizeGraphE.py  ──►  HTML/PNG visualizations
  retrieval/hybrid_planner.py  ──►  Query-time hybrid search
```

---

## Module Map

| Module | Purpose |
|--------|---------|
| **vectorizerE.py** | Main ingestion: parse Markdown, chunk, embed, build document graph, extract knowledge |
| **knowledge/entity_extractor.py** | LLM-based entity and relation extraction from chunk text |
| **knowledge/graph_builder.py** | Build aggregated knowledge graph from chunk-level triples |
| **knowledge/** | Package for entity extraction and graph construction |
| **retrieval/hybrid_planner.py** | Combine vector search with knowledge-graph traversal for queries |
| **visualizeGraphE.py** | Interactive/static graph visualizations |

---

## Data Flow

1. **Input**: Markdown file or folder containing `.md` files
2. **Parse**: Extract sections, headers, tables; chunk with structure metadata
3. **Embed**: Generate embeddings via Ollama, store in Chroma
4. **Document graph**: Add section nodes, chunk nodes, `contains`/`belongs_to`/`follows` edges
5. **Entity extraction**: Per chunk, LLM extracts entities and relations (if enabled)
6. **Knowledge graph**: Aggregate triples into a unified graph; link entities to chunks via `mentions`
7. **Similarity edges**: Add `similar_to` edges between semantically similar chunks
8. **Output**: Vector DB, document graph JSON, knowledge graph JSON, vector mapping JSON

---

## Quick Reference

### Run Vectorization

```bash
python vectorizerE.py path/to/document.md
# or
python vectorizerE.py path/to/folder/  # processes .md files in folder/subfolders
```

### Run Visualization

```bash
python visualizeGraphE.py output/<doc> --knowledge-graph output/<doc>/knowledge/<doc>_knowledge_graph.json
```

### Run Hybrid Retrieval

```python
from retrieval.hybrid_planner import HybridRetrievalPlanner, load_knowledge_graph
from pathlib import Path

# Load vector store + knowledge graph, then:
planner = HybridRetrievalPlanner(vector_retriever, knowledge_graph)
plan = planner.plan("How does X relate to Y?")
for c in plan.candidates:
    print(c.chunk_id, c.score, c.reason)
```

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [vectorizerE.md](./vectorizerE.md) | In-depth documentation for the main vectorization pipeline |
| [entity_extractor.md](./entity_extractor.md) | Entity and relation extraction using LLM structured output |
| [graph_builder.md](./graph_builder.md) | Knowledge graph construction from extraction records |
| [knowledge_package.md](./knowledge_package.md) | Overview of the `knowledge` package |
| [hybrid_planner.md](./hybrid_planner.md) | Hybrid retrieval combining vector + knowledge search |

Each document includes:
- Component description
- API reference
- Configuration options
- PlantUML sequence diagrams (inline and standalone)

### Standalone PlantUML Sequence Diagrams

| Diagram | Path |
|---------|------|
| vectorizerE pipeline | [diagrams/vectorizerE_sequence.puml](./diagrams/vectorizerE_sequence.puml) |
| entity_extractor | [diagrams/entity_extractor_sequence.puml](./diagrams/entity_extractor_sequence.puml) |
| graph_builder | [diagrams/graph_builder_sequence.puml](./diagrams/graph_builder_sequence.puml) |
| knowledge package | [diagrams/knowledge_package_sequence.puml](./diagrams/knowledge_package_sequence.puml) |
| hybrid_planner | [diagrams/hybrid_planner_sequence.puml](./diagrams/hybrid_planner_sequence.puml) |

To render PlantUML diagrams: use [PlantUML](https://plantuml.com/) (CLI, VS Code extension, or online editor).
