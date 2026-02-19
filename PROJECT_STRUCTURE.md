# Project Structure

```
AMD-Hackathon/
├── email_ingestion/          # Email pipeline: Gmail → Markdown → Vectorize → Collection
│   ├── __init__.py
│   ├── fetcher.py            # Gmail API, OAuth, message parsing
│   ├── to_markdown.py        # Email → expandable markdown
│   ├── collection.py        # Merge per-email into collection (Chroma + graph)
│   └── pipeline.py           # Orchestration (poll, batch, collection)
│
├── ingestion/                # Markdown vectorization and document graph
│   ├── __init__.py
│   └── vectorizer_e.py       # Chunking, embeddings, document graph, knowledge extraction
│
├── visualization/            # Graph visualization (HTML, PNG)
│   ├── __init__.py
│   └── graph.py              # Document/knowledge graph viz (pyvis, matplotlib)
│
├── retrieval/                # RAG: hybrid planner, graph expansion, loaders
│   ├── __init__.py
│   ├── chunk_utils.py
│   ├── graph_expander.py
│   ├── hybrid_planner.py
│   ├── loaders.py            # load_document_graph, load_knowledge_graph
│   ├── retrieve.py
│   └── debug_analysis.py
│
├── knowledge/                # Entity extraction and knowledge graph
│   ├── __init__.py
│   ├── entity_extractor.py
│   └── graph_builder.py
│
├── scripts/                  # One-off and demo scripts
│   ├── demo_pipeline.py
│   ├── check_graph_expansion_rag.py
│   └── effective_chunk_size.py
│
├── docs/                     # Documentation and runbooks
│   ├── commands/             # CLI runbooks (e.g. email-pipeline-runbook.md)
│   └── documentation/       # Module docs, diagrams
│
├── output/                   # Generated output (vector DBs, graphs, emails)
│   └── emails/               # Per-email folders + collection/
│
├── email_pipeline.py         # Entry point: python email_pipeline.py batch|poll|collection
├── vectorizerE.py            # Entry point: python vectorizerE.py <path.md>
├── visualizeGraphE.py        # Entry point: python visualizeGraphE.py [path]
├── requirements.txt
├── credentials.json          # Gmail OAuth (gitignored)
└── README.md
```

## Entry points

| Command | Purpose |
|--------|--------|
| `python email_pipeline.py batch` | Fetch recent emails, vectorize, build collection |
| `python email_pipeline.py poll` | Continuous polling for new emails |
| `python email_pipeline.py collection` | Rebuild collection from existing emails |
| `python vectorizerE.py <path.md>` | Vectorize markdown file/folder |
| `python visualizeGraphE.py [path]` | Generate graph visualizations |

## Package dependency flow

- **email_ingestion** → ingestion (vectorize_markdown_content), visualization (visualize_directory)
- **ingestion** → knowledge (entity_extractor, graph_builder)
- **retrieval** → ingestion (DocumentGraph via loaders)
- **visualization** → standalone (loads graph JSON from disk)
