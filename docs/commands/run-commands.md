# Run Commands

Quick reference for all runnable commands. See [email-pipeline-runbook.md](email-pipeline-runbook.md) for setup and workflow details.

---

## Email pipeline

```bash
# One-time: fetch recent emails, vectorize, build collection
python email_pipeline.py batch --max-results 50

# Continuous: poll for new emails and process them
python email_pipeline.py poll --poll-interval 60

# Rebuild collection only (from existing email folders)
python email_pipeline.py collection
```

**Options**

| Option | Description |
|--------|-------------|
| `--emails-root PATH` | Root directory for output (default: `output/emails`) |
| `--poll-interval SECONDS` | Poll interval for `poll` mode (default: 60) |
| `--max-results N` | Max messages to fetch in `batch` mode (default: 50) |
| `--label LABEL_ID` | Gmail label filter (e.g. INBOX); can repeat |

**Examples**

```bash
python email_pipeline.py batch --emails-root output/emails
python email_pipeline.py batch --label INBOX --max-results 20
python email_pipeline.py poll --poll-interval 30
```

---

## Markdown vectorization

```bash
# Vectorize a single .md file
python vectorizerE.py path/to/file.md

# Vectorize a folder (uses first .md found, or all subfolders with .md)
python vectorizerE.py path/to/folder
```

Output: `output/<doc_stem>/` (vector_db, document graph, knowledge, vector_mapping).

---

## Graph visualization

```bash
# Use most recent folder in output/
python visualizeGraphE.py

# Visualize a specific folder (e.g. one email or collection)
python visualizeGraphE.py output/emails/msg_xxx
python visualizeGraphE.py output/emails/collection

# Point at a graph file directly
python visualizeGraphE.py --graph-file path/to/doc_document_graph.json

# Optional: override knowledge graph path
python visualizeGraphE.py output/emails/msg_xxx --knowledge-graph path/to/kg.json
```

Output: `visualizations/` in the same folder (interactive HTML, simplified PNG, full PNG).

---

## Scripts

```bash
# Demo: vectorize → export → optional GraphRAG index
python scripts/demo_pipeline.py --markdown path/to/file.md
```

---

## Summary

| Task | Command |
|------|---------|
| Sync emails (batch) | `python email_pipeline.py batch --max-results 50` |
| Real-time emails | `python email_pipeline.py poll --poll-interval 60` |
| Rebuild collection | `python email_pipeline.py collection` |
| Vectorize markdown | `python vectorizerE.py <path.md or path/folder>` |
| Visualize graph | `python visualizeGraphE.py [path]` |
