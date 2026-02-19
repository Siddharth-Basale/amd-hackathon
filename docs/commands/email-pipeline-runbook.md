# Email Pipeline Runbook

Step-by-step guide for first-time setup and ongoing usage. For a quick list of all run commands, see [run-commands.md](run-commands.md).

---

## First-Time Setup

### 1. Install dependencies
```bash
cd D:\Projects\AMD-Hackathon
pip install -r requirements.txt
```

### 2. Ensure credentials are in place
- `credentials.json` should exist in the project root (from Google Cloud Console).
- Do not commit it; it's in `.gitignore`.

### 3. Start Ollama (for embeddings)
The pipeline uses **Ollama** with `nomic-embed-text:v1.5`. Start the server:
```bash
ollama serve
```

In a separate terminal, pull the embedding model if needed:
```bash
ollama pull nomic-embed-text:v1.5
```

### 4. First run: Gmail authentication
The **first time** you run any mode that fetches emails, a browser window will open for OAuth:
```bash
python email_pipeline.py batch --max-results 5
```

- Sign in with your Gmail account
- Grant “read mail” permission
- A `token.json` file will be created in the project root (also in `.gitignore`)

The pipeline then fetches up to 5 recent INBOX messages, converts each to markdown, vectorizes them, and builds the collection.

---

## Ongoing Usage

### Option A: One-time batch (sync recent emails)
```bash
python email_pipeline.py batch --max-results 50
```

- Fetches recent messages from INBOX (or specified labels)
- Skips emails already processed (checks for existing `output/emails/<email_id>/`)
- Rebuilds the collection from all processed emails

### Option B: Real-time polling (process new emails as they arrive)
```bash
python email_pipeline.py poll --poll-interval 60
```

- Runs indefinitely
- Every 60 seconds, polls Gmail for **new** messages (via history API)
- Processes each new message and appends to the collection incrementally
- Stores last `historyId` in `output/emails/.state.json`

### Option C: Rebuild collection only
If you already have many emails under `output/emails/<email_id>/` and only want to refresh the merged collection:
```bash
python email_pipeline.py collection
```

- Does not require Gmail or Ollama
- Merges all per-email vector stores and graphs, computes cross-email similarity

---

## CLI Options

| Flag | Description |
|------|-------------|
| `--emails-root PATH` | Root directory for output (default: `output/emails`) |
| `--poll-interval SECONDS` | Poll interval for `poll` mode (default: 60) |
| `--max-results N` | Max messages to fetch in `batch` mode (default: 50) |
| `--label LABEL_ID` | Gmail label filter (e.g. INBOX); can repeat |

**Examples:**
```bash
python email_pipeline.py batch --emails-root D:\my_emails
python email_pipeline.py batch --label INBOX --max-results 20
python email_pipeline.py poll --poll-interval 30
```

---

## Output Layout

```
output/emails/
├── msg_<id>/                    # Per-email folder
│   ├── msg_<id>.md
│   ├── vector_db/
│   ├── msg_<id>_vector_mapping.json
│   ├── msg_<id>_document_graph.json
│   └── knowledge/
└── collection/                  # Merged index
    ├── vector_db/collection/
    ├── collection_vector_mapping.json
    ├── collection_document_graph.json
    └── collection_email_index.json
```

---

## Quick Reference

| Task | Command |
|------|---------|
| **First time** | `pip install -r requirements.txt` → start Ollama → `python email_pipeline.py batch --max-results 5` |
| **Sync more mail** | `python email_pipeline.py batch --max-results 50` |
| **Real-time mode** | `python email_pipeline.py poll --poll-interval 60` |
| **Rebuild collection** | `python email_pipeline.py collection` |
