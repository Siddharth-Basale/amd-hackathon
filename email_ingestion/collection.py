"""
Email Collection Builder - Merges per-email vector stores and graphs into
output/emails/collection/ with cross-email similarity edges.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ingestion.vectorizer_e import DocumentGraph, EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# Same as vectorizerE
SIMILARITY_THRESHOLD = 0.50
MAX_SIMILAR_PER_CHUNK = 5
SCALE_FACTOR = 150.0


def _get_email_folders(emails_root: Path) -> List[Path]:
    """List email folders (excluding collection)."""
    if not emails_root.exists():
        return []
    return [
        d
        for d in emails_root.iterdir()
        if d.is_dir() and d.name != "collection"
    ]


def _load_vector_mapping(email_dir: Path, email_id: str) -> List[Dict[str, Any]]:
    """Load vector_mapping.json for an email."""
    mapping_path = email_dir / f"{email_id}_vector_mapping.json"
    if not mapping_path.exists():
        logger.warning("No vector mapping at %s", mapping_path)
        return []
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_metadata_for_chroma(meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB rejects empty list metadata values. Omit empty lists; convert non-empty to strings."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, list):
            if len(v) == 0:
                continue  # Omit empty lists (Chroma rejects them)
            out[k] = ",".join(str(x) for x in v)  # Convert to string for Chroma compatibility
        else:
            out[k] = v
    return out


def build_collection(
    emails_root: Path,
    collection_dir: Optional[Path] = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Merge all emails in emails_root into output/emails/collection/.
    - Single Chroma with all chunks (metadata includes email_id)
    - Merged document graph with prefixed chunk IDs
    - Cross-email similarity edges
    - collection_email_index.json mapping email_id -> chunk_ids
    """
    emails_root = Path(emails_root)
    collection_dir = collection_dir or emails_root / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)

    email_folders = sorted(_get_email_folders(emails_root))
    if not email_folders:
        logger.warning("No email folders found in %s", emails_root)
        return {"email_count": 0}

    all_chunks: List[Dict[str, Any]] = []
    email_index: Dict[str, List[str]] = {}

    # Load all vector mappings
    for email_dir in email_folders:
        email_id = email_dir.name
        mapping = _load_vector_mapping(email_dir, email_id)
        for item in mapping:
            chunk_id = item.get("vector_number", len(all_chunks))
            composite_id = f"{email_id}::chunk::{chunk_id}"
            content = item.get("content", "")
            meta = item.get("metadata", {}).copy()
            meta["email_id"] = email_id
            meta["composite_chunk_id"] = composite_id
            meta["chunk_index"] = chunk_id
            meta = _sanitize_metadata_for_chroma(meta)
            all_chunks.append({
                "content": content,
                "metadata": meta,
                "composite_id": composite_id,
                "email_id": email_id,
            })
            email_index.setdefault(email_id, []).append(composite_id)

    if not all_chunks:
        logger.warning("No chunks to merge")
        return {"email_count": 0}

    logger.info("Merging %d chunks from %d emails into collection", len(all_chunks), len(email_index))

    # Create collection Chroma
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_db_path = collection_dir / "vector_db" / "collection"
    vector_db_path.mkdir(parents=True, exist_ok=True)

    # Clear existing collection via Chroma API (avoids Windows file-lock from rmtree)
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(vector_db_path))
        try:
            client.delete_collection("collection")
        except Exception:
            pass
    except Exception as e:
        logger.warning("Could not clear collection via API: %s", e)

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(vector_db_path),
        collection_name="collection",
    )

    # Add documents
    docs = [
        Document(page_content=c["content"], metadata=c["metadata"])
        for c in all_chunks
    ]
    vector_store.add_documents(docs)
    logger.info("Added %d documents to collection Chroma", len(docs))

    # Build merged DocumentGraph with composite chunk IDs
    merged_graph = DocumentGraph()
    # We need to add chunk nodes - DocumentGraph expects add_chunk_node(chunk_id, chunk)
    # For collection we use composite_id as chunk_id - but add_chunk_node uses int chunk_id.
    # Create a custom graph structure for collection that uses string IDs.
    merged_graph.graph.clear()
    merged_graph.chunk_nodes.clear()
    merged_graph.section_nodes.clear()

    for c in all_chunks:
        composite_id = c["composite_id"]
        node_id = f"chunk:{composite_id}"
        merged_graph.graph.add_node(
            node_id,
            type="chunk",
            chunk_id=composite_id,
            heading=c["metadata"].get("heading", ""),
            section_path=c["metadata"].get("section_path", ""),
            email_id=c["email_id"],
        )
        merged_graph.chunk_nodes[composite_id] = node_id

    # Add follows edges within same email
    for email_id, chunk_ids in email_index.items():
        for i, cid in enumerate(chunk_ids):
            node = merged_graph.chunk_nodes.get(cid)
            if not node:
                continue
            # Next chunk in same email
            if i + 1 < len(chunk_ids):
                next_cid = chunk_ids[i + 1]
                next_node = merged_graph.chunk_nodes.get(next_cid)
                if next_node:
                    merged_graph.add_edge(node, next_node, relation="follows")
            # Prev chunk
            if i > 0:
                prev_cid = chunk_ids[i - 1]
                prev_node = merged_graph.chunk_nodes.get(prev_cid)
                if prev_node:
                    merged_graph.add_edge(prev_node, node, relation="follows")

    # Cross-email similarity edges
    similarity_edges = 0
    for i, c in enumerate(all_chunks):
        if (i + 1) % 20 == 0:
            logger.info("Computing cross-email similarity for chunk %d/%d", i + 1, len(all_chunks))
        composite_id = c["composite_id"]
        email_id = c["email_id"]
        node = merged_graph.chunk_nodes.get(composite_id)
        if not node:
            continue

        try:
            results = vector_store.similarity_search_with_score(
                c["content"],
                k=MAX_SIMILAR_PER_CHUNK + 5,
            )
            count = 0
            for similar_doc, distance_score in results:
                sim_composite = similar_doc.metadata.get("composite_chunk_id")
                sim_email = similar_doc.metadata.get("email_id")
                if not sim_composite or sim_composite == composite_id:
                    continue
                if sim_email == email_id:
                    continue  # Only cross-email
                if sim_composite not in merged_graph.chunk_nodes:
                    continue

                similarity = 1.0 / (1.0 + (distance_score / SCALE_FACTOR))
                similarity = max(0.0, min(1.0, similarity))
                if similarity < similarity_threshold:
                    continue

                sim_node = merged_graph.chunk_nodes[sim_composite]
                # Avoid duplicate edges
                if merged_graph.graph.has_edge(node, sim_node):
                    ed = merged_graph.graph.get_edge_data(node, sim_node)
                    if ed and ed.get("relation") == "similar_to":
                        continue
                merged_graph.add_edge(node, sim_node, relation="similar_to", similarity=similarity)
                similarity_edges += 1
                count += 1
                if count >= MAX_SIMILAR_PER_CHUNK:
                    break
        except Exception as e:
            logger.warning("Similarity search error for %s: %s", composite_id, e)

    logger.info("Added %d cross-email similarity edges", similarity_edges)

    # Save outputs
    graph_path = collection_dir / "collection_document_graph.json"
    merged_graph.save(graph_path)

    index_path = collection_dir / "collection_email_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(email_index, f, indent=2)

    # Build vector_mapping-style JSON for collection
    mapping_out = []
    for c in all_chunks:
        mapping_out.append({
            "composite_chunk_id": c["composite_id"],
            "email_id": c["email_id"],
            "content": c["content"],
            "metadata": c["metadata"],
        })
    mapping_path = collection_dir / "collection_vector_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_out, f, indent=2, ensure_ascii=False)

    logger.info("Collection saved to %s", collection_dir)
    return {
        "email_count": len(email_index),
        "chunk_count": len(all_chunks),
        "similarity_edges": similarity_edges,
        "collection_dir": str(collection_dir),
    }


def append_email_to_collection(
    email_id: str,
    email_dir: Path,
    collection_dir: Path,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Incrementally add one email's chunks to the collection and update similarity edges.
    """
    mapping = _load_vector_mapping(email_dir, email_id)
    if not mapping:
        return {"added": 0}

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_db_path = collection_dir / "vector_db" / "collection"
    if not vector_db_path.exists():
        vector_db_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(vector_db_path),
        collection_name="collection",
    )

    # Load existing email index
    index_path = collection_dir / "collection_email_index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            email_index = json.load(f)
    else:
        email_index = {}

    # Add new chunks
    new_chunks = []
    for item in mapping:
        chunk_id = item.get("vector_number", 0)
        composite_id = f"{email_id}::chunk::{chunk_id}"
        content = item.get("content", "")
        meta = item.get("metadata", {}).copy()
        meta["email_id"] = email_id
        meta["composite_chunk_id"] = composite_id
        meta["chunk_index"] = chunk_id
        meta = _sanitize_metadata_for_chroma(meta)
        doc = Document(page_content=content, metadata=meta)
        vector_store.add_documents([doc])
        new_chunks.append({
            "content": content,
            "metadata": meta,
            "composite_id": composite_id,
            "email_id": email_id,
        })
        email_index.setdefault(email_id, []).append(composite_id)

    # Load merged graph, add new chunk nodes and edges
    graph_path = collection_dir / "collection_document_graph.json"
    merged_graph = DocumentGraph()
    if graph_path.exists():
        merged_graph.load(graph_path)

    for c in new_chunks:
        node_id = f"chunk:{c['composite_id']}"
        merged_graph.graph.add_node(
            node_id,
            type="chunk",
            chunk_id=c["composite_id"],
            heading=c["metadata"].get("heading", ""),
            section_path=c["metadata"].get("section_path", ""),
            email_id=c["email_id"],
        )
        merged_graph.chunk_nodes[c["composite_id"]] = node_id

    # Follows within new email
    cids = email_index[email_id]
    for i, cid in enumerate(cids):
        node = merged_graph.chunk_nodes.get(cid)
        if not node:
            continue
        if i + 1 < len(cids):
            next_node = merged_graph.chunk_nodes.get(cids[i + 1])
            if next_node:
                merged_graph.add_edge(node, next_node, relation="follows")
        if i > 0:
            prev_node = merged_graph.chunk_nodes.get(cids[i - 1])
            if prev_node:
                merged_graph.add_edge(prev_node, node, relation="follows")

    # Cross-email similarity for new chunks only
    similarity_edges = 0
    for c in new_chunks:
        results = vector_store.similarity_search_with_score(c["content"], k=MAX_SIMILAR_PER_CHUNK + 5)
        count = 0
        for similar_doc, distance_score in results:
            sim_composite = similar_doc.metadata.get("composite_chunk_id")
            sim_email = similar_doc.metadata.get("email_id")
            if not sim_composite or sim_composite == c["composite_id"]:
                continue
            if sim_email == email_id:
                continue
            if sim_composite not in merged_graph.chunk_nodes:
                continue
            similarity = 1.0 / (1.0 + (distance_score / SCALE_FACTOR))
            similarity = max(0.0, min(1.0, similarity))
            if similarity < similarity_threshold:
                continue
            sim_node = merged_graph.chunk_nodes[sim_composite]
            node = merged_graph.chunk_nodes[c["composite_id"]]
            merged_graph.add_edge(node, sim_node, relation="similar_to", similarity=similarity)
            similarity_edges += 1
            count += 1
            if count >= MAX_SIMILAR_PER_CHUNK:
                break

    merged_graph.save(graph_path)
    with open(index_path, "w") as f:
        json.dump(email_index, f, indent=2)

    # Append to vector mapping
    mapping_path = collection_dir / "collection_vector_mapping.json"
    existing = []
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            existing = json.load(f)
    for c in new_chunks:
        existing.append({
            "composite_chunk_id": c["composite_id"],
            "email_id": c["email_id"],
            "content": c["content"],
            "metadata": c["metadata"],
        })
    with open(mapping_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    return {"added": len(new_chunks), "similarity_edges": similarity_edges}
