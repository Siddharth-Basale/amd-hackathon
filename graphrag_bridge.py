"""
GraphRAG Bridge
================

Utility to convert the VectorizerE outputs (graph + chunk JSON files) into
tables that follow Microsoft GraphRAG's knowledge model. The script gathers
entities, relationships, text units, and documents across one or more sources,
adds optional cross-document links, and writes the results as Parquet files that
can be ingested by GraphRAG's indexing workflows.

Usage
-----
    python graphrag_bridge.py output/file output/file2 --dest output/graphrag --collection demo

After running the script, call GraphRAG with a minimal workflow configuration:
    graphrag init --root <project_root>
    graphrag index --root <project_root>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger("graphrag_bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class IdCounter:
    """Simple incremental counter for human_readable_id fields."""

    def __init__(self, start: int = 0) -> None:
        self.value = start

    def next(self) -> int:
        self.value += 1
        return self.value


def slugify(value: str) -> str:
    """Create a filesystem/identifier friendly slug."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "unnamed"


def normalize_key(text: Optional[str]) -> Optional[str]:
    """Normalise section or heading text for matching across documents."""
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip().lower()
    return cleaned or None


def estimate_tokens(text: str) -> int:
    """Rough token estimate using word count."""
    if not text:
        return 0
    words = re.findall(r"\w+", text)
    return max(len(words), 1)


def truncate_text(text: str, limit: int = 480) -> str:
    """Limit text length while keeping whole sentences when possible."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_period = truncated.rfind(". ")
    last_newline = truncated.rfind("\n")
    cut = max(last_period, last_newline)
    if cut > limit * 0.5:
        return truncated[: cut + 1].strip()
    return truncated.strip() + "..."


def find_source_files(source_dir: Path) -> Tuple[Path, Path]:
    """Locate the document graph and vector mapping JSON files."""
    graph_files = list(source_dir.glob("*_document_graph.json"))
    vector_files = list(source_dir.glob("*_vector_mapping.json"))

    if not graph_files or not vector_files:
        raise FileNotFoundError(
            f"Expected *_document_graph.json and *_vector_mapping.json in {source_dir}"
        )
    # Prefer the lexicographically first match (consistent naming from VectorizerE)
    return graph_files[0], vector_files[0]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_destination(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    logger.info("Writing %s (%d rows)", path.name, len(frame))
    try:
        frame.to_parquet(path, engine="pyarrow", index=False)
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to write Parquet files. Install dependencies from requirements.txt."
        ) from exc


def build_tables(
    sources: Sequence[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create GraphRAG tables from the provided VectorizerE output folders."""

    entity_counter = IdCounter()
    relationship_counter = IdCounter()
    text_unit_counter = IdCounter()
    document_counter = IdCounter()

    entities: List[Dict] = []
    relationships: List[Dict] = []
    text_units: List[Dict] = []
    documents: List[Dict] = []

    entity_degrees: Counter = Counter()
    text_unit_map: Dict[str, Dict] = {}
    entity_titles: Dict[str, str] = {}

    section_index: Dict[str, List[Dict]] = defaultdict(list)
    heading_index: Dict[str, List[Dict]] = defaultdict(list)

    cross_relationship_keys: set = set()

    for source_dir in sources:
        graph_path, vector_path = find_source_files(source_dir)
        graph_data = load_json(graph_path)
        vector_data = load_json(vector_path)
        chunk_text_unit_index: Dict[str, str] = {}

        doc_id = graph_path.stem.replace("_document_graph", "")
        source_md = source_dir.parent / f"{doc_id}.md"
        md_path = source_md if source_md.exists() else None

        logger.info("Processing %s (graph: %s)", doc_id, graph_path.name)

        # Prepare node maps
        node_id_map: Dict[str, str] = {}
        section_nodes: Dict[str, Dict] = {}

        for node in graph_data.get("nodes", []):
            node_type = node.get("type")
            node_id = node.get("id")
            if node_type == "section":
                section_path = node.get("section_path") or node.get("section_title") or node_id.split("section:", 1)[-1]
                section_title = node.get("section_title") or section_path
                slug = slugify(section_path)
                entity_id = f"{doc_id}::section::{slug}"
                section_nodes[node_id] = {
                    "entity_id": entity_id,
                    "section_path": section_path,
                    "title": section_title,
                    "level": node.get("level"),
                    "start_line": node.get("start_line"),
                }
                node_id_map[node_id] = entity_id
            elif node_type == "chunk":
                chunk_id = node.get("chunk_id")
                if chunk_id is None:
                    continue
                entity_id = f"{doc_id}::chunk::{chunk_id}"
                node_id_map[node_id] = entity_id

        # Prepare chunk data from vector mapping
        chunk_records: Dict[int, Dict] = {}
        ordered_chunks: List[Dict] = []
        for entry in vector_data:
            chunk_id = entry.get("vector_number")
            if chunk_id is None:
                continue
            metadata = entry.get("metadata", {})
            heading = entry.get("heading") or metadata.get("heading") or f"Chunk {chunk_id}"
            section_path = entry.get("section_path") or metadata.get("section_path") or ""
            content = entry.get("content", "")
            sequential_position = metadata.get("sequential_position", chunk_id)
            section_title = metadata.get("section_title")
            start_line = metadata.get("start_line")

            text_unit_id = f"{doc_id}::tu::{chunk_id}"
            entity_id = f"{doc_id}::chunk::{chunk_id}"
            chunk_identifier = entity_id

            record = {
                "chunk_id": chunk_id,
                "entity_id": entity_id,
                "chunk_identifier": chunk_identifier,
                "text_unit_id": text_unit_id,
                "section_path": section_path,
                "heading": heading,
                "section_title": section_title,
                "content": content,
                "sequential_position": sequential_position,
                "start_line": start_line,
                "metadata": metadata,
            }
            chunk_records[chunk_id] = record
            ordered_chunks.append(record)

        ordered_chunks.sort(key=lambda item: item["sequential_position"])

        knowledge_dir = source_dir / "knowledge"
        knowledge_chunks: List[Dict[str, Any]] = []
        aggregated_entities: List[Dict[str, Any]] = []
        aggregated_relations: List[Dict[str, Any]] = []
        if knowledge_dir.exists():
            knowledge_chunk_path = knowledge_dir / f"{doc_id}_chunk_knowledge.json"
            if knowledge_chunk_path.exists():
                logger.info("Loading knowledge extraction from %s", knowledge_chunk_path)
                payload = load_json(knowledge_chunk_path)
                knowledge_chunks = payload.get("chunks", [])
                aggregated_entities = payload.get("entities", [])
                aggregated_relations = payload.get("relations", [])
            else:
                logger.info("Knowledge directory present but no chunk knowledge file for %s", doc_id)

        # Map sections to chunks
        section_chunks: Dict[str, List[Dict]] = defaultdict(list)
        for record in chunk_records.values():
            section_chunks[record["section_path"]].append(record)

        # Build section entities
        section_entity_by_path: Dict[str, str] = {}
        for original_id, section_info in section_nodes.items():
            section_path = section_info["section_path"]
            section_title = section_info["title"]
            section_entity_id = section_info["entity_id"]
            section_chunk_records = section_chunks.get(section_path, [])
            section_text_unit_ids = [chunk["text_unit_id"] for chunk in section_chunk_records]
            section_description_chunks = [chunk["content"] for chunk in section_chunk_records[:2]]
            section_description = truncate_text("\n\n".join(section_description_chunks)) if section_description_chunks else f"Section {section_title}"

            entity_row = {
                "id": section_entity_id,
                "human_readable_id": entity_counter.next(),
                "title": section_title,
                "type": "section",
                "description": section_description,
                "text_unit_ids": section_text_unit_ids,
                "document_ids": [doc_id],
                "frequency": len(section_text_unit_ids),
                "degree": 0,  # filled later
                "metadata": {
                    "section_path": section_path,
                    "level": section_info.get("level"),
                    "start_line": section_info.get("start_line"),
                },
            }
            entities.append(entity_row)
            entity_titles[section_entity_id] = section_title
            section_entity_by_path[section_path] = section_entity_id

            normalized = normalize_key(section_path)
            if normalized:
                section_index[normalized].append(
                    {
                        "entity_id": section_entity_id,
                        "doc_id": doc_id,
                        "title": section_title,
                    }
                )

        # Build chunk entities and text units
        for record in ordered_chunks:
            chunk_entity_id = record["entity_id"]
            text_unit_id = record["text_unit_id"]
            section_path = record["section_path"]
            section_entity_id = section_entity_by_path.get(section_path)

            entity_row = {
                "id": chunk_entity_id,
                "human_readable_id": entity_counter.next(),
                "title": record["heading"],
                "type": "chunk",
                "description": truncate_text(record["content"]),
                "text_unit_ids": [text_unit_id],
                "document_ids": [doc_id],
                "frequency": 1,
                "degree": 0,  # filled later
                "metadata": {
                    "section_path": section_path,
                    "sequential_position": record["sequential_position"],
                    "start_line": record.get("start_line"),
                },
            }
            entities.append(entity_row)
            entity_titles[chunk_entity_id] = record["heading"]

            entity_ids = [chunk_entity_id]
            if section_entity_id:
                entity_ids.append(section_entity_id)

            text_unit_row = {
                "id": text_unit_id,
                "human_readable_id": text_unit_counter.next(),
                "text": record["content"],
                "n_tokens": estimate_tokens(record["content"]),
                "document_id": doc_id,
                "entity_ids": entity_ids,
                "relationship_ids": [],
                "metadata": {
                    "heading": record["heading"],
                    "section_path": section_path,
                    "sequential_position": record["sequential_position"],
                    "start_line": record.get("start_line"),
                },
            }
            text_units.append(text_unit_row)
            text_unit_map[text_unit_id] = text_unit_row
            chunk_text_unit_index[entity_id] = text_unit_id

            normalized_heading = normalize_key(record["heading"])
            if normalized_heading:
                heading_index[normalized_heading].append(
                    {
                        "entity_id": chunk_entity_id,
                        "doc_id": doc_id,
                        "text_unit_id": text_unit_id,
                    }
                )

        # Build document entry
        document_text = "\n\n".join([record["content"] for record in ordered_chunks])
        document_row = {
            "id": doc_id,
            "human_readable_id": document_counter.next(),
            "title": doc_id,
            "text": document_text,
            "text_unit_ids": [record["text_unit_id"] for record in ordered_chunks],
            "metadata": {"source_path": str(md_path) if md_path else ""},
        }
        documents.append(document_row)

        # Build relationships
        seen_relationships: set = set()

        def chunk_from_entity(entity_id: str) -> Optional[Dict]:
            if "::chunk::" not in entity_id:
                return None
            try:
                chunk_idx = int(entity_id.rsplit("::", 1)[-1])
            except ValueError:
                return None
            return chunk_records.get(chunk_idx)

        for edge in graph_data.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            if src not in node_id_map or tgt not in node_id_map:
                continue

            source_id = node_id_map[src]
            target_id = node_id_map[tgt]
            relation = edge.get("relation", "related_to")

            dedupe_key = (source_id, target_id, relation)
            if dedupe_key in seen_relationships:
                continue
            seen_relationships.add(dedupe_key)

            text_unit_ids = []
            chunk_source = chunk_from_entity(source_id)
            chunk_target = chunk_from_entity(target_id)
            if chunk_source:
                text_unit_ids.append(chunk_source["text_unit_id"])
            if chunk_target:
                text_unit_ids.append(chunk_target["text_unit_id"])

            text_unit_ids = list(dict.fromkeys(text_unit_ids))

            weight = float(edge.get("similarity", 1.0))
            description = edge.get("description")
            if not description:
                source_label = entity_titles.get(source_id, source_id)
                target_label = entity_titles.get(target_id, target_id)
                description = f"{relation} relationship between {source_label} and {target_label}"

            relationship_id = f"{doc_id}::{relationship_counter.next()}"
            relationship_row = {
                "id": relationship_id,
                "human_readable_id": relationship_counter.value,
                "source": source_id,
                "target": target_id,
                "description": description,
                "relation_type": relation,
                "weight": weight,
                "text_unit_ids": text_unit_ids,
                "document_ids": [doc_id],
            }
            relationships.append(relationship_row)

            entity_degrees[source_id] += 1
            entity_degrees[target_id] += 1
            for tu_id in text_unit_ids:
                text_unit = text_unit_map.get(tu_id)
                if text_unit is not None:
                    text_unit["relationship_ids"].append(relationship_id)

        knowledge_entity_map: Dict[str, str] = {}
        knowledge_name_map: Dict[str, str] = {}

        def register_knowledge_entity(entity: Dict[str, Any]) -> Optional[str]:
            entity_id_raw = entity.get("id")
            name = entity.get("name")
            if not entity_id_raw or not name:
                return None
            node_id = f"{doc_id}::{entity_id_raw}"
            if node_id in knowledge_entity_map.values():
                return node_id
            text_unit_ids = [
                chunk_text_unit_index.get(chunk_id)
                for chunk_id in entity.get("source_chunks", [])
                if chunk_text_unit_index.get(chunk_id)
            ]
            entity_row = {
                "id": node_id,
                "human_readable_id": entity_counter.next(),
                "title": name,
                "type": "knowledge_entity",
                "description": entity.get("description", ""),
                "text_unit_ids": text_unit_ids,
                "document_ids": [doc_id],
                "frequency": len(entity.get("source_chunks", [])),
                "degree": 0,
                "metadata": {
                    "aliases": entity.get("aliases", []),
                    "source_chunks": entity.get("source_chunks", []),
                },
            }
            entities.append(entity_row)
            entity_titles[node_id] = name
            knowledge_entity_map[entity_id_raw] = node_id
            knowledge_name_map[name.lower()] = node_id
            for alias in entity.get("aliases", []):
                knowledge_name_map[alias.lower()] = node_id
            entity_degrees[node_id] += 0
            return node_id

        for entity in aggregated_entities:
            register_knowledge_entity(entity)

        for record in knowledge_chunks:
            chunk_identifier = record.get("chunk_id")
            if not chunk_identifier:
                continue
            chunk_entity_id = chunk_identifier
            text_unit_id = chunk_text_unit_index.get(chunk_entity_id)
            for entity in record.get("entities", []):
                name = entity.get("name")
                if not name:
                    continue
                target_node = knowledge_name_map.get(name.lower())
                if not target_node:
                    continue
                relationship_id = f"{doc_id}::{relationship_counter.next()}"
                relationship_row = {
                    "id": relationship_id,
                    "human_readable_id": relationship_counter.value,
                    "source": target_node,
                    "target": chunk_entity_id,
                    "description": f"{name} mentioned in {chunk_entity_id}",
                    "relation_type": "mentions",
                    "weight": 1.0,
                    "text_unit_ids": [text_unit_id] if text_unit_id else [],
                    "document_ids": [doc_id],
                    "metadata": {
                        "section_path": record.get("section_path"),
                        "heading": record.get("heading"),
                    },
                }
                relationships.append(relationship_row)
                entity_degrees[target_node] += 1
                entity_degrees[chunk_entity_id] += 1
                if text_unit_id:
                    text_unit = text_unit_map.get(text_unit_id)
                    if text_unit is not None:
                        text_unit["relationship_ids"].append(relationship_id)

        for relation in aggregated_relations:
            source_raw = relation.get("source")
            target_raw = relation.get("target")
            rel_type = relation.get("relation")
            if not source_raw or not target_raw or not rel_type:
                continue
            source_node = knowledge_entity_map.get(source_raw)
            target_node = knowledge_entity_map.get(target_raw)
            if not source_node or not target_node:
                continue
            relationship_id = f"{doc_id}::{relationship_counter.next()}"
            text_unit_ids = [
                chunk_text_unit_index.get(chunk_id)
                for chunk_id in relation.get("chunk_ids", [])
                if chunk_text_unit_index.get(chunk_id)
            ]
            relationship_row = {
                "id": relationship_id,
                "human_readable_id": relationship_counter.value,
                "source": source_node,
                "target": target_node,
                "description": relation.get("relation"),
                "relation_type": rel_type,
                "weight": 1.0,
                "text_unit_ids": text_unit_ids,
                "document_ids": [doc_id],
                "metadata": {
                    "evidence": relation.get("evidence", []),
                    "chunk_ids": relation.get("chunk_ids", []),
                },
            }
            relationships.append(relationship_row)
            entity_degrees[source_node] += 1
            entity_degrees[target_node] += 1
            for tu_id in text_unit_ids:
                text_unit = text_unit_map.get(tu_id)
                if text_unit is not None:
                    text_unit["relationship_ids"].append(relationship_id)

    # Cross-document section links
    for key, entries in section_index.items():
        doc_ids = {entry["doc_id"] for entry in entries}
        if len(doc_ids) < 2:
            continue
        for left, right in combinations(entries, 2):
            sorted_pair = tuple(sorted([left["entity_id"], right["entity_id"]]))
            rel_key = (*sorted_pair, "cross_section_match")
            if rel_key in cross_relationship_keys:
                continue
            cross_relationship_keys.add(rel_key)
            description = f"Shared section path between {left['doc_id']} and {right['doc_id']}"
            relationship_id = f"cross::section::{relationship_counter.next()}"
            relationship_row = {
                "id": relationship_id,
                "human_readable_id": relationship_counter.value,
                "source": sorted_pair[0],
                "target": sorted_pair[1],
                "description": description,
                "relation_type": "cross_section_match",
                "weight": 1.0,
                "text_unit_ids": [],
                "document_ids": sorted(list({left["doc_id"], right["doc_id"]})),
            }
            relationships.append(relationship_row)
            entity_degrees[sorted_pair[0]] += 1
            entity_degrees[sorted_pair[1]] += 1

    # Cross-document chunk heading links
    for key, entries in heading_index.items():
        doc_ids = {entry["doc_id"] for entry in entries}
        if len(doc_ids) < 2:
            continue
        for left, right in combinations(entries, 2):
            sorted_pair = tuple(sorted([left["entity_id"], right["entity_id"]]))
            rel_key = (*sorted_pair, "cross_heading_match")
            if rel_key in cross_relationship_keys:
                continue
            cross_relationship_keys.add(rel_key)
            description = "Chunks share the same heading across documents"
            relationship_id = f"cross::heading::{relationship_counter.next()}"
            relationship_row = {
                "id": relationship_id,
                "human_readable_id": relationship_counter.value,
                "source": sorted_pair[0],
                "target": sorted_pair[1],
                "description": description,
                "relation_type": "cross_heading_match",
                "weight": 0.9,
                "text_unit_ids": [],
                "document_ids": sorted(list({left["doc_id"], right["doc_id"]})),
            }
            relationships.append(relationship_row)
            entity_degrees[sorted_pair[0]] += 1
            entity_degrees[sorted_pair[1]] += 1

    # Fill degree and combined degree
    entity_degree_map = dict(entity_degrees)
    for entity in entities:
        entity["degree"] = entity_degree_map.get(entity["id"], 0)
        entity.setdefault("frequency", max(1, len(entity.get("text_unit_ids", []))))

    relationship_combined: List[int] = []
    for relationship in relationships:
        source_degree = entity_degree_map.get(relationship["source"], 0)
        target_degree = entity_degree_map.get(relationship["target"], 0)
        relationship["combined_degree"] = source_degree + target_degree
        relationship_combined.append(source_degree + target_degree)

    # Deduplicate relationship IDs inside text units
    for entry in text_units:
        entry["relationship_ids"] = sorted(set(entry.get("relationship_ids", [])))

    # Convert metadata dicts to JSON strings for Parquet compatibility
    def serialise_metadata(frame: List[Dict]) -> None:
        for entry in frame:
            if "metadata" in entry:
                entry["metadata"] = json.dumps(entry["metadata"], ensure_ascii=False)

    serialise_metadata(entities)
    serialise_metadata(text_units)
    serialise_metadata(documents)

    entities_df = pd.DataFrame(entities)
    relationships_df = pd.DataFrame(relationships)
    text_units_df = pd.DataFrame(text_units)
    documents_df = pd.DataFrame(documents)

    return entities_df, relationships_df, text_units_df, documents_df


def write_settings_template(dest: Path, collection: str) -> None:
    """Create a starter settings YAML to help run GraphRAG."""
    template = f"""# Generated by graphrag_bridge.py
project:
  name: {collection}

paths:
  root: .
  output: {dest.as_posix()}
  graph_tables: {dest.as_posix()}
  documents: {dest.as_posix()}

workflows:
  - create_communities
  - create_community_reports
  - generate_text_embeddings
"""
    settings_path = dest / "settings.template.yaml"
    settings_path.write_text(template, encoding="utf-8")
    logger.info("Wrote GraphRAG settings template to %s", settings_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VectorizerE outputs into GraphRAG-compatible parquet tables."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Paths to VectorizerE output folders (e.g. output/file).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("output/graphrag"),
        help="Destination root directory for GraphRAG tables.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="collection",
        help="Name of the collection folder under the destination root.",
    )
    parser.add_argument(
        "--settings-template",
        action="store_true",
        help="Generate a starter settings.template.yaml alongside the tables.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source_dirs = [source.resolve() for source in args.sources]

    for source in source_dirs:
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")

    entities_df, relationships_df, text_units_df, documents_df = build_tables(source_dirs)

    collection_dir = args.dest / args.collection
    ensure_destination(collection_dir)

    write_parquet(entities_df, collection_dir / "entities.parquet")
    write_parquet(relationships_df, collection_dir / "relationships.parquet")
    write_parquet(text_units_df, collection_dir / "text_units.parquet")
    write_parquet(documents_df, collection_dir / "documents.parquet")

    if args.settings_template:
        write_settings_template(collection_dir, args.collection)

    logger.info("Completed GraphRAG bridge conversion.")


if __name__ == "__main__":
    main()
