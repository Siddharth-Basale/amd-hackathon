# graph_builder.py — In-Depth Documentation

## Overview

The `graph_builder` module builds an aggregated **knowledge graph** from chunk-level extraction records. It converts per-chunk entity/relation triples into a unified `networkx.MultiDiGraph` with entity nodes, chunk nodes, and edges for `mentions` and entity–entity relations.

---

## Architecture

### Data Flow

```
chunk_records  ──►  build_graph()  ──►  nx.MultiDiGraph
                           │
                           ├── Add chunk nodes
                           ├── Add entity nodes (deduplicated by _entity_key)
                           ├── Add entity→chunk "mentions" edges
                           └── Add entity→entity relation edges
```

---

## API Reference

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_graph` | `(document_id: str, chunk_records: Iterable[Dict]) -> nx.MultiDiGraph` | Build MultiDiGraph from chunk records |
| `graph_to_dict` | `(graph: nx.MultiDiGraph) -> Dict` | Serialize graph to dict (nodes, edges) |
| `aggregate_entities` | `(graph: nx.MultiDiGraph) -> List[Dict]` | Flat list of entity metadata |
| `aggregate_relations` | `(graph: nx.MultiDiGraph) -> List[Dict]` | Deduplicated relations with evidence |
| `save_graph` | `(graph: nx.MultiDiGraph, path: Path) -> None` | Persist graph as JSON |

---

## Graph Structure

### Node Types

| Type | ID Format | Attributes |
|------|-----------|------------|
| `chunk` | `chunk:<chunk_id>` | chunk_id, heading, section_path |
| `entity` | `entity:<normalized_name>` | name, entity_type, description, aliases, documents, source_chunks |

### Edge Types

| Relation | Source | Target | Attributes |
|----------|--------|--------|------------|
| `mentions` | entity | chunk | Links entity to chunk where it appears |
| *(custom)* | entity | entity | relation, evidence, chunk_id |

---

## Chunk Record Format

Each record in `chunk_records` should have:

```python
{
    "chunk_id": "doc::chunk::0",
    "heading": "...",
    "section_path": "...",
    "entities": [
        {"name": "...", "type": "PERSON", "description": "...", "aliases": [...], "source_ids": [...]}
    ],
    "relations": [
        {"source": "...", "relation": "...", "target": "...", "evidence": "..."}
    ]
}
```

---

## Aggregation Logic

- **Entity deduplication**: `_entity_key(name)` = `name.strip().lower()`
- **Entity nodes**: Merged across chunks; `source_chunks` and `documents` aggregated
- **Relations**: Deduplicated by (source, relation, target); evidence and chunk_ids collected

---

## PlantUML Sequence Diagram

```plantuml
@startuml graph_builder
title graph_builder.py — Knowledge Graph Construction

participant "Caller (vectorizerE)" as Caller
participant "build_graph" as Build
participant "entity_nodes (dict)" as EntityMap
participant "nx.MultiDiGraph" as Graph
participant "aggregate_entities" as AggE
participant "aggregate_relations" as AggR
participant "save_graph" as Save

Caller -> Build: build_graph(document_id, knowledge_records)

loop for each chunk record
    Build -> Graph: add_node(chunk_node_id, type="chunk", ...)
    
    loop for each entity in record.entities
        Build -> Build: _entity_key(name)
        alt entity not in entity_nodes
            Build -> EntityMap: entity_nodes[node_id] = {...}
        end
        Build -> Graph: add_edge(entity_node, chunk_node, relation="mentions")
    end
    
    loop for each relation in record.relations
        alt source/target not in entity_nodes
            Build -> EntityMap: add entity nodes
        end
        Build -> Graph: add_edge(source_entity, target_entity, relation, evidence, chunk_id)
    end
end

loop for each entity in entity_nodes
    Build -> Graph: add_node(entity_id, type="entity", ...)
end

Build --> Caller: graph

Caller -> AggE: aggregate_entities(graph)
AggE --> Caller: List[Dict]

Caller -> AggR: aggregate_relations(graph)
AggR --> Caller: List[Dict]

Caller -> Save: save_graph(graph, path)
Save -> Save: graph_to_dict(graph)
Save -> Save: json.dump(payload, path)
Save --> Caller: (persisted)
@enduml
```
