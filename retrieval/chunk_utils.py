"""
Chunk ID normalization between document graph and knowledge graph formats.
"""

from __future__ import annotations

from typing import Optional, Union


def doc_id_to_kg(doc_stem: str, chunk_id: Union[int, str]) -> str:
    """
    Convert document graph chunk ID (int) to knowledge graph format.
    Returns "doc_stem::chunk::{chunk_id}".
    """
    return f"{doc_stem}::chunk::{chunk_id}"


def kg_to_doc_id(chunk_id: str) -> Optional[int]:
    """
    Extract document graph chunk ID (int) from knowledge graph format.
    E.g. "docname::chunk::5" -> 5.
    Returns None if the format is invalid.
    """
    if not chunk_id or "::chunk::" not in chunk_id:
        return None
    try:
        tail = chunk_id.rsplit("::", 1)[-1]
        return int(tail)
    except (ValueError, IndexError):
        return None


def normalize_to_kg(chunk_id: Union[int, str], doc_stem: str) -> str:
    """
    Normalize a chunk ID (int or kg string) to knowledge graph format.
    """
    if isinstance(chunk_id, int):
        return doc_id_to_kg(doc_stem, chunk_id)
    return chunk_id


def normalize_to_doc_id(chunk_id: Union[int, str], doc_stem: str) -> Optional[int]:
    """
    Normalize a chunk ID to document graph format (int).
    If already int, return as-is if it looks valid (>= 0).
    """
    if isinstance(chunk_id, int):
        return chunk_id if chunk_id >= 0 else None
    return kg_to_doc_id(chunk_id)
