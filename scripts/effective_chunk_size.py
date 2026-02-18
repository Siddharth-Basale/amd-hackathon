#!/usr/bin/env python3
"""
Print effective_chunk_size for an .md file (same logic as vectorizerE adaptive chunking).
Usage: python scripts/effective_chunk_size.py path/to/file.md
"""

import os
import sys
from pathlib import Path

# Same config as vectorizerE (respects env vars)
EMBEDDING_MAX_CHARS = 2000
CHUNK_SIZE = EMBEDDING_MAX_CHARS
MIN_CHUNKS = int(os.getenv("PLAN_E_MIN_CHUNKS", "8"))
MIN_CHUNK_CHARS = int(os.getenv("PLAN_E_MIN_CHUNK_CHARS", "300"))


def compute_effective_chunk_size(total_chars: int) -> int:
    """Same formula as vectorizerE.compute_effective_chunk_size."""
    return min(
        CHUNK_SIZE,
        max(MIN_CHUNK_CHARS, total_chars // MIN_CHUNKS),
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/effective_chunk_size.py <path/to/file.md>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)
    if path.suffix.lower() != ".md":
        print(f"Warning: expected .md file, got {path.suffix}")

    content = path.read_text(encoding="utf-8")
    total_chars = len(content)
    effective = compute_effective_chunk_size(total_chars)

    print(f"File: {path}")
    print(f"Total chars: {total_chars}")
    print(f"effective_chunk_size: {effective}")
    print(f"  (MIN_CHUNKS={MIN_CHUNKS}, MIN_CHUNK_CHARS={MIN_CHUNK_CHARS}, CHUNK_SIZE={CHUNK_SIZE})")
    if total_chars > 0:
        approx_chunks = max(1, (total_chars + effective - 1) // effective)
        print(f"Approx. chunks: ~{approx_chunks}")


if __name__ == "__main__":
    main()
