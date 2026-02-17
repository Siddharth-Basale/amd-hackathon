"""
Run a lightweight end-to-end demo covering vectorization → GraphRAG export → indexing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd: Path) -> None:
    print(f"\n› Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the GraphRAG hybrid pipeline.")
    parser.add_argument("--markdown", required=True, help="Path to a markdown file or folder containing markdown.")
    parser.add_argument(
        "--collection",
        default="demo",
        help="Collection name for GraphRAG exports (default: demo).",
    )
    parser.add_argument(
        "--dest",
        default="output/graphrag",
        help="Destination root for GraphRAG exports (default: output/graphrag).",
    )
    parser.add_argument(
        "--graphrag-root",
        default=None,
        help="Optional graphrag project root. If provided, the script will call `graphrag index` afterwards.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip graphrag index even if --graphrag-root is supplied.",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    markdown_path = Path(args.markdown).resolve()
    if not markdown_path.exists():
        parser.error(f"Markdown path not found: {markdown_path}")

    vectorizer_cmd = [sys.executable, "vectorizerE.py", str(markdown_path)]
    run_command(vectorizer_cmd, repo_root)

    doc_output = markdown_path.stem if markdown_path.is_file() else markdown_path.name
    doc_folder = repo_root / "output" / doc_output
    if not doc_folder.exists():
        print(f"Expected vectorizer outputs at {doc_folder}, but nothing was created.")
        sys.exit(1)

    bridge_cmd = [
        sys.executable,
        "graphrag_bridge.py",
        str(doc_folder),
        "--dest",
        str(Path(args.dest)),
        "--collection",
        args.collection,
    ]
    run_command(bridge_cmd, repo_root)

    if args.graphrag_root and not args.skip_index:
        graphrag_root = Path(args.graphrag_root).resolve()
        if not graphrag_root.exists():
            print(f"GraphRAG project root not found: {graphrag_root}")
        else:
            index_cmd = ["graphrag", "index", "--root", str(graphrag_root)]
            try:
                run_command(index_cmd, graphrag_root)
            except subprocess.CalledProcessError as exc:  # pragma: no cover - informative
                print(f"graphrag index failed: {exc}. You can rerun manually if desired.")

    print("\n✓ Demo completed.")
    print("Next steps:")
    print(f"  1. Inspect knowledge outputs in {doc_folder / 'knowledge'}")
    print(f"  2. Exports available in {Path(args.dest) / args.collection}")
    print("  3. Run queries, e.g.:")
    if args.graphrag_root:
        print(f"     graphrag query --root {args.graphrag_root} --mode global \"What decisions affect renewal delays?\"")
    else:
        print("     (Run `graphrag init` + `graphrag index` to enable querying)")


if __name__ == "__main__":
    main()
