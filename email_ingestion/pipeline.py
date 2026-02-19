"""
Email Pipeline - Orchestrates real-time email ingestion:
1. Fetch new emails via Gmail API (polling)
2. Convert to markdown (expandable hierarchy)
3. Vectorize per-email (ingestion)
4. Update collection (merged vector store + graph)
5. Generate visualizations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from email_ingestion.fetcher import GmailFetcher
from email_ingestion.to_markdown import email_id_from_message, save_email_markdown
from email_ingestion.collection import append_email_to_collection, build_collection
from ingestion.vectorizer_e import vectorize_markdown_content, check_ollama_running

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EMAILS_ROOT = Path("output/emails")
STATE_FILE = ".state.json"


def run_single_email(
    message_id: str,
    emails_root: Path,
    fetcher: GmailFetcher,
) -> bool:
    """
    Fetch one email, convert to markdown, vectorize, and add to collection.
    Returns True on success.
    """
    try:
        msg = fetcher.fetch_message(message_id)
    except Exception as e:
        logger.error("Failed to fetch message %s: %s", message_id, e)
        return False

    email_id = email_id_from_message(msg.message_id)
    email_dir = emails_root / email_id
    collection_dir = emails_root / "collection"

    # 1. Convert to markdown and save
    md_path = save_email_markdown(msg, emails_root, email_id=email_id)
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    # 2. Vectorize (solo email)
    try:
        vectorize_markdown_content(
            markdown_content=markdown_content,
            output_root=emails_root,
            doc_stem=email_id,
        )
    except Exception as e:
        logger.error("Vectorization failed for %s: %s", email_id, e)
        return False

    # 3. Update collection (incremental or full build)
    try:
        if collection_dir.exists() and (collection_dir / "collection_email_index.json").exists():
            append_email_to_collection(email_id, email_dir, collection_dir)
        else:
            build_collection(emails_root, collection_dir)
    except Exception as e:
        logger.error("Collection update failed for %s: %s", email_id, e)
        return False

    # 4. Generate all visualizations (per-email + collection)
    try:
        from visualization.graph import visualize_directory
        visualize_directory(email_dir, quiet=True)
        logger.info("Created visualizations for %s", email_id)
        if collection_dir.exists():
            visualize_directory(collection_dir, quiet=True)
            logger.info("Created collection visualizations")
    except Exception as e:
        logger.warning("Visualization failed for %s: %s", email_id, e)

    logger.info("Processed email %s successfully", email_id)
    return True


def run_polling_loop(
    emails_root: Path,
    poll_interval: float = 60.0,
    label_ids: Optional[List[str]] = None,
) -> None:
    """
    Continuously poll for new emails and process them.
    """
    fetcher = GmailFetcher()
    state_path = emails_root / STATE_FILE
    emails_root.mkdir(parents=True, exist_ok=True)

    for message_id in fetcher.iter_new_messages(
        state_path=state_path,
        label_ids=label_ids,
        poll_interval_seconds=poll_interval,
    ):
        run_single_email(message_id, emails_root, fetcher)


def run_batch(
    emails_root: Path,
    max_results: int = 50,
    label_ids: Optional[List[str]] = None,
) -> None:
    """
    Fetch recent emails (one-time batch), process each.
    """
    fetcher = GmailFetcher()
    emails_root.mkdir(parents=True, exist_ok=True)

    messages, _ = fetcher.list_messages(
        label_ids=label_ids or ["INBOX"],
        max_results=max_results,
    )
    logger.info("Found %d messages to process", len(messages))

    for m in messages:
        message_id = m.get("id")
        if not message_id:
            continue
        email_id = f"msg_{message_id}"
        email_dir = emails_root / email_id
        # Skip if already processed
        if (email_dir / f"{email_id}_vector_mapping.json").exists():
            logger.info("Skipping already processed %s", email_id)
            continue
        run_single_email(message_id, emails_root, fetcher)

    # Final collection build (in case we started from scratch)
    build_collection(emails_root, emails_root / "collection")
    # Visualize collection after full rebuild
    try:
        from visualization.graph import visualize_directory
        visualize_directory(emails_root / "collection", quiet=True)
        logger.info("Created collection visualizations")
    except Exception as e:
        logger.warning("Collection visualization failed: %s", e)


def run_build_collection_only(emails_root: Path) -> None:
    """Rebuild collection from existing email folders."""
    emails_root = Path(emails_root)
    result = build_collection(emails_root)
    logger.info("Collection build result: %s", result)
    try:
        from visualization.graph import visualize_directory
        visualize_directory(emails_root / "collection", quiet=True)
        logger.info("Created collection visualizations")
    except Exception as e:
        logger.warning("Collection visualization failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Email pipeline: Gmail -> Markdown -> Vectorize -> Collection")
    parser.add_argument(
        "mode",
        choices=["poll", "batch", "collection"],
        help="poll: continuous polling; batch: one-time fetch; collection: rebuild collection only",
    )
    parser.add_argument(
        "--emails-root",
        type=Path,
        default=DEFAULT_EMAILS_ROOT,
        help="Root directory for output/emails",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="Poll interval in seconds (for poll mode)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Max messages to fetch (for batch mode)",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="label_ids",
        help="Gmail label IDs to filter (e.g. INBOX). Can repeat.",
    )
    args = parser.parse_args()

    # Ensure Ollama is running (needed for vectorization)
    if args.mode != "collection":
        ok, _ = check_ollama_running()
        if not ok:
            logger.error("Ollama server is not running. Start it with: ollama serve")
            sys.exit(1)

    if args.mode == "poll":
        run_polling_loop(
            args.emails_root,
            poll_interval=args.poll_interval,
            label_ids=args.label_ids,
        )
    elif args.mode == "batch":
        run_batch(
            args.emails_root,
            max_results=args.max_results,
            label_ids=args.label_ids or ["INBOX"],
        )
    else:
        run_build_collection_only(args.emails_root)


if __name__ == "__main__":
    main()
