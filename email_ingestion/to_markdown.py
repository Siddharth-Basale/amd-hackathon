"""
Email to Markdown - Converts raw Gmail messages to expandable markdown.
Uses <details>/<summary> for collapsible sections (GitHub markdown compatible).
Preserves hierarchy: Subject, Metadata, Body, Attachments.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

from email_ingestion.fetcher import AttachmentInfo, RawEmailMessage

logger = logging.getLogger(__name__)


def slugify(text: str, max_len: int = 50) -> str:
    """Create safe filename slug from text."""
    s = re.sub(r"[^\w\s-]", "", text)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:max_len] if max_len else s


def email_id_from_message(message_id: str, subject: str = "", date_str: str = "") -> str:
    """
    Generate stable email_id. Plan recommends msg_{message_id} for reliability.
    """
    return f"msg_{message_id}"


def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown, preserving structure (headings, lists, tables)."""
    try:
        import html2text

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0
        h.ignore_emphasis = False
        h.single_line_break = False
        return h.handle(html or "")
    except ImportError:
        logger.warning("html2text not installed; using plain text fallback")
        return _strip_html(html or "")


def _strip_html(html: str) -> str:
    """Fallback: strip HTML tags."""
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def to_markdown(msg: RawEmailMessage, use_collapsible: bool = True) -> str:
    """
    Convert RawEmailMessage to markdown with expandable hierarchy.
    - # Subject (top level)
    - ## Metadata (collapsible table: From, To, Cc, Date, Labels)
    - ## Body (main content)
    - ## Attachments (collapsible list)
    """
    lines: List[str] = []
    subject = msg.subject or "(No Subject)"
    lines.append(f"# {subject}\n")

    # Metadata section - collapsible
    meta_rows = [
        ("From", msg.from_addr),
        ("To", msg.to_addr),
        ("Cc", msg.cc_addr),
        ("Date", msg.date),
        ("Labels", ", ".join(msg.label_ids) if msg.label_ids else "-"),
    ]
    meta_table = "\n".join(f"| {k} | {v} |" for k, v in meta_rows)
    meta_table = "| Field | Value |\n| --- | --- |\n" + meta_table
    if use_collapsible:
        lines.append("<details>\n<summary>Metadata</summary>\n\n")
        lines.append(meta_table)
        lines.append("\n\n</details>\n\n")
    else:
        lines.append("## Metadata\n\n")
        lines.append(meta_table)
        lines.append("\n\n")

    # Body section
    body_text = msg.body_text
    body_html = msg.body_html
    if body_text:
        body_content = body_text
    elif body_html:
        body_content = html_to_markdown(body_html)
    else:
        body_content = "(No body content)"

    lines.append("## Body\n\n")
    lines.append(body_content.strip())
    lines.append("\n\n")

    # Attachments section - collapsible
    if msg.attachments:
        att_lines = []
        for a in msg.attachments:
            att_lines.append(f"- **{a.filename}** — {a.mime_type}, {a.size} bytes")
        att_content = "\n".join(att_lines)
        if use_collapsible:
            lines.append("<details>\n<summary>Attachments</summary>\n\n")
            lines.append(att_content)
            lines.append("\n\n</details>\n")
        else:
            lines.append("## Attachments\n\n")
            lines.append(att_content)
            lines.append("\n")
    else:
        lines.append("## Attachments\n\n*(none)*\n")

    return "".join(lines)


def save_email_markdown(
    msg: RawEmailMessage,
    output_dir: Path,
    email_id: Optional[str] = None,
) -> Path:
    """
    Convert email to markdown and save to output_dir/<email_id>/<email_id>.md.
    Returns path to saved .md file.
    """
    email_id = email_id or email_id_from_message(msg.message_id)
    md_content = to_markdown(msg)
    out_folder = output_dir / email_id
    out_folder.mkdir(parents=True, exist_ok=True)
    md_path = out_folder / f"{email_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved markdown to %s", md_path)
    return md_path
