"""Email pipeline: Gmail fetch, markdown conversion, collection building."""

from email_ingestion.fetcher import AttachmentInfo, GmailFetcher, RawEmailMessage
from email_ingestion.to_markdown import email_id_from_message, save_email_markdown, to_markdown
from email_ingestion.collection import append_email_to_collection, build_collection

__all__ = [
    "AttachmentInfo",
    "RawEmailMessage",
    "GmailFetcher",
    "email_id_from_message",
    "save_email_markdown",
    "to_markdown",
    "append_email_to_collection",
    "build_collection",
]
