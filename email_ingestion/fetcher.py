"""
Email Fetcher - Gmail API integration for real-time email retrieval.
Uses credentials.json for OAuth2; saves token.json after first auth.
Supports polling via users().history().list() for new messages.
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass
class AttachmentInfo:
    """Attachment metadata (no body downloaded by default)."""

    filename: str
    mime_type: str
    size: int
    attachment_id: str


@dataclass
class RawEmailMessage:
    """Parsed email message from Gmail API."""

    message_id: str
    thread_id: str
    label_ids: List[str]
    headers: Dict[str, str]
    subject: str
    from_addr: str
    to_addr: str
    cc_addr: str
    date: str
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    attachments: List[AttachmentInfo] = field(default_factory=list)
    raw_payload: Optional[Dict[str, Any]] = None


def _get_credentials(credentials_path: Path, token_path: Path):
    """Build OAuth2 credentials from credentials.json, refreshing with token.json."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Credentials file not found: {credentials_path}. "
                    "Place credentials.json from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        logger.info("OAuth token saved to %s", token_path)

    return creds


def _decode_body(data: Optional[str], encoding: str = "utf-8") -> str:
    """Decode base64url body. Gmail uses base64url."""
    if not data:
        return ""
    try:
        padded = data + "=" * (4 - len(data) % 4)
        raw = base64.urlsafe_b64decode(padded)
        return raw.decode(encoding, errors="replace")
    except Exception as e:
        logger.warning("Failed to decode body: %s", e)
        return ""


def _get_header(headers: List[Dict[str, str]], name: str) -> str:
    """Extract header value by name (case-insensitive)."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "").strip()
    return ""


def _extract_body(payload: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Extract text/plain and text/html body from MIME payload."""
    text_plain = None
    text_html = None

    mime_type = (payload.get("mimeType") or "").lower()
    body = payload.get("body")
    if body and body.get("data"):
        decoded = _decode_body(body.get("data"))
        if "text/plain" in mime_type:
            text_plain = decoded
        elif "text/html" in mime_type:
            text_html = decoded

    for part in payload.get("parts", []) or []:
        pt, ph = _extract_body(part)
        if pt and text_plain is None:
            text_plain = pt
        if ph and text_html is None:
            text_html = ph

    return text_plain, text_html


def _extract_attachments(payload: Dict[str, Any]) -> List[AttachmentInfo]:
    """Extract attachment metadata from payload parts."""
    attachments = []
    for part in payload.get("parts", []) or []:
        if part.get("body", {}).get("attachmentId"):
            attachments.append(
                AttachmentInfo(
                    filename=part.get("filename") or "unnamed",
                    mime_type=part.get("mimeType", "application/octet-stream"),
                    size=int(part.get("body", {}).get("size", 0) or 0),
                    attachment_id=part["body"]["attachmentId"],
                )
            )
        # Recurse into nested parts (e.g. multipart/alternative)
        attachments.extend(_extract_attachments(part))
    return attachments


def _parse_message(msg: Dict[str, Any]) -> RawEmailMessage:
    """Parse Gmail API message response into RawEmailMessage."""
    payload = msg.get("payload", {})
    headers_list = payload.get("headers", [])
    headers = {h.get("name", "").lower(): h.get("value", "") for h in headers_list}

    text_plain, text_html = _extract_body(payload)
    attachments = _extract_attachments(payload)

    return RawEmailMessage(
        message_id=msg["id"],
        thread_id=msg.get("threadId", ""),
        label_ids=msg.get("labelIds", []),
        headers=headers,
        subject=headers.get("subject", "(No Subject)"),
        from_addr=headers.get("from", ""),
        to_addr=headers.get("to", ""),
        cc_addr=headers.get("cc", ""),
        date=headers.get("date", ""),
        body_text=text_plain,
        body_html=text_html,
        attachments=attachments,
        raw_payload=payload,
    )


class GmailFetcher:
    """
    Fetches emails from Gmail API.
    Supports polling via history list for new messages.
    """

    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
    ):
        self.credentials_path = credentials_path or Path("credentials.json")
        self.token_path = token_path or Path("token.json")
        self._service = None

    def _get_service(self):
        if self._service is None:
            creds = _get_credentials(self.credentials_path, self.token_path)
            from googleapiclient.discovery import build

            self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def get_profile(self) -> Dict[str, Any]:
        """Get user profile (includes historyId for watch/polling)."""
        return self._get_service().users().getProfile(userId="me").execute()

    def list_messages(
        self,
        label_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        max_results: int = 50,
        page_token: Optional[str] = None,
    ) -> tuple[List[Dict[str, str]], Optional[str]]:
        """
        List message IDs. Returns (list of {id, threadId}, next_page_token).
        """
        params: Dict[str, Any] = {"userId": "me", "maxResults": max_results}
        if label_ids:
            params["labelIds"] = label_ids
        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token

        resp = self._get_service().users().messages().list(**params).execute()
        messages = resp.get("messages", []) or []
        return messages, resp.get("nextPageToken")

    def get_message(self, message_id: str, format: str = "full") -> Dict[str, Any]:
        """Fetch full message by ID."""
        return (
            self._get_service()
            .users()
            .messages()
            .get(userId="me", id=message_id, format=format)
            .execute()
        )

    def fetch_message(self, message_id: str) -> RawEmailMessage:
        """Fetch and parse a single message."""
        msg = self.get_message(message_id)
        return _parse_message(msg)

    def history_list(
        self,
        start_history_id: str,
        history_types: Optional[List[str]] = None,
        label_ids: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> tuple[List[str], Optional[str]]:
        """
        List changes since start_history_id.
        Returns (list of added message IDs, new_history_id).
        Gmail API supports labelId (singular); we use first label if provided.
        """
        params: Dict[str, Any] = {
            "userId": "me",
            "startHistoryId": start_history_id,
            "maxResults": max_results,
        }
        if history_types:
            params["historyTypes"] = history_types
        if label_ids and len(label_ids) > 0:
            params["labelId"] = label_ids[0]

        resp = self._get_service().users().history().list(**params).execute()
        message_ids = []
        for record in resp.get("history", []) or []:
            for added in record.get("messagesAdded", []) or []:
                mid = added.get("message", {}).get("id")
                if mid:
                    message_ids.append(mid)

        return message_ids, resp.get("historyId")

    def iter_new_messages(
        self,
        state_path: Path,
        label_ids: Optional[List[str]] = None,
        poll_interval_seconds: float = 60.0,
    ) -> Iterator[str]:
        """
        Iterator that yields new message IDs as they arrive.
        Uses state_path to persist historyId. Blocks between polls.
        """
        import time

        state = {}
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)

        history_id = state.get("historyId")
        if not history_id:
            profile = self.get_profile()
            history_id = profile.get("historyId")
            state["historyId"] = history_id
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info("Initialized historyId: %s", history_id)
            yield from []

        while True:
            try:
                ids, new_history_id = self.history_list(
                    start_history_id=history_id,
                    history_types=["messageAdded"],
                    label_ids=label_ids,
                )
                for mid in ids:
                    yield mid
                if new_history_id:
                    history_id = new_history_id
                    state["historyId"] = history_id
                    with open(state_path, "w") as f:
                        json.dump(state, f, indent=2)
            except Exception as e:
                logger.warning("History poll error: %s", e)
            time.sleep(poll_interval_seconds)
