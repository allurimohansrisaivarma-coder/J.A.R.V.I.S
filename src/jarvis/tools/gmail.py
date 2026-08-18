"""Gmail operations with validated inputs and durable OAuth."""

from __future__ import annotations

import base64
from email.message import EmailMessage
from email.utils import getaddresses

from googleapiclient.errors import HttpError

from jarvis.tools.google_auth import GoogleOAuth

# gmail.compose includes drafting and sending. Requesting gmail.send as a
# second scope forced needless re-consent for existing compose grants.
SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
)


class GmailOperationError(RuntimeError):
    """A Gmail request that did not complete."""


def _decode(data: str) -> str:
    if not data:
        return ""
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def _plain_text(payload: dict) -> str:
    if payload.get("mimeType") == "text/plain":
        body = _decode(payload.get("body", {}).get("data", ""))
        if body:
            return body
    for part in payload.get("parts", []):
        body = _plain_text(part)
        if body:
            return body
    return ""


def _headers(payload: dict) -> dict[str, str]:
    return {item["name"].lower(): item["value"] for item in payload.get("headers", [])}


class GoogleGmailTool:
    def __init__(self) -> None:
        self.service = None

    def authenticate(self) -> None:
        if self.service is None:
            self.service = GoogleOAuth("gmail", "v1", SCOPES).build_service()

    def _service(self):
        self.authenticate()
        assert self.service is not None
        return self.service

    @staticmethod
    def _raise(operation: str, error: HttpError) -> None:
        detail = error.reason if getattr(error, "reason", None) else str(error)
        raise GmailOperationError(f"Gmail could not {operation}: {detail}") from error

    @staticmethod
    def _validate_recipients(to: str) -> str:
        addresses = [address for _, address in getaddresses([to]) if "@" in address]
        if not addresses:
            raise GmailOperationError(
                "A draft needs at least one complete recipient email address before JARVIS can save it."
            )
        return ", ".join(addresses)

    def get_emails(self, query: str = "is:unread", max_results: int = 5) -> str:
        try:
            results = self._service().users().messages().list(
                userId="me", q=query, maxResults=max(1, min(max_results, 20))
            ).execute()
            messages = results.get("messages", [])
            if not messages:
                return f"No emails found matching query: {query}"
            summaries = []
            for message in messages:
                item = self._service().users().messages().get(
                    userId="me", id=message["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                ).execute()
                headers = _headers(item.get("payload", {}))
                summaries.append(
                    f"From: {headers.get('from', 'Unknown Sender')}\n"
                    f"Subject: {headers.get('subject', 'No Subject')}\n"
                    f"Date: {headers.get('date', 'Unknown Date')}\n"
                    f"Snippet: {item.get('snippet', '')}"
                )
            return f"Emails matching '{query}':\n\n" + "\n---\n".join(summaries)
        except HttpError as error:
            self._raise("read email", error)

    def get_draft(self, draft_id: str) -> str:
        try:
            draft = self._service().users().drafts().get(
                userId="me", id=draft_id, format="full"
            ).execute()
            message = draft.get("message", {})
            payload = message.get("payload", {})
            headers = _headers(payload)
            body = _plain_text(payload) or message.get("snippet", "")
            return (
                f"Draft ID: {draft_id}\nTo: {headers.get('to', 'Unknown Recipient')}\n"
                f"Subject: {headers.get('subject', 'No Subject')}\nBody: {body}"
            )
        except HttpError as error:
            self._raise("fetch draft", error)

    def get_recent_drafts(self, max_results: int = 5) -> str:
        try:
            drafts = self._service().users().drafts().list(
                userId="me", maxResults=max(1, min(max_results, 20))
            ).execute().get("drafts", [])
            if not drafts:
                return "No existing drafts found."
            details = []
            for draft in drafts:
                item = self._service().users().drafts().get(
                    userId="me", id=draft["id"], format="metadata"
                ).execute()
                headers = _headers(item.get("message", {}).get("payload", {}))
                details.append(
                    f"Draft ID: {draft['id']} | To: {headers.get('to', 'Unknown Recipient')} | "
                    f"Subject: {headers.get('subject', 'No Subject')}"
                )
            return "\n".join(details)
        except HttpError as error:
            self._raise("list drafts", error)

    def _raw_message(self, to: str, subject: str, body: str) -> str:
        message = EmailMessage()
        message["To"] = self._validate_recipients(to)
        message["Subject"] = (subject or "No Subject").strip()
        message.set_content(body or "")
        return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    def create_draft(self, to: str, subject: str, body: str) -> str:
        if not body or len(body.strip()) < 2:
            raise GmailOperationError("JARVIS requires the complete email body to draft the email. Do not leave the body empty.")
        try:
            draft = self._service().users().drafts().create(userId="me", body={
                "message": {"raw": self._raw_message(to, subject, body)}
            }).execute()
            return f"Draft created successfully.\n{self.get_draft(draft['id'])}"
        except HttpError as error:
            self._raise("create draft", error)

    def update_draft(self, draft_id: str, to: str, subject: str, body: str) -> str:
        if not draft_id.strip():
            raise GmailOperationError("JARVIS needs the draft ID before it can update a draft.")
        if not body or len(body.strip()) < 2:
            raise GmailOperationError("JARVIS requires the complete email body to update the draft. Do not leave the body empty.")
        try:
            draft = self._service().users().drafts().update(userId="me", id=draft_id, body={
                "message": {"raw": self._raw_message(to, subject, body)}
            }).execute()
            return f"Draft updated successfully.\n{self.get_draft(draft['id'])}"
        except HttpError as error:
            self._raise("update draft", error)

    def send_draft(self, draft_id: str) -> str:
        if not draft_id.strip():
            raise GmailOperationError("JARVIS needs an explicit draft ID before sending an email.")
        try:
            self._service().users().drafts().get(userId="me", id=draft_id).execute()
            message = self._service().users().drafts().send(
                userId="me", body={"id": draft_id}
            ).execute()
            return f"Email sent successfully. Message ID: {message['id']}"
        except HttpError as error:
            self._raise("send draft", error)
