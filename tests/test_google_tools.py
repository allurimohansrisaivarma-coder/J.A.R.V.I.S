"""Focused safety and configuration tests for Google integrations."""

import base64
from email import message_from_bytes
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jarvis.core.conversation import ConversationManager
from jarvis.core.session import SessionManager
from jarvis.tools.calendar_tool import CalendarOperationError, GoogleCalendarTool
from jarvis.tools.gmail import SCOPES, GmailOperationError, GoogleGmailTool
from jarvis.tools.google_auth import GoogleOAuth


def test_gmail_uses_minimum_draft_and_send_scope() -> None:
    assert "https://www.googleapis.com/auth/gmail.compose" in SCOPES
    assert "https://www.googleapis.com/auth/gmail.send" not in SCOPES


def test_gmail_draft_message_requires_a_real_recipient() -> None:
    tool = GoogleGmailTool()
    with pytest.raises(GmailOperationError, match="complete recipient"):
        tool._raw_message("Alex", "Subject", "Body")


def test_gmail_draft_message_does_not_set_invalid_from_header() -> None:
    tool = GoogleGmailTool()
    raw = tool._raw_message("alex@example.com", "Hello", "Body")
    message = message_from_bytes(base64.urlsafe_b64decode(raw))
    assert message["To"] == "alex@example.com"
    assert message["From"] is None
    assert message["Subject"] == "Hello"


def test_calendar_rejects_timezone_less_event_times() -> None:
    tool = GoogleCalendarTool()
    with pytest.raises(CalendarOperationError, match="missing a timezone"):
        tool.create_event("Stand-up", "2026-08-17T09:00:00", "2026-08-17T10:00:00")


def test_calendar_rejects_reversed_event_times() -> None:
    tool = GoogleCalendarTool()
    with pytest.raises(CalendarOperationError, match="after its start"):
        tool.create_event("Stand-up", "2026-08-17T10:00:00+04:00", "2026-08-17T09:00:00+04:00")


@pytest.mark.parametrize("confirmation", ["yes", "Yes, send it", "go ahead", "confirm send"])
def test_send_confirmation_is_unambiguous(confirmation: str) -> None:
    assert SessionManager._is_send_confirmation(confirmation)


@pytest.mark.parametrize(
    "non_confirmation", ["yes, add a calendar event", "send a new email", "maybe later"]
)
def test_send_confirmation_does_not_overreach(non_confirmation: str) -> None:
    assert not SessionManager._is_send_confirmation(non_confirmation)


def test_google_oauth_noninteractive_mode_does_not_open_consent(tmp_path, monkeypatch):
    oauth = GoogleOAuth("calendar", "v3", ("scope",))
    monkeypatch.setattr(oauth, "_token_candidates", lambda: [tmp_path / "missing-token.json"])

    assert oauth.credentials(interactive=False) is None


@pytest.mark.asyncio
async def test_email_send_request_creates_draft_then_requires_confirmation(monkeypatch):
    session = SessionManager()
    session.router = SimpleNamespace(
        generate_with_fallback=AsyncMock(
            return_value=SimpleNamespace(
                content=(
                    '{"to":"alex@example.com","subject":"Status",'
                    '"body":"The project is ready.","draft_id":""}'
                )
            )
        )
    )
    session.conversation = ConversationManager()
    session.conversation.new_conversation()

    monkeypatch.setattr(GoogleGmailTool, "authenticate", lambda self: None)
    monkeypatch.setattr(GoogleGmailTool, "get_recent_drafts", lambda self, count: "None")
    monkeypatch.setattr(
        GoogleGmailTool,
        "create_draft",
        lambda self, to, subject, body: (
            "Draft created successfully.\nDraft ID: draft-123\n"
            f"To: {to}\nSubject: {subject}\nBody: {body}"
        ),
    )

    preview = await session._build_google_context(
        "Send an email to alex@example.com with the project status", "now"
    )

    assert session._pending_email_draft_id == "draft-123"
    assert "has NOT been sent" in preview
    assert "explicit confirmation" in preview

    sent_ids: list[str] = []
    monkeypatch.setattr(
        GoogleGmailTool,
        "send_draft",
        lambda self, draft_id: sent_ids.append(draft_id) or "Email sent successfully.",
    )
    confirmation = await session._build_google_context("yes, send it", "now")

    assert sent_ids == ["draft-123"]
    assert session._pending_email_draft_id is None
    assert "was sent" in confirmation


@pytest.mark.asyncio
async def test_calendar_request_authenticates_and_returns_events(monkeypatch):
    session = SessionManager()
    session.router = SimpleNamespace()
    session.conversation = ConversationManager()
    session.conversation.new_conversation()
    authenticated: list[bool] = []

    monkeypatch.setattr(
        GoogleCalendarTool,
        "authenticate",
        lambda self, interactive=True: authenticated.append(interactive) or True,
    )
    monkeypatch.setattr(
        GoogleCalendarTool,
        "get_upcoming_events",
        lambda self, count: "Upcoming events:\n- 2026-08-22T09:00:00+04:00: Stand-up",
    )

    context = await session._build_google_context("What's on my calendar?", "now")

    assert authenticated == [True]
    assert "Stand-up" in context
