"""Focused safety and configuration tests for Google integrations."""

import base64
from email import message_from_bytes

import pytest

from jarvis.core.session import SessionManager
from jarvis.tools.calendar import CalendarOperationError, GoogleCalendarTool
from jarvis.tools.gmail import SCOPES, GmailOperationError, GoogleGmailTool


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
        tool.create_event(
            "Stand-up", "2026-08-17T10:00:00+04:00", "2026-08-17T09:00:00+04:00"
        )


@pytest.mark.parametrize("confirmation", ["yes", "Yes, send it", "go ahead", "confirm send"])
def test_send_confirmation_is_unambiguous(confirmation: str) -> None:
    assert SessionManager._is_send_confirmation(confirmation)


@pytest.mark.parametrize("non_confirmation", ["yes, add a calendar event", "send a new email", "maybe later"])
def test_send_confirmation_does_not_overreach(non_confirmation: str) -> None:
    assert not SessionManager._is_send_confirmation(non_confirmation)
