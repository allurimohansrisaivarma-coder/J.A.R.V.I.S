"""Google Calendar operations with validated, timezone-aware event creation."""

from __future__ import annotations

from datetime import UTC, datetime

from googleapiclient.errors import HttpError

from jarvis.tools.google_auth import GoogleOAuth

SCOPES = ("https://www.googleapis.com/auth/calendar",)


class CalendarOperationError(RuntimeError):
    """A Calendar request that did not complete."""


class GoogleCalendarTool:
    def __init__(self) -> None:
        self.service = None
        self.timezone = "UTC"

    def authenticate(self, interactive: bool = True) -> bool:
        if self.service is None:
            self.service = GoogleOAuth("calendar", "v3", SCOPES).build_service(
                interactive=interactive
            )
            if self.service is None:
                return False
            try:
                service = self.service
                assert service is not None
                self.timezone = (
                    service.settings().get(setting="timezone").execute().get("value", "UTC")
                )
            except HttpError:
                # Calendar accepts RFC3339 offsets even if this optional lookup fails.
                self.timezone = "UTC"
        return True

    def _service(self):
        self.authenticate()
        assert self.service is not None
        return self.service

    @staticmethod
    def _parse_datetime(value: str, name: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value)
        except (AttributeError, ValueError) as exc:
            raise CalendarOperationError(
                f"{name} must be an ISO-8601 date and time, for example 2026-08-16T15:00:00+04:00."
            ) from exc
        if parsed.tzinfo is None:
            raise CalendarOperationError(
                f"{name} is missing a timezone. JARVIS must include the user's calendar timezone."
            )
        return parsed

    def get_upcoming_events(self, max_results: int = 10) -> str:
        try:
            events = (
                self._service()
                .events()
                .list(
                    calendarId="primary",
                    timeMin=datetime.now(UTC).isoformat(),
                    maxResults=max(1, min(max_results, 25)),
                    singleEvents=True,
                    orderBy="startTime",
                    timeZone=self.timezone,
                )
                .execute()
                .get("items", [])
            )
            if not events:
                return "No upcoming events found."
            return "Upcoming events:\n" + "\n".join(
                f"- {item.get('start', {}).get('dateTime', item.get('start', {}).get('date'))}: "
                f"{item.get('summary', 'No Title')}"
                for item in events
            )
        except HttpError as error:
            raise CalendarOperationError(
                f"Calendar could not list upcoming events: {error}"
            ) from error

    def create_event(
        self, summary: str, start_time: str, end_time: str = "", description: str = ""
    ) -> str:
        start = self._parse_datetime(start_time, "Start time")
        if not end_time:
            from datetime import timedelta

            end = start + timedelta(hours=1)
        else:
            end = self._parse_datetime(end_time, "End time")

        if end <= start:
            raise CalendarOperationError("The event end time must be after its start time.")
        if not (summary or "").strip():
            raise CalendarOperationError("A calendar event needs a title.")
        event = {
            "summary": summary.strip(),
            "description": description or "",
            "start": {"dateTime": start.isoformat(), "timeZone": self.timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": self.timezone},
        }
        try:
            created = self._service().events().insert(calendarId="primary", body=event).execute()
            return f"Event created: {created.get('htmlLink', created.get('id', 'confirmed'))}"
        except HttpError as error:
            raise CalendarOperationError(f"Calendar could not create the event: {error}") from error
