"""Shared, durable OAuth support for JARVIS Google integrations.

OAuth grants belong to the Google account, not to an individual feature. Keeping
the lifecycle in one place prevents Gmail and Calendar from silently using
different stale tokens.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build


class GoogleAuthorizationError(RuntimeError):
    """An actionable Google OAuth configuration or authorization error."""


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_CONFIG_DIR = PROJECT_ROOT / "src" / "jarvis" / "config"


def _google_data_dir() -> Path:
    """Return a user-owned location that is safe from source-tree cleanup."""
    configured = os.environ.get("JARVIS_GOOGLE_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("APPDATA", Path.home())) / "JARVIS"


class GoogleOAuth:
    """Load, refresh, or obtain one service-specific OAuth grant."""

    def __init__(self, service_name: str, version: str, scopes: Sequence[str]) -> None:
        self.service_name = service_name
        self.version = version
        self.scopes = tuple(scopes)
        self.data_dir = _google_data_dir()
        self.token_path = self.data_dir / f"google_{service_name}_token.json"

    def _credential_candidates(self) -> list[Path]:
        explicit = os.environ.get("JARVIS_GOOGLE_CREDENTIALS_PATH")
        candidates = [self.data_dir / "credentials.json"]
        if explicit:
            candidates.insert(0, Path(explicit).expanduser())
        # Check project root config dir
        candidates.append(PROJECT_ROOT / "config" / "credentials.json")
        # Check project root directly
        candidates.append(PROJECT_ROOT / "credentials.json")
        # Check legacy config dir
        candidates.append(LEGACY_CONFIG_DIR / "credentials.json")
        return candidates

    def _token_candidates(self) -> list[Path]:
        # Retain legacy locations once so current users are migrated without a
        # needless sign-in. New/refreshed grants are always saved in AppData.
        legacy_name = "gmail_token.json" if self.service_name == "gmail" else "token.json"
        return [
            self.token_path,
            self.data_dir / legacy_name,
            PROJECT_ROOT / "config" / legacy_name,
            PROJECT_ROOT / legacy_name,
            LEGACY_CONFIG_DIR / legacy_name,
        ]

    def _load_usable_token(self) -> Credentials | None:
        for candidate in self._token_candidates():
            if not candidate.is_file():
                continue
            try:
                credentials = Credentials.from_authorized_user_file(str(candidate))
            except (OSError, ValueError):
                continue
            if not credentials.has_scopes(self.scopes):
                continue
            if credentials.valid:
                return credentials
            if credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                except RefreshError:
                    continue
                if credentials.valid and credentials.has_scopes(self.scopes):
                    return credentials
        return None

    def _client_secrets_path(self) -> Path:
        for candidate in self._credential_candidates():
            if candidate.is_file():
                return candidate
        searched = ", ".join(str(path) for path in self._credential_candidates())
        raise GoogleAuthorizationError(
            "Google OAuth client credentials were not found. Place the Desktop OAuth "
            f"credentials.json in {self.data_dir} (preferred) or set "
            f"JARVIS_GOOGLE_CREDENTIALS_PATH. Checked: {searched}."
        )

    def credentials(self, *, interactive: bool = True) -> Credentials | None:
        credentials = self._load_usable_token()
        if credentials is None:
            if not interactive:
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._client_secrets_path()), self.scopes
                )
                credentials = flow.run_local_server(port=0, open_browser=True)
            except GoogleAuthorizationError:
                raise
            except Exception as exc:
                raise GoogleAuthorizationError(
                    "Google authorization could not be completed. Ensure the Gmail and "
                    "Google Calendar APIs are enabled, this account is a consent-screen "
                    f"test user when applicable, then sign in again. Details: {exc}"
                ) from exc

        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.token_path.write_text(credentials.to_json(), encoding="utf-8")
        except OSError as exc:
            raise GoogleAuthorizationError(
                f"Google access was granted but JARVIS could not save its token at "
                f"{self.token_path}: {exc}"
            ) from exc
        return credentials

    def build_service(self, *, interactive: bool = True) -> Resource | None:
        credentials = self.credentials(interactive=interactive)
        if credentials is None:
            return None
        return build(
            self.service_name,
            self.version,
            credentials=credentials,
            cache_discovery=False,
        )
