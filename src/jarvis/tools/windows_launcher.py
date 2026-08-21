"""Safe, deterministic Windows application discovery and launching."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class LaunchRequest:
    """A locally recognized application launch command."""

    app_name: str
    arguments: str = ""


_ALIASES = {
    "anti gravity": "antigravity",
    "antigravity ide": "antigravity ide",
    "microsoft teams": "teams",
    "teams app": "teams",
    "google chrome": "chrome",
    "chrome browser": "chrome",
    "microsoft edge": "edge",
    "edge browser": "edge",
    "mozilla firefox": "firefox",
    "firefox browser": "firefox",
    "any browser": "browser",
    "default browser": "browser",
    "web browser": "browser",
    "calculator": "calculator",
    "calc": "calculator",
    "task manager": "task manager",
    "control panel": "control panel",
    "windows settings": "settings",
    "settings": "settings",
    "file explorer": "explorer",
    "windows explorer": "explorer",
    "explorer": "explorer",
    "command prompt": "command prompt",
    "cmd": "command prompt",
    "powershell": "powershell",
    "terminal": "terminal",
    "notepad": "notepad",
}

_EXECUTABLES = {
    "calculator": ("calc.exe",),
    "task manager": ("taskmgr.exe",),
    "control panel": ("control.exe",),
    "explorer": ("explorer.exe",),
    "command prompt": ("cmd.exe",),
    "powershell": ("powershell.exe",),
    "terminal": ("wt.exe", "powershell.exe"),
    "notepad": ("notepad.exe",),
}

_SITE_URLS = {
    "gmail": "https://mail.google.com/",
    "google": "https://www.google.com/",
    "youtube": "https://www.youtube.com/",
    "calendar": "https://calendar.google.com/",
}

_NON_APPLICATION_TARGETS = {
    "file",
    "folder",
    "document",
    "website",
    "webpage",
    "page",
    "tab",
    "link",
    "url",
}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


@lru_cache(maxsize=1)
def _start_menu_shortcuts() -> tuple[tuple[str, Path], ...]:
    """Index registered Start Menu apps once per process."""
    roots = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        Path(os.environ.get("ProgramData", "")) / "Microsoft/Windows/Start Menu/Programs",
    ]
    shortcuts: list[tuple[str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for shortcut in root.rglob("*.lnk"):
            normalized = _normalize_name(shortcut.stem)
            if normalized and "uninstall" not in normalized:
                shortcuts.append((normalized, shortcut))
    return tuple(shortcuts)


def _find_start_menu_shortcut(app_name: str) -> Path | None:
    query = _normalize_name(app_name)
    query_tokens = set(query.split()) - {"app", "application", "desktop"}
    if not query_tokens:
        return None

    matches: list[tuple[int, int, Path]] = []
    for registered_name, shortcut in _start_menu_shortcuts():
        registered_tokens = set(registered_name.split()) - {"app", "application", "pwa"}
        if query == registered_name:
            return shortcut
        if query_tokens.issubset(registered_tokens):
            matches.append(
                (
                    len(registered_tokens - query_tokens),
                    len(registered_name),
                    shortcut,
                )
            )
    return min(matches, default=(0, 0, None))[-1]


def _browser_candidates(browser: str) -> list[Path | str]:
    program_files = os.environ.get("ProgramFiles", "")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates: dict[str, list[Path | str]] = {
        "chrome": [
            "chrome.exe",
            Path(program_files) / "Google/Chrome/Application/chrome.exe",
            Path(program_files_x86) / "Google/Chrome/Application/chrome.exe",
            Path(local_app_data) / "Google/Chrome/Application/chrome.exe",
        ],
        "edge": [
            "msedge.exe",
            Path(program_files) / "Microsoft/Edge/Application/msedge.exe",
            Path(program_files_x86) / "Microsoft/Edge/Application/msedge.exe",
            Path(local_app_data) / "Microsoft/Edge/Application/msedge.exe",
        ],
        "firefox": [
            "firefox.exe",
            Path(program_files) / "Mozilla Firefox/firefox.exe",
            Path(program_files_x86) / "Mozilla Firefox/firefox.exe",
        ],
    }
    if browser == "browser":
        return candidates["chrome"] + candidates["edge"] + candidates["firefox"]
    return candidates.get(browser, [])


def _resolve(candidates: list[Path | str]) -> str | None:
    for candidate in candidates:
        value = str(candidate)
        if not value or value == ".":
            continue
        resolved = shutil.which(value)
        if resolved:
            return resolved
        path = Path(value)
        if path.is_file():
            return str(path)
    return None


def resolve_application(app_name: str) -> tuple[str, str]:
    """Return ``(display_name, executable_or_uri)`` for a safe known application."""
    normalized = _normalize_name(app_name)
    normalized_aliases = {_normalize_name(alias): target for alias, target in _ALIASES.items()}
    normalized = normalized_aliases.get(normalized, normalized)

    if normalized == "settings":
        return "Windows Settings", "ms-settings:"
    if normalized in {"chrome", "edge", "firefox", "browser"}:
        executable = _resolve(_browser_candidates(normalized))
        if executable:
            display = Path(executable).stem.replace("msedge", "Microsoft Edge")
            return display, executable
        raise FileNotFoundError("No supported browser installation was found.")
    if normalized in _EXECUTABLES:
        executable = _resolve(list(_EXECUTABLES[normalized]))
        if executable:
            return normalized.title(), executable
        raise FileNotFoundError(f"Windows could not locate {normalized}.")

    shortcut = _find_start_menu_shortcut(normalized)
    if shortcut:
        return shortcut.stem, str(shortcut)
    raise FileNotFoundError(f"Windows could not find an installed application named '{app_name}'.")


def launch_application(app_name: str, arguments: str = "") -> str:
    """Launch a known Windows application without a shell."""
    display_name, target = resolve_application(app_name)
    if target == "ms-settings:" or Path(target).suffix.casefold() == ".lnk":
        os.startfile(target)
    else:
        command = [target]
        if arguments.strip():
            command.extend(shlex.split(arguments, posix=False))
        subprocess.Popen(
            command,
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
    return f"Launched {display_name} successfully."


def parse_launch_request(text: str) -> LaunchRequest | None:
    """Recognize explicit voice/text launch commands without involving an LLM."""
    normalized = re.sub(r"[^a-z0-9:/.?=&_-]+", " ", text.casefold()).strip()
    if not re.search(r"\b(?:open|launch|start|run)\b", normalized):
        return None

    browser_hint = next(
        (name for name in ("chrome", "edge", "firefox") if re.search(rf"\b{name}\b", normalized)),
        None,
    )
    for site, url in _SITE_URLS.items():
        if re.search(rf"\b{site}\b", normalized) and (
            browser_hint or "browser" in normalized or normalized.startswith("open ")
        ):
            return LaunchRequest(browser_hint or "browser", url)

    # Longest aliases first prevents "settings" from beating "windows settings".
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return LaunchRequest(_ALIASES[alias])
    for browser in ("chrome", "edge", "firefox"):
        if re.search(rf"\b{browser}\b", normalized):
            return LaunchRequest(browser)

    command = re.search(r"\b(?:open|launch|start|run)\b\s+(?:the\s+)?(.+)$", normalized)
    if command:
        candidate = re.sub(
            r"\b(?:please|for me|right now|now|sir)\b.*$",
            "",
            command.group(1),
        ).strip()
        candidate_tokens = set(candidate.split())
        if candidate and not candidate_tokens.intersection(_NON_APPLICATION_TARGETS):
            return LaunchRequest(candidate)
    return None
