"""Tests for deterministic Windows application launching."""

from pathlib import Path
from unittest.mock import MagicMock

from jarvis.tools import windows_launcher


def test_voice_launch_phrases_are_parsed_without_an_llm():
    expected = {
        "Hey, Davis, open Chrome for me.": "chrome",
        "Open any browser.": "browser",
        "Java is open calculator": "calculator",
        "Javis open settings": "settings",
        "Service Open Task Manager": "task manager",
        "Open anti-gravity": "anti-gravity",
        "Open teams": "teams",
    }

    for phrase, app_name in expected.items():
        request = windows_launcher.parse_launch_request(phrase)
        assert request is not None
        assert request.app_name == app_name


def test_dashboard_is_not_treated_as_an_installed_application():
    assert windows_launcher.parse_launch_request("Open the global dashboard") is None


def test_browser_resolution_checks_standard_install_directories(tmp_path, monkeypatch):
    chrome = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
    chrome.parent.mkdir(parents=True)
    chrome.touch()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "missing-program-files"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "missing-x86"))
    monkeypatch.setattr(windows_launcher.shutil, "which", lambda _name: None)

    display, executable = windows_launcher.resolve_application("chrome")

    assert display == "chrome"
    assert Path(executable) == chrome


def test_launch_uses_resolved_executable_without_a_shell(monkeypatch):
    popen = MagicMock()
    monkeypatch.setattr(
        windows_launcher,
        "resolve_application",
        lambda _name: ("Google Chrome", r"C:\Program Files\Google\Chrome\chrome.exe"),
    )
    monkeypatch.setattr(windows_launcher.subprocess, "Popen", popen)

    result = windows_launcher.launch_application("chrome", "https://example.com")

    assert result == "Launched Google Chrome successfully."
    command = popen.call_args.args[0]
    assert command == [r"C:\Program Files\Google\Chrome\chrome.exe", "https://example.com"]
    assert "shell" not in popen.call_args.kwargs


def test_installed_start_menu_apps_and_pwas_are_discovered(tmp_path, monkeypatch):
    start_menu = tmp_path / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    antigravity = start_menu / "Antigravity.lnk"
    teams = start_menu / "Chrome Apps" / "Microsoft Teams (PWA).lnk"
    teams.parent.mkdir(parents=True)
    antigravity.touch()
    teams.touch()
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("ProgramData", str(tmp_path / "empty"))
    windows_launcher._start_menu_shortcuts.cache_clear()

    assert Path(windows_launcher.resolve_application("anti-gravity")[1]) == antigravity
    assert Path(windows_launcher.resolve_application("teams")[1]) == teams

    windows_launcher._start_menu_shortcuts.cache_clear()


def test_start_menu_shortcut_is_launched_through_windows_shell(monkeypatch):
    startfile = MagicMock()
    monkeypatch.setattr(
        windows_launcher,
        "resolve_application",
        lambda _name: ("Microsoft Teams", r"C:\Start Menu\Microsoft Teams.lnk"),
    )
    monkeypatch.setattr(windows_launcher.os, "startfile", startfile)

    result = windows_launcher.launch_application("teams")

    assert result == "Launched Microsoft Teams successfully."
    startfile.assert_called_once_with(r"C:\Start Menu\Microsoft Teams.lnk")
