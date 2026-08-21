"""Tests for safe, useful local-file context."""

import pytest

import jarvis.context.file_source as file_source_module
from jarvis.context.file_source import FileContextSource


@pytest.mark.asyncio
async def test_cv_search_returns_paths_without_binary_content(tmp_path, monkeypatch):
    """A filename search should report matches, not inject document bytes."""
    cv_path = tmp_path / "mycv.pdf"
    cv_path.write_bytes(b"%PDF-1.7\nraw document bytes")
    monkeypatch.setattr(file_source_module, "ALLOWED_ROOTS", [tmp_path])

    result = await FileContextSource().gather_context("Where is my CV saved?")

    assert str(cv_path) in result
    assert "%PDF" not in result


@pytest.mark.asyncio
async def test_explicit_path_outside_approved_roots_is_denied(tmp_path, monkeypatch):
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    monkeypatch.setattr(file_source_module, "ALLOWED_ROOTS", [approved])

    result = await FileContextSource().gather_context(f'read "{outside}"')

    assert "outside" in result.lower()
    assert "private" not in result


@pytest.mark.asyncio
async def test_secret_file_contents_are_never_returned(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=do-not-leak", encoding="utf-8")
    monkeypatch.setattr(file_source_module, "ALLOWED_ROOTS", [tmp_path])

    result = await FileContextSource().gather_context(f'read "{env_file}"')

    assert "protected" in result.lower()
    assert "do-not-leak" not in result


@pytest.mark.asyncio
async def test_explicit_safe_text_file_can_be_read(tmp_path, monkeypatch):
    note = tmp_path / "project notes.md"
    note.write_text("ship it", encoding="utf-8")
    monkeypatch.setattr(file_source_module, "ALLOWED_ROOTS", [tmp_path])

    result = await FileContextSource().gather_context(f'read "{note}"')

    assert "ship it" in result
