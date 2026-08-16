"""Tests for safe, useful local-file context."""

import pytest

from jarvis.context.file_source import FileContextSource
import jarvis.context.file_source as file_source_module


@pytest.mark.asyncio
async def test_cv_search_returns_paths_without_binary_content(tmp_path, monkeypatch):
    """A filename search should report matches, not inject document bytes."""
    cv_path = tmp_path / "mycv.pdf"
    cv_path.write_bytes(b"%PDF-1.7\nraw document bytes")
    monkeypatch.setattr(file_source_module, "ALLOWED_ROOTS", [tmp_path])

    result = await FileContextSource().gather_context("Where is my CV saved?")

    assert str(cv_path) in result
    assert "%PDF" not in result
