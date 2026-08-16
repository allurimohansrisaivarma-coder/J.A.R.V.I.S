"""Pytest fixtures for the Jarvis test suite."""

from pathlib import Path
from typing import Generator

import pytest

from jarvis.config.settings import Settings


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment variables for testing."""
    monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("JARVIS_GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("JARVIS_SYSTEM__PORT", "9999")


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary configuration directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def settings(mock_env: None) -> Settings:
    """Provide a test Settings instance with mock keys."""
    return Settings(
        gemini_api_key="test-gemini-key",
        groq_api_key="test-groq-key"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as requiring integration with external services"
    )
