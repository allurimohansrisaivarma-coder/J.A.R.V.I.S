"""Tests for configuration system."""

import os

import pytest
from pydantic import ValidationError

from jarvis.config.settings import (
    GeminiSettings,
    GroqSettings,
    LLMSettings,
    LoggingSettings,
    RouterSettings,
    Settings,
)


class TestSettingsModels:
    """Test nested settings models."""

    def test_gemini_defaults(self):
        """Verify Gemini settings have correct defaults."""
        settings = GeminiSettings()
        assert settings.model_flash == "gemini-1.5-flash"
        assert settings.model_pro == "gemini-1.5-pro"
        assert settings.temperature == 0.7
        assert settings.max_output_tokens == 2048

    def test_groq_defaults(self):
        """Verify Groq settings have correct defaults."""
        settings = GroqSettings()
        assert settings.model == "llama-3.1-8b-instant"
        assert settings.temperature == 0.3

    def test_router_defaults(self):
        """Verify router settings have correct defaults."""
        settings = RouterSettings()
        assert settings.complexity_threshold == 0.8
        assert "hi" in settings.fast_keywords
        assert "hello" in settings.fast_keywords

    def test_logging_defaults(self):
        """Verify logging settings have correct defaults."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.format == "json"
        assert settings.max_size_mb == 10

    def test_llm_nested_settings(self):
        """Verify LLM settings contain nested Gemini and Groq."""
        settings = LLMSettings()
        assert isinstance(settings.gemini, GeminiSettings)
        assert isinstance(settings.groq, GroqSettings)
        assert settings.default_provider == "gemini"


class TestMainSettings:
    """Test the main Settings class."""

    def test_settings_with_valid_keys(self):
        """Settings should load when API keys are provided."""
        settings = Settings(
            gemini_api_key="test-gemini-key-123",
            groq_api_key="test-groq-key-456",
        )
        assert settings.gemini_api_key == "test-gemini-key-123"
        assert settings.groq_api_key == "test-groq-key-456"

    def test_settings_rejects_empty_keys(self):
        """Settings should reject empty API key strings."""
        with pytest.raises(ValidationError):
            Settings(gemini_api_key="", groq_api_key="test-key")

    def test_settings_rejects_whitespace_keys(self):
        """Settings should reject whitespace-only API key strings."""
        with pytest.raises(ValidationError):
            Settings(gemini_api_key="   ", groq_api_key="test-key")

    def test_settings_env_prefix(self, monkeypatch):
        """Settings should read from JARVIS_ prefixed env vars."""
        monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "env-gemini-key")
        monkeypatch.setenv("JARVIS_GROQ_API_KEY", "env-groq-key")
        settings = Settings()
        assert settings.gemini_api_key == "env-gemini-key"
        assert settings.groq_api_key == "env-groq-key"

    def test_settings_nested_env_override(self, monkeypatch):
        """Nested settings should be overridable via env vars with delimiter."""
        monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "key1")
        monkeypatch.setenv("JARVIS_GROQ_API_KEY", "key2")
        monkeypatch.setenv("JARVIS_SYSTEM__PORT", "9999")
        settings = Settings()
        assert settings.system.port == 9999

    def test_settings_default_port(self):
        """Default system port should be 8741."""
        settings = Settings(
            gemini_api_key="test-key",
            groq_api_key="test-key",
        )
        assert settings.system.port == 8741
