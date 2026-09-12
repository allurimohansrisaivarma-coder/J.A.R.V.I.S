"""Tests for configuration system."""

import pytest
import yaml

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
        assert settings.model_flash == "gemini-3.7-flash"
        assert settings.model_pro == "gemini-3.7-flash"
        assert settings.temperature == 0.7
        assert settings.max_output_tokens == 2048

    def test_groq_defaults(self):
        """Verify Groq settings have correct defaults."""
        settings = GroqSettings()
        assert settings.model == "openai/gpt-oss-20b"
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
        assert settings.max_size_mb == 2

    def test_llm_nested_settings(self):
        """Verify LLM settings contain nested Gemini and Groq."""
        settings = LLMSettings()
        assert isinstance(settings.gemini, GeminiSettings)
        assert isinstance(settings.groq, GroqSettings)
        assert settings.default_provider == "groq"


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

    def test_settings_accepts_optional_empty_key(self):
        settings = Settings(gemini_api_key="", groq_api_key="test-key")
        assert settings.gemini_api_keys == []
        assert settings.primary_groq_key == "test-key"

    def test_settings_accepts_whitespace_as_unconfigured(self):
        settings = Settings(gemini_api_key="   ", groq_api_key="test-key")
        assert settings.gemini_api_keys == []

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
        assert settings.system.weather_city == ""

    def test_environment_overrides_yaml(self, tmp_path, monkeypatch):
        """Environment values must have higher priority than custom YAML."""
        config = tmp_path / "jarvis.yaml"
        config.write_text("system:\n  port: 8000\n", encoding="utf-8")
        monkeypatch.setenv("JARVIS_SYSTEM__PORT", "9999")
        monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "env-gemini")
        monkeypatch.setenv("JARVIS_GROQ_API_KEY", "env-groq")

        settings = Settings.from_yaml(config, env_file=tmp_path / "missing.env")

        assert settings.system.port == 9999

    def test_save_to_yaml_does_not_persist_api_keys(self, tmp_path):
        """Saving UI configuration must never copy API secrets to disk."""
        output = tmp_path / "saved.yaml"
        settings = Settings(gemini_api_key="gemini-secret", groq_api_key="groq-secret")

        settings.save_to_yaml(output)
        saved = yaml.safe_load(output.read_text(encoding="utf-8"))

        assert "gemini_api_keys" not in saved
        assert "groq_api_keys" not in saved
        serialized = output.read_text(encoding="utf-8")
        assert "gemini-secret" not in serialized
        assert "groq-secret" not in serialized

    def test_invalid_yaml_root_has_clear_error(self, tmp_path):
        config = tmp_path / "jarvis.yaml"
        config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

        with pytest.raises(TypeError, match="root must be a mapping"):
            Settings.from_yaml(config, env_file=tmp_path / "missing.env")
