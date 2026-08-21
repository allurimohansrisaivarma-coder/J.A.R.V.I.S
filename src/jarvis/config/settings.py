"""Application settings and configuration management."""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(vars(sys)["_MEIPASS"])
    APP_DIR = Path(sys.executable).parent
    PROJECT_ROOT = APP_DIR
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    APP_DIR = BUNDLE_DIR
    PROJECT_ROOT = BUNDLE_DIR


class GeminiSettings(BaseModel):
    """Settings for Google Gemini models."""

    model_flash: str = "gemini-3.7-flash"
    model_pro: str = "gemini-3.7-flash"
    thinking_level_standard: str = "low"
    thinking_level_complex: str = "high"
    temperature: float = 0.7
    max_output_tokens: int = 2048


class GroqSettings(BaseModel):
    """Settings for Groq inference."""

    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.3
    max_output_tokens: int = 1024


class LLMSettings(BaseModel):
    """LLM provider configuration."""

    default_provider: Literal["groq", "gemini"] = "groq"
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)


class RouterSettings(BaseModel):
    """Query router configuration."""

    complexity_threshold: float = 0.8
    fast_keywords: list[str] = Field(
        default_factory=lambda: ["hi", "hello", "thanks", "yes", "no", "ok"]
    )


class VoiceSettings(BaseModel):
    """Voice interaction configuration."""

    enabled: bool = False
    push_to_talk_key: str = "ctrl+shift+j"
    auto_detect: bool = False
    tts_voice: str = "en-GB-RyanNeural"
    # A brisk, low-pitched British delivery suits short live responses better
    # than Edge's default pace.
    tts_rate: str = "+20%"
    tts_pitch: str = "-5Hz"
    stt_model: str = "whisper-large-v3-turbo"
    stt_language: str | None = "en"
    silence_duration: float = 0.55


class MemorySettings(BaseModel):
    """Long-term memory configuration."""

    enabled: bool = False
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["password", "secret", "token", "*.env"]
    )


class LoggingSettings(BaseModel):
    """Application logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs" / "jarvis.log")
    max_size_mb: int = 2
    backup_count: int = 1


class SystemSettings(BaseModel):
    """Core system configuration."""

    port: int = 8741
    weather_city: str = ""


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",  # This gets overridden in get_settings dynamically
        env_file_encoding="utf-8",
    )

    # API Keys
    gemini_api_keys: str | list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "gemini_api_keys",
            "gemini_api_key",
            "GEMINI_API_KEYS",
            "GEMINI_API_KEY",
            "JARVIS_GEMINI_API_KEYS",
            "JARVIS_GEMINI_API_KEY",
        ),
    )
    groq_api_keys: str | list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "groq_api_keys",
            "groq_api_key",
            "GROQ_API_KEYS",
            "GROQ_API_KEY",
            "JARVIS_GROQ_API_KEYS",
            "JARVIS_GROQ_API_KEY",
        ),
    )

    @property
    def primary_gemini_key(self) -> str:
        """Helper to get the primary key for simple backwards compatibility."""
        if isinstance(self.gemini_api_keys, list) and self.gemini_api_keys:
            return self.gemini_api_keys[0]
        if isinstance(self.gemini_api_keys, str) and self.gemini_api_keys:
            return self.gemini_api_keys.split(",")[0].strip()
        return ""

    @property
    def gemini_api_key(self) -> str:
        """Return the primary Gemini key for legacy callers."""
        return self.primary_gemini_key

    @property
    def primary_groq_key(self) -> str:
        """Helper to get the primary key for simple backwards compatibility."""
        if isinstance(self.groq_api_keys, list) and self.groq_api_keys:
            return self.groq_api_keys[0]
        if isinstance(self.groq_api_keys, str) and self.groq_api_keys:
            return self.groq_api_keys.split(",")[0].strip()
        return ""

    @property
    def groq_api_key(self) -> str:
        """Return the primary Groq key for legacy callers."""
        return self.primary_groq_key

    # Nested config models
    llm: LLMSettings = Field(default_factory=LLMSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)

    @field_validator("gemini_api_keys")
    @classmethod
    def check_gemini_keys(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            if v == "fallback":
                return []
            if not v.strip():
                raise ValueError("GEMINI_API_KEYS must not be empty.")
            keys = [k.strip() for k in v.split(",") if k.strip()]
        else:
            keys = [k.strip() for k in v if k.strip()]

        if not keys:
            raise ValueError("At least one GEMINI_API_KEYS must be provided.")
        if len(keys) > 5:
            raise ValueError("Maximum of 5 Gemini API keys allowed.")
        return keys

    @field_validator("groq_api_keys")
    @classmethod
    def check_groq_keys(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            if v == "fallback":
                return []
            if not v.strip():
                raise ValueError("GROQ_API_KEYS must not be empty.")
            keys = [k.strip() for k in v.split(",") if k.strip()]
        else:
            keys = [k.strip() for k in v if k.strip()]

        if not keys:
            raise ValueError("At least one GROQ_API_KEYS must be provided.")
        if len(keys) > 5:
            raise ValueError("Maximum of 5 Groq API keys allowed.")
        return keys

    @classmethod
    def from_yaml(cls, path: Path, *, env_file: Path | None = None) -> "Settings":
        """Load configuration from a YAML file, overlaid with environment variables."""
        yaml_data = _load_yaml(path)
        return cls.from_mapping(yaml_data, env_file=env_file)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, env_file: Path | None = None) -> "Settings":
        """Validate mapping defaults while letting environment values win."""
        values = dict(data)
        values.setdefault("gemini_api_keys", "fallback")
        values.setdefault("groq_api_keys", "fallback")
        overlay = _EnvironmentFirstSettings(
            **values,
            _env_file=env_file if env_file is not None else _find_env_file(),
        )
        return cls.model_validate(overlay.model_dump())

    def save_to_yaml(self, path: Path | None = None) -> None:
        """Save current settings to a YAML file."""
        if path is None:
            path = PROJECT_ROOT / "config" / "jarvis.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)

        # API credentials deliberately remain in environment variables. Saving
        # UI preferences must never copy secrets into a YAML file.
        data = self.model_dump(mode="json", exclude={"gemini_api_keys", "groq_api_keys"})

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


class _EnvironmentFirstSettings(Settings):
    """Settings variant used only when YAML supplies lower-priority defaults."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return env_settings, dotenv_settings, init_settings, file_secret_settings


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return loaded


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_env_file() -> Path | None:
    candidates = [
        APP_DIR / ".env",
        APP_DIR.parent / ".env",
        APP_DIR.parent.parent / ".env",
        PROJECT_ROOT / ".env",
    ]
    return next((path for path in candidates if path.exists()), None)


@lru_cache
def get_settings() -> Settings:
    """Get the cached settings singleton.

    Loads configuration in order:
    1. Internal defaults.yaml
    2. User jarvis.yaml
    3. Environment variables (overrides)
    """
    defaults_path = (
        BUNDLE_DIR / "jarvis" / "config" / "defaults.yaml"
        if getattr(sys, "frozen", False)
        else BUNDLE_DIR / "src" / "jarvis" / "config" / "defaults.yaml"
    )
    user_config_path = PROJECT_ROOT / "config" / "jarvis.yaml"

    config_data = _load_yaml(defaults_path)
    if user_config_path.exists():
        config_data = _merge_dicts(config_data, _load_yaml(user_config_path))
    env_file = _find_env_file()

    return Settings.from_mapping(config_data, env_file=env_file)
