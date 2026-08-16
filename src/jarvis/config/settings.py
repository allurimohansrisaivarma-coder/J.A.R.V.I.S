"""Application settings and configuration management."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import sys

if getattr(sys, 'frozen', False):
    BUNDLE_DIR = Path(sys._MEIPASS)
    APP_DIR = Path(sys.executable).parent
    PROJECT_ROOT = APP_DIR
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    APP_DIR = BUNDLE_DIR
    PROJECT_ROOT = BUNDLE_DIR


class GeminiSettings(BaseModel):
    """Settings for Google Gemini models."""
    model_flash: str = "gemini-3.5-flash"
    model_pro: str = "gemini-3.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048


class GroqSettings(BaseModel):
    """Settings for Groq inference."""
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.3
    max_output_tokens: int = 1024


class LLMSettings(BaseModel):
    """LLM provider configuration."""
    default_provider: str = "gemini"
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


class Settings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_prefix="JARVIS_", 
        env_nested_delimiter="__", 
        extra="ignore",
        env_file=".env",  # This gets overridden in get_settings dynamically
        env_file_encoding="utf-8"
    )

    # API Keys
    gemini_api_keys: str | list[str] = Field(default_factory=list, validation_alias="GEMINI_API_KEYS")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    @property
    def primary_gemini_key(self) -> str:
        """Helper to get the primary key for simple backwards compatibility."""
        if isinstance(self.gemini_api_keys, list) and self.gemini_api_keys:
            return self.gemini_api_keys[0]
        if isinstance(self.gemini_api_keys, str) and self.gemini_api_keys:
            return self.gemini_api_keys.split(",")[0].strip()
        return ""

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
            if v == "fallback" or not v.strip():
                return []
            keys = [k.strip() for k in v.split(",") if k.strip()]
        else:
            keys = [k.strip() for k in v if k.strip()]
            
        if not keys:
            raise ValueError("At least one GEMINI_API_KEYS must be provided.")
        if len(keys) > 5:
            raise ValueError("Maximum of 5 Gemini API keys allowed.")
        return keys

    @field_validator("groq_api_key")
    @classmethod
    def check_groq_key(cls, v: str) -> str:
        """Validate that API keys are set and not empty."""
        if not v or not str(v).strip():
            raise ValueError("API key for groq_api_key must be provided and not empty.")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load configuration from a YAML file, overlaid with environment variables."""
        yaml_data = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        try:
            return cls(**yaml_data)
        except Exception:
            yaml_data["gemini_api_keys"] = "fallback"
            yaml_data["groq_api_key"] = "fallback"
            return cls(**yaml_data)

    def save_to_yaml(self, path: Path = None) -> None:
        """Save current settings to a YAML file."""
        if path is None:
            path = PROJECT_ROOT / "config" / "jarvis.yaml"
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # model_dump() excludes validation_alias fields by default, but we need them in yaml if we don't alias them properly on export.
        # It's safer to just dump the core dict
        data = self.model_dump(mode="json", exclude={"gemini_api_keys", "groq_api_key"})
        
        # Add back keys if they are not fallback
        if self.gemini_api_keys and self.gemini_api_keys != "fallback":
            data["GEMINI_API_KEYS"] = self.gemini_api_keys
        if self.groq_api_key and self.groq_api_key != "fallback":
            data["GROQ_API_KEY"] = self.groq_api_key
            
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


@lru_cache()
def get_settings() -> Settings:
    """Get the cached settings singleton.
    
    Loads configuration in order:
    1. Internal defaults.yaml
    2. User jarvis.yaml
    3. Environment variables (overrides)
    """
    defaults_path = BUNDLE_DIR / "src" / "jarvis" / "config" / "defaults.yaml"
    user_config_path = PROJECT_ROOT / "config" / "jarvis.yaml"

    if user_config_path.exists():
        return Settings.from_yaml(user_config_path)
    
    if defaults_path.exists():
        return Settings.from_yaml(defaults_path)

    env_paths = [
        APP_DIR / ".env",
        APP_DIR.parent / ".env",
        APP_DIR.parent.parent / ".env",
        PROJECT_ROOT / ".env"
    ]
    env_file = next((p for p in env_paths if p.exists()), None)

    # Allow instantiation without throwing validation errors if env vars aren't set during module load
    try:
        return Settings(_env_file=env_file)
    except Exception:
        return Settings(gemini_api_keys="fallback", groq_api_key="fallback", _env_file=env_file)
