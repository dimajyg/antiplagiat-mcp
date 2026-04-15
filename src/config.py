"""Server configuration and per-request credentials.

The server itself only holds defaults loaded from `.env` (used for local dev
or as fallback). Real production credentials arrive **per request** in HTTP
headers and are wrapped in `RequestCredentials` — never persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Process-wide config. Loaded once from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8765
    log_level: str = "INFO"

    models_dir: Path = Path("./models")
    cache_db: Path = Path("./cache.sqlite")

    openrouter_api_key: str = ""
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    serper_api_key: str = ""
    sapling_api_key: str = ""
    gptzero_api_key: str = ""


@dataclass(frozen=True, slots=True)
class RequestCredentials:
    """Per-request credentials extracted from MCP request headers.

    Falls back to server defaults from `ServerSettings` when a header is absent
    (useful for local testing). Never logged, never written to disk.
    """

    openrouter_key: str
    serper_key: str
    sapling_key: str
    gptzero_key: str

    @classmethod
    def from_headers(cls, headers: dict[str, str], defaults: ServerSettings) -> RequestCredentials:
        def pick(name: str, fallback: str) -> str:
            return headers.get(name) or headers.get(name.lower()) or fallback

        return cls(
            openrouter_key=pick("X-OpenRouter-Key", defaults.openrouter_api_key),
            serper_key=pick("X-Serper-Key", defaults.serper_api_key),
            sapling_key=pick("X-Sapling-Key", defaults.sapling_api_key),
            gptzero_key=pick("X-GPTZero-Key", defaults.gptzero_api_key),
        )


settings = ServerSettings()
