from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    openai_model_top: str = Field(default="gpt-4o", alias="OPENAI_MODEL_TOP")
    openai_image_model: str = Field(default="gpt-image-1", alias="OPENAI_IMAGE_MODEL")
    anthropic_model_sonnet: str = Field(
        default="claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL_SONNET"
    )
    anthropic_model_opus: str = Field(
        default="claude-opus-4-7", alias="ANTHROPIC_MODEL_OPUS"
    )

    dry_run: bool = Field(default=False, alias="DRY_RUN")
    max_spend_cents_per_day: int = Field(default=1500, alias="MAX_SPEND_CENTS_PER_DAY")
    embedding_batch_size: int = Field(default=96, alias="EMBEDDING_BATCH_SIZE")

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @computed_field
    @property
    def db_path(self) -> Path:
        return self.data_dir / "pipeline.db"

    @computed_field
    @property
    def output_dir(self) -> Path:
        """Top-level human-readable output folder. Per-event briefs land here."""
        return self.project_root / "output"

    def validate_keys(self) -> None:
        if self.dry_run:
            return
        if not self.anthropic_api_key.strip():
            raise RuntimeError("ANTHROPIC_API_KEY missing (or set DRY_RUN=true)")
        if not self.openai_api_key.strip():
            raise RuntimeError("OPENAI_API_KEY missing (or set DRY_RUN=true)")


@lru_cache
def get_settings() -> Settings:
    return Settings()
