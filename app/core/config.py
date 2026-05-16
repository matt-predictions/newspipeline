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

    max_spend_cents_per_day: int = Field(default=1500, alias="MAX_SPEND_CENTS_PER_DAY")
    embedding_batch_size: int = Field(default=96, alias="EMBEDDING_BATCH_SIZE")

    enable_sora_render: bool = Field(default=False, alias="ENABLE_SORA_RENDER")
    sora_model_id: str = Field(default="sora-2", alias="SORA_MODEL_ID")
    sora_max_wait_s: int = Field(default=600, alias="SORA_MAX_WAIT_S")
    sora_poll_interval_s: float = Field(default=10.0, alias="SORA_POLL_INTERVAL_S")

    max_parallel_stories: int = Field(default=3, alias="MAX_PARALLEL_STORIES")

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

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    def validate_keys(self) -> None:
        """Require at least one provider key.

        Both keys → full pipeline (OpenAI: brief+image+sora, Anthropic: panel+proposer).
        Only OPENAI_API_KEY → conversation + proposer get routed through OpenAI.
        Only ANTHROPIC_API_KEY → brief gets routed through Anthropic; hero image
                                  and Sora skip silently (Anthropic does no image gen).
        Neither → hard fail with a clear message.
        """
        if not (self.has_openai or self.has_anthropic):
            raise RuntimeError(
                "No API keys set — add at least ONE of ANTHROPIC_API_KEY or "
                "OPENAI_API_KEY to .env. Both is best; one works too."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
