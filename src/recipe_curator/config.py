"""Configuration management for recipe_curator."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

ScalarValue = str | int | bool
ConfigData = dict[str, object]

_pydantic = importlib.import_module("pydantic")
BaseModel = _pydantic.BaseModel
Field = _pydantic.Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CONFIG_PATH = PROJECT_ROOT / "project_config.yml"
VALID_ENVS = {"dev", "acc", "prd"}


class ProjectConfig(BaseModel):
    """Environment-specific project configuration."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="LLM endpoint name")
    embedding_endpoint: str = Field(..., description="Embedding endpoint name")
    warehouse_id: str = Field(..., description="Warehouse ID")
    vector_search_endpoint: str = Field(
        ..., description="Vector search endpoint name"
    )
    system_prompt: str = Field(
        default=(
            "You are a sports nutrition recipe assistant that helps users discover "
            "and adapt recipes to their available ingredients and fitness goals."
        ),
        description="System prompt for the recipe assistant",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str | Path, env: str = "dev") -> ProjectConfig:
        """Load one environment section from a YAML config file."""
        env_name = _validate_env(env)
        config_data = _load_yaml_data(Path(config_path))

        if env_name not in config_data:
            raise ValueError(f"Environment '{env_name}' not found in config file")

        env_section = config_data[env_name]
        if not isinstance(env_section, dict):
            raise ValueError(f"Environment '{env_name}' config must be a mapping")

        return cls(**env_section)

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.db_schema}.{self.volume}"

    @property
    def recipes_raw_table(self) -> str:
        """Default fully qualified source table."""
        return f"{self.catalog}.{self.db_schema}.recipes_raw"

    @property
    def recipes_curated_table(self) -> str:
        """Default fully qualified curated table."""
        return f"{self.catalog}.{self.db_schema}.recipes_curated"


class ModelConfig(BaseModel):
    """Model generation defaults."""

    temperature: float = Field(0.2, description="Model temperature")
    max_tokens: int = Field(1200, description="Maximum tokens")
    top_p: float = Field(0.95, description="Top-p sampling parameter")


class VectorSearchConfig(BaseModel):
    """Vector search defaults."""

    embedding_dimension: int = Field(1024, description="Embedding dimension")
    similarity_metric: str = Field("cosine", description="Similarity metric")
    num_results: int = Field(5, description="Number of results to return")


class ChunkingConfig(BaseModel):
    """Chunking defaults for long recipe/instruction text."""

    chunk_size: int = Field(512, description="Chunk size in tokens")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    separator: str = Field("\n\n", description="Separator for chunking")


def _validate_env(env: str) -> str:
    """Validate environment name."""
    env_name = (env or "dev").strip().lower()
    if env_name not in VALID_ENVS:
        raise ValueError(
            f"Invalid environment: {env}. Expected one of {sorted(VALID_ENVS)}"
        )
    return env_name


def _parse_scalar(raw_value: str) -> ScalarValue:
    """Parse a YAML scalar value."""
    value = raw_value.strip().strip('"').strip("'")
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.isdigit():
        return int(value)
    return value


def _parse_simple_yaml(text: str) -> ConfigData:
    """Fallback parser for the simple env:key:value structure used in this repo."""
    config: ConfigData = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" "):
            if line.endswith(":"):
                current_section = line[:-1].strip()
                config[current_section] = {}
            elif ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = _parse_scalar(value)
            continue

        if current_section and ":" in line:
            key, value = line.strip().split(":", 1)
            section_value = config.get(current_section)
            if isinstance(section_value, dict):
                section_value[key.strip()] = _parse_scalar(value)

    return config


def _load_yaml_data(config_path: Path) -> ConfigData:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = config_path.read_text(encoding="utf-8")
    try:
        yaml_module = importlib.import_module("yaml")

        parsed = yaml_module.safe_load(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level YAML config must be a mapping")
        return parsed
    except ModuleNotFoundError:
        return _parse_simple_yaml(raw)


def _resolve_config_path(config_path: str | Path) -> Path:
    """Resolve configuration file path."""
    path = Path(config_path)
    if path.is_absolute() and path.exists():
        return path

    if path.exists():
        return path.resolve()

    for current in [Path.cwd(), *Path.cwd().parents[:3]]:
        candidate = current / path
        if candidate.exists():
            return candidate.resolve()

    fallback = PROJECT_ROOT / path.name
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(f"Could not locate config file: {config_path}")


def load_config(
    config_path: str | Path = "project_config.yml", env: str | None = None
) -> ProjectConfig:
    """Load project config for the requested environment."""
    env_name = _validate_env(env or get_env())
    resolved_path = _resolve_config_path(config_path)
    return ProjectConfig.from_yaml(resolved_path, env_name)


def get_env(spark: object | None = None) -> str:
    """Get current environment from Databricks widget or environment variables."""
    if spark is not None:
        try:
            dbutils_module = importlib.import_module("pyspark.dbutils")
            widget_value = dbutils_module.DBUtils(spark).widgets.get("env")
            if widget_value:
                return widget_value
        except Exception:  # noqa: BLE001
            pass

    return os.getenv("BUNDLE_TARGET") or os.getenv("ENV") or "dev"
