"""Load and validate the central YAML project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"


class StrictModel(BaseModel):
    """Base model that rejects misspelled or undocumented settings."""

    model_config = ConfigDict(extra="forbid")


class ProjectMetadata(StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    sprint: int = Field(ge=1)


class DatasetConfig(StrictModel):
    name: str | None
    identifier: str | None = None
    candidate: str | None = None
    status: str | None = None
    version: int | None = Field(default=None, ge=1)
    export_name: str | None = None
    annotation_format: str | None = None
    license: str | None = None
    class_names: list[str] | None
    split_seed: int | None = Field(default=None, ge=0)
    duplicate_detection: DuplicateDetectionConfig | None = None
    object_size_rule: ObjectSizeRuleConfig | None = None
    paths: DatasetPathsConfig | None = None

    @field_validator("class_names")
    @classmethod
    def validate_class_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [name.strip() for name in value]
        if not cleaned or any(not name for name in cleaned):
            raise ValueError("class_names must be null or a non-empty list of non-empty names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("class_names must be unique")
        return cleaned


class DuplicateDetectionConfig(StrictModel):
    method: str = Field(min_length=1)
    hamming_threshold: int = Field(ge=0, le=64)


class ObjectSizeRuleConfig(StrictModel):
    small_max_area_px: int = Field(gt=0)
    medium_max_area_px: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_ordered_bounds(self) -> ObjectSizeRuleConfig:
        if self.medium_max_area_px <= self.small_max_area_px:
            raise ValueError("medium object-size bound must exceed the small bound")
        return self


class DatasetPathsConfig(StrictModel):
    raw: Path
    audit: Path
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    excluded_manifest: Path
    duplicate_groups: Path
    split_metadata: Path


class PathsConfig(StrictModel):
    datasets_root: Path
    real_data: Path
    synthetic_data: Path
    splits: Path
    artifacts_root: Path
    experiment_runs: Path
    models: Path
    reports: Path


class YoloConfig(StrictModel):
    base_model: str = Field(min_length=1)
    image_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)


class ProjectConfig(StrictModel):
    project: ProjectMetadata
    seed: int = Field(ge=0)
    dataset: DatasetConfig
    paths: PathsConfig
    yolo: YoloConfig


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load YAML from *config_path* and return a validated configuration."""

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        raw: Any = yaml.safe_load(config_file)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return ProjectConfig.model_validate(raw)


def resolve_config_paths(config: ProjectConfig, root: str | Path = PROJECT_ROOT) -> dict[str, Path]:
    """Resolve every configured project path against *root* without creating it."""

    root_path = Path(root).expanduser().resolve()
    return {
        name: value.resolve() if value.is_absolute() else (root_path / value).resolve()
        for name, value in config.paths.model_dump().items()
    }


def validate_project_directories(root: str | Path = PROJECT_ROOT) -> list[Path]:
    """Return required repository directories that are absent from *root*."""

    root_path = Path(root).expanduser().resolve()
    required = (
        "apps/api",
        "apps/web",
        "src/synthdet/config",
        "src/synthdet/data",
        "src/synthdet/synthetic",
        "src/synthdet/training",
        "src/synthdet/evaluation",
        "src/synthdet/inference",
        "scripts",
        "configs",
        "datasets",
        "artifacts",
        "models",
        "reports/figures",
        "reports/tables",
        "notebooks",
        "docs",
        "tests",
    )
    return [root_path / relative for relative in required if not (root_path / relative).is_dir()]
