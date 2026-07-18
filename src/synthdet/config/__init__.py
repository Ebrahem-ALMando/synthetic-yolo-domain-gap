"""Typed project configuration."""

from synthdet.config.loader import (
    ProjectConfig,
    load_config,
    resolve_config_paths,
    validate_project_directories,
)

__all__ = [
    "ProjectConfig",
    "load_config",
    "resolve_config_paths",
    "validate_project_directories",
]

