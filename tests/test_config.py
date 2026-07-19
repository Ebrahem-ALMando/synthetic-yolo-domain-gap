from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from synthdet.config import load_config, resolve_config_paths, validate_project_directories
from synthdet.config.loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def test_default_config_loads_with_protocol_defaults() -> None:
    config = load_config()

    assert config.seed == 42
    assert config.yolo.base_model == "yolo11n.pt"
    assert config.dataset.identifier == "aquarium-combined-v2-raw-1024"
    assert config.dataset.class_names == [
        "fish",
        "jellyfish",
        "penguin",
        "puffin",
        "shark",
        "starfish",
        "stingray",
    ]
    assert config.dataset.duplicate_detection is not None
    assert config.dataset.duplicate_detection.hamming_threshold == 6
    assert config.project.sprint == 3
    assert config.synthetic.name == "aquarium-synthetic-v1"
    assert config.synthetic.pool_size == 427
    assert (
        config.synthetic.pool_identity
        == "3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec"
    )


def test_relative_paths_resolve_from_project_root() -> None:
    config = load_config()
    paths = resolve_config_paths(config)

    assert paths["datasets_root"] == PROJECT_ROOT / "datasets"
    assert paths["experiment_runs"] == PROJECT_ROOT / "artifacts" / "experiments"
    assert all(path.is_absolute() for path in paths.values())


def test_missing_config_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_threshold_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["yolo"]["confidence_threshold"] = 1.5
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(invalid_path)


def test_essential_project_directories_exist() -> None:
    assert validate_project_directories() == []


def test_missing_essential_directories_are_reported(tmp_path: Path) -> None:
    missing = validate_project_directories(tmp_path)

    assert tmp_path / "apps" / "api" in missing
    assert tmp_path / "tests" in missing
