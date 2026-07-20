from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from synthdet.training.intake import ARCHIVE_NAME, validate_return_archive


def _write_return_fixture(root: Path, member_name: str = "runs/real_only/results.csv") -> None:
    payload = b"epoch,metrics/mAP50-95(B)\n1,0.2\n"
    entry = {
        "path": member_name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    completion = b'{"status":"completed"}\n'
    completion_entry = {
        "path": "completion/training_completion_manifest.json",
        "size_bytes": len(completion),
        "sha256": hashlib.sha256(completion).hexdigest(),
    }
    inventory = {
        "training_identity": "a" * 64,
        "test_set_access_count": 0,
        "contains_dataset_images": False,
        "contains_test_outputs": False,
        "contains_secrets": False,
        "inventory": [completion_entry, entry],
    }
    archive = root / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(completion_entry["path"], completion)
        handle.writestr(member_name, payload)
        handle.writestr(
            "results_archive_inventory.json",
            json.dumps(inventory, sort_keys=True),
        )
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / f"{ARCHIVE_NAME}.sha256").write_text(
        f"{archive_hash}  {ARCHIVE_NAME}\n", encoding="utf-8"
    )
    (root / f"{ARCHIVE_NAME}.inventory.json").write_text(
        json.dumps({**inventory, "archive_sha256": archive_hash}), encoding="utf-8"
    )
    (root / "training_completion_manifest.json").write_bytes(completion)
    (root / "final_profile.json").write_text("{}\n", encoding="utf-8")


def test_return_archive_validates_every_member(tmp_path: Path) -> None:
    _write_return_fixture(tmp_path)
    result = validate_return_archive(tmp_path)
    assert result["archive_name"] == ARCHIVE_NAME
    assert result["inventory_file_count"] == 2


@pytest.mark.parametrize(
    "member_name",
    ["../escape.csv", "datasets/raw/test.jpg", "runs/real_only/credentials.json"],
)
def test_return_archive_rejects_unsafe_or_forbidden_members(
    tmp_path: Path, member_name: str
) -> None:
    _write_return_fixture(tmp_path, member_name)
    with pytest.raises(ValueError):
        validate_return_archive(tmp_path)


def test_return_archive_rejects_member_hash_mismatch(tmp_path: Path) -> None:
    _write_return_fixture(tmp_path)
    inventory_path = tmp_path / f"{ARCHIVE_NAME}.inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["inventory"][1]["sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_return_archive(tmp_path)
