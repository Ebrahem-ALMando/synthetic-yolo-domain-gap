"""Structural contract for the restart-safe Sprint 4B Colab notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SECTIONS = (
    "1. Runtime and GPU inspection",
    "2. Google Drive mounting",
    "3. Training-bundle upload or Drive path selection",
    "4. Bundle SHA-256 validation",
    "5. Safe extraction",
    "6. Pinned dependency installation",
    "7. CUDA and PyTorch verification",
    "8. Repository revision and frozen-identity validation",
    "9. Experiment-view validation",
    "10. GPU batch-profile preflight",
    "11. Final profile freezing",
    "12. Sequential five-regime training",
    "13. Per-regime output validation",
    "14. Persistent Drive output copy",
    "15. Final training-completion audit",
    "16. Results archive creation",
)
CONFIGURATION_VARIABLES = (
    "BUNDLE_PATH",
    "PERSISTENT_OUTPUT_DIRECTORY",
    "REGIME_SELECTION",
    "DEVICE",
)
BANNED_CODE = (
    "real_test",
    "test/images",
    "mode='test'",
    'mode="test"',
    ".predict(",
    "yolo predict",
)


def validate_notebook(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Notebook must use nbformat 4")
    cells = notebook.get("cells", [])
    markdown = [
        "".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "markdown"
    ]
    positions = []
    for section in SECTIONS:
        matches = [index for index, value in enumerate(markdown) if section in value]
        if len(matches) != 1:
            raise ValueError(f"Notebook must contain exactly one section heading: {section}")
        positions.append(matches[0])
    if positions != sorted(positions):
        raise ValueError("Notebook sections are not in the required order")
    code_cells = [
        "".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code"
    ]
    code = "\n".join(code_cells)
    missing = [name for name in CONFIGURATION_VARIABLES if name not in code_cells[0]]
    if missing:
        raise ValueError("First configuration cell is missing: " + ", ".join(missing))
    if "EXPECTED_REPOSITORY_REVISION" in code_cells[0]:
        raise ValueError("User configuration must not require an expected repository revision")
    if re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", code, flags=re.IGNORECASE):
        raise ValueError("Notebook contains a forbidden literal Git commit SHA")
    required_revision_flow = (
        "training_bundle_inventory.json",
        "expected_repository_revision",
        "RESOLVED_REPOSITORY_REVISION",
        "--expected-revision",
    )
    missing_revision_flow = [token for token in required_revision_flow if token not in code]
    if missing_revision_flow:
        raise ValueError(
            "Notebook does not resolve revision from bundle inventory: "
            + ", ".join(missing_revision_flow)
        )
    lowered = code.lower()
    violations = [token for token in BANNED_CODE if token in lowered]
    if violations:
        raise ValueError("Forbidden test/inference command in notebook: " + ", ".join(violations))
    if "torch.cuda.is_available()" not in code or "@" not in code:
        raise ValueError("Notebook lacks explicit CUDA availability and tensor-operation checks")
    if "gpu_preflight.py" not in code or "colab_train.py" not in code:
        raise ValueError("Notebook lacks preflight or sequential-training execution")
    return {
        "status": "passed",
        "section_count": len(SECTIONS),
        "code_cell_count": len(code_cells),
        "no_test_inference_commands": True,
    }
