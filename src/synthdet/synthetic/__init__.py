"""Train-only, provenance-aware copy-paste synthetic generation."""

from synthdet.synthetic.contracts import (
    SyntheticConfig,
    derive_seed,
    load_synthetic_config,
    verify_active_split,
)

__all__ = [
    "SyntheticConfig",
    "derive_seed",
    "load_synthetic_config",
    "verify_active_split",
]
