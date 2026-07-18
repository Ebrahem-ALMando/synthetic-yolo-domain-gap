# Protocol-controlled manifests

Frozen, public-data manifests belong under a dataset-specific directory after successful
acquisition, validation, duplicate review, and source grouping.
# Frozen manifest versions

`aquarium/v1/` is the byte-for-byte preserved Sprint 2 split. The original files directly under
`aquarium/` remain an unchanged legacy mirror so their published paths and hashes stay traceable.

`aquarium/v2/` is the Sprint 2.5 split created only after the focused Penguin capture-dependency
review. Each version is immutable: regeneration must use the non-destructive verification mode,
not overwrite the frozen files.
