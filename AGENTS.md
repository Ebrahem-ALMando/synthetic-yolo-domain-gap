# Instructions for coding agents

These rules apply to the entire repository.

1. Inspect the workspace, `PROJECT_STATE.md`, relevant documentation, and Git status before making
   changes. Preserve user files and unrelated work.
2. Never fabricate metrics, charts, result files, model weights, completed runs, or claims that
   training or evaluation occurred.
3. Never place real test images in training or validation data. Never use test images as copy-paste
   object sources, synthetic backgrounds, augmentation inputs, or synthetic-generation references.
4. Keep the fixed real test split unchanged across all experiments. Treat split manifests as
   protocol-controlled artifacts and check for leakage before every run.
5. Use deterministic seed 42 by default. Record any justified deviation in the experiment config
   and decision log; seed Python, NumPy, augmentation, and training frameworks where applicable.
6. Keep datasets, generated artifacts, YOLO runs, caches, secrets, and model weights out of Git.
   Do not weaken `.gitignore` to commit them.
7. Make every future result traceable to a versioned experiment configuration and a unique output
   directory. Do not enter evaluation values manually.
8. Use portable paths and commands so the project remains usable on Windows and Linux.
9. Avoid speculative abstractions and placeholder business logic. Implement only the active sprint.
10. Run available tests, environment checks, and static validation before claiming completion.
    Report unavailable dependencies or tools honestly.
11. Update `PROJECT_STATE.md` at the end of every sprint with facts only.

