# Agent Rules

These rules are the source of truth for Codex, Claude Code, and other agents
working in this repository. Keep this file short and operational. Detailed
background docs may exist, but agents should not assume they have read `docs/`
unless the user explicitly asks.

## First Principles

- Inspect the existing tree before creating files.
- Prefer the closest existing directory over a new top-level directory.
- Do not create new top-level directories unless this file lists them or the
  user explicitly approves.
- Keep source code, reusable scripts, human configs, runtime artifacts, and
  final reusable artifacts separate.
- Do not commit local machine paths, large generated artifacts, logs, caches, or
  checkpoints unless the user explicitly asks.

## Repository Layout

- `model/`: model source code and reusable model modules.
- `dataloader/`: dataset adapters, batch-format adapters, and loading code.
- `conf/`: human-maintained YAML configs.
- `conf/generated/`: generated configs that can be recreated.
- `scripts/`: reusable command-line, evaluation, export, and Slurm helpers.
- `tests/`: tests only. Do not add or stage tests unless the user asks.
- `pretrained/`: external frozen pretrained assets, such as X-Codec,
  Spark-TTS, local WavLM, or other downloaded model dependencies.
- `runs/<exp>/`: one experiment bundle.
- `runs/<exp>/ckpts/`: training checkpoints for that experiment.
- `runs/<exp>/logs/`: TensorBoard, Slurm, W&B local files, and live logs for
  that experiment.
- `artifacts/`: curated reusable final artifacts, such as frozen discriminator,
  generator, fusion, or exported model checkpoints.
- `outputs/`: generated inference outputs and evaluation outputs.
- `tmp/`: scratch files that are safe to delete.
- `docs/`: optional human-readable notes. Do not rely on these as active agent
  instructions unless the user points to them.

Legacy runtime directories such as `logs/` and `checkpoints/` may exist. New
experiments should prefer `runs/<exp>/logs/` and `runs/<exp>/ckpts/` unless an
existing script/config still expects the legacy path.

## Experiment Conventions

- Scripts choose the process; configs choose the data.
- Keep reusable task names generic, such as `train`, `infer`, `infer_all`,
  `smoke_sr`, or `smoke_hybrid`.
- Choose dataset and paths through config fields and environment overrides, not
  through dataset-specific wrapper scripts.
- Human-edited experiment configs belong in `conf/`.
- Generated configs belong in `conf/generated/` or under a specific
  `runs/<exp>/` bundle.
- Configs must make data format explicit with fields such as `dataset_type`.
- W&B run names should be distinguishable across repositories:
  `<repo_name>__<experiment>__<dataset_type>__<timestamp>`.
- W&B config metadata should include available fields such as `repo_name`,
  `experiment`, `config_path`, `log_dir`, `checkpoint_dir`,
  `train_dataset_type`, `val_dataset_type`, `test_dataset_type`, `max_epochs`,
  `val_check_interval`, and `devices`.

## USE Simulation And Data Loading

- Support explicit dataset modes rather than path conventions:
  `use_simulation_onthefly`, `use_simulation_fixed`,
  `use_simulation_rolling_cache`, `gap_webdataset`,
  `hybrid_unise_webdataset_stream`, and
  `hybrid_unise_webdataset_fixed_recipe` when relevant.
- Do not hard-code local USE Simulation paths in reusable scripts. Put them in
  config fields such as `use_simulation_root`, `pair_manifest`, `clean_json`,
  `noise_json`, `rir_json`, `simulation_config`, `active_root`, or
  `split_root`.
- Keep model-repository loaders as thin adapters from shared data/sample formats
  to this repository's batch format.
- Shared degradation logic should live in the USE Simulation repository or other
  shared data repository, not copied deeply into this model repository.
- Evaluation outputs intended for AVQI/ABQI/ABQY-style tooling should include
  enhanced wavs plus `inf.scp` and `ref.scp` when references are available.

## Script Rules

- Keep reusable scripts in `scripts/`.
- Avoid one-off shell scripts at the repository root.
- Prefer one configurable entry point over many local dataset wrappers.
- Slurm helpers should use generic tasks and environment overrides such as
  `CONFIG_PATH`, `CKPT_PATH`, `OUTPUT_DIR`, `RUN_LOG_DIR`, and `RUN_ID`.
- Local smoke/debug scripts may live in `tmp/` if they are not reusable.

## Python Rules

- Put imports at the top of Python files.
- Avoid conditional, branch-based, fallback, or bottom-of-file imports.
- Function-local imports are allowed only for expensive optional dependencies,
  real circular imports, or keeping optional dependencies out of common paths.
- Avoid broad `try/except`. Catch specific exceptions, keep protected scopes
  small, and log or re-raise with context.
- Do not silently swallow errors.

## Git Hygiene

- Before committing, check `git status --short` and staged file lists.
- Do not stage ignored runtime artifacts, checkpoints, pretrained assets,
  generated outputs, or tests unless the user explicitly asks.
- Preserve user changes. Do not revert unrelated files.
- If updating from remote, prefer a fast-forward or rebase workflow that avoids
  unnecessary merge commits.
