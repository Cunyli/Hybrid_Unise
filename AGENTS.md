# Hybrid_Unise Agent Rules

This file is the active instruction entry point for agents working in this
repository. It is intentionally specific to this cloned-and-extended UniSE
project, not a universal template for every audio repository.

`CLAUDE.md` should only point here. Do not duplicate rules there. `docs/` is for
local handoff notes between agents, not the authoritative rule source.

## Project Reality

- This repository started from QuarkAudio-UniSE and now contains both the
  original UniSE/BiCodec path and a Hybrid-UniSE path.
- Existing upstream/demo folders such as `AudioSamples/`, `Figure/`,
  `checkpoints/`, and `logs/` may remain. Do not reorganize them casually.
- New work should follow the layout below, but compatibility with existing
  scripts/configs matters more than forcing a perfect migration in one patch.

## Placement Rules

- Inspect the tree before creating files.
- Prefer an existing directory over a new top-level directory.
- Do not create a new top-level directory unless it is listed here or the user
  explicitly approves.
- Keep source, configs, reusable scripts, runtime outputs, and curated artifacts
  separate.
- Do not commit local absolute paths, generated logs, caches, checkpoints,
  pretrained weights, or large outputs unless the user explicitly asks.

## Source And Config Layout

- `model/`: model source code. Original UniSE modules and Hybrid-UniSE modules
  both live here.
- `model/audio/`: reusable audio/STFT utilities.
- `model/bicodec/`, `model/llm/`: original UniSE/BiCodec components.
- `dataloader/`: dataset adapters and batch-format adapters.
- `dataloader/simulation/`: local simulation helpers used by existing loaders.
- `conf/`: human-maintained YAML configs.
- `conf/generated/`: generated configs that can be recreated.
- `scripts/`: reusable command-line, evaluation, export, and Slurm helpers.
- `tests/`: tests only. Do not add or stage tests unless the user asks.

## Artifact Layout

- `pretrained/`: external frozen pretrained assets, such as X-Codec, Spark-TTS,
  WavLM, or downloaded Hugging Face checkpoints.
- `runs/<exp>/`: preferred bundle for a new experiment.
- `runs/<exp>/ckpts/`: training checkpoints for that experiment.
- `runs/<exp>/logs/`: TensorBoard, Slurm, W&B local files, and live logs.
- `runs/<exp>/outputs/`: experiment-specific generated outputs.
- `artifacts/`: curated reusable final artifacts, such as frozen discriminator,
  generator, fusion, or exported model checkpoints.
- `outputs/`: ad hoc inference/evaluation outputs not yet tied to a run bundle.
- `tmp/`: scratch files that are safe to delete.
- `docs/`: ignored local handoff notes for agents.

Legacy runtime directories `checkpoints/` and `logs/` still exist and may be
referenced by existing configs. For new experiment wiring, prefer
`runs/<exp>/ckpts/` and `runs/<exp>/logs/`; update legacy paths only when doing
that migration deliberately.

## Important Entry Points

- Main Lightning training entry: `train.py`
- Generic Slurm wrapper: `scripts/slurm.sh`
- Data registry: `dataloader/data_module.py`
- Rolling-cache loader: `dataloader/rolling_cache.py`
- GAP WebDataset loader: `dataloader/gap_webdataset.py`
- Hybrid WebDataset protocol loader: `dataloader/hybrid_webdataset_protocol.py`
- Hybrid model: `model/hybrid_model.py`
- Hybrid inference API: `model/hybrid_inference.py`
- Hybrid helper scripts:
  - `scripts/train_hybrid.py`
  - `scripts/test_hybrid.py`
  - `scripts/infer_hybrid_directory.py`
  - `scripts/validate_hybrid_config.py`

## Experiment Rules

- Scripts choose the process; configs choose the data.
- Keep reusable task names generic, such as `train`, `infer`, `infer_all`,
  `smoke_sr`, `smoke_hybrid`, or `eval_token_validation`.
- Use config fields and environment overrides for datasets, checkpoints, output
  roots, and run IDs. Avoid dataset-specific wrapper scripts.
- Configs must make data format explicit with `dataset_type`.
- W&B run names should be distinguishable:
  `<repo_name>__<experiment>__<dataset_type>__<timestamp>`.
- W&B metadata should include available fields such as `repo_name`,
  `experiment`, `config_path`, `log_dir`, `checkpoint_dir`,
  `train_dataset_type`, `val_dataset_type`, `test_dataset_type`, `max_epochs`,
  `val_check_interval`, and `devices`.

## Data Loading Rules

- Supported dataset types include `use_simulation_onthefly`,
  `use_simulation_fixed`, `use_simulation_rolling_cache`, `gap_webdataset`,
  `hybrid_unise_webdataset_stream`, and
  `hybrid_unise_webdataset_fixed_recipe`.
- Do not hard-code local USE Simulation, GAP, or dry-data paths in reusable
  scripts. Put paths in config fields such as `use_simulation_root`,
  `pair_manifest`, `clean_json`, `noise_json`, `rir_json`,
  `simulation_config`, `active_root`, `split_root`, and `cache_dir`.
- Keep loaders as thin adapters from shared sample formats to this repository's
  batch format.
- Shared degradation logic should remain in shared data/simulation repositories
  where possible. Do not copy a large external data system into this repo.
- Hybrid SR/USE batches should use dict format with `degraded_wav`,
  `clean_wav`, `sample_rate`, `length`, and `utterance_id`.
- SFI STFT expects one sample rate per batch. Bucket or sample by rate before
  batching mixed-rate data.

## Output Rules

- AVQI/ABQI/ABQY-style evaluation outputs should include enhanced wavs plus
  `inf.scp` and `ref.scp` when references are available.
- Do not put source code under `logs/`, `outputs/`, `runs/`, `tmp/`,
  `pretrained/`, `checkpoints/`, or `artifacts/`.

## Python Rules

- Put imports at the top of Python files.
- Avoid conditional, branch-based, fallback, or bottom-of-file imports.
- Function-local imports are allowed only for expensive optional dependencies,
  real circular imports, or keeping optional dependencies out of common paths.
- Avoid broad `try/except`. Catch specific exceptions, keep protected scopes
  small, and log or re-raise with context.
- Do not silently swallow errors.

## Git Hygiene

- Before committing, inspect `git status --short` and the staged file list.
- Do not stage ignored runtime artifacts, checkpoints, pretrained assets,
  generated outputs, local docs handoff notes, or tests unless the user asks.
- Preserve user changes. Do not revert unrelated files.
- When updating from remote, prefer fast-forward/rebase workflows and avoid
  unnecessary merge commits.
