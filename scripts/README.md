# Script guide

This directory keeps executable entry points together while classifying them by
side effect. Read the relevant config before running a model command.

## Inspection and validation

| Script | Purpose | Side effect |
|---|---|---|
| `validate_hybrid_config.py` | Validate Hybrid YAML structure and fail on unsupported objective keys | Read-only |
| `check_hybrid_artifacts.py` | Verify expected source/config/document contracts | Read-only |
| `audit_hybrid_requirements.py` | Check that the implemented modules match documented requirements | Read-only |
| `check_xcodec_backend.py` | Inspect or exercise a selected X-Codec backend | May load the explicitly selected external model |
| `smoke_hybrid_forward.py` | Run a tiny random-input forward/loss check | Computes locally; does not train or submit |

## Training, test, and inference

| Script | Purpose | Requirement |
|---|---|---|
| `train_hybrid.py` | Validate a config, then call the preserved training entry point | Starts training; use an explicit config |
| `test_hybrid.py` | Test a checkpoint through the data module | Requires data and a compatible checkpoint |
| `infer_hybrid_directory.py` | Enhance a directory with the Hybrid path | Requires a trusted compatible checkpoint |

The root `train.py` and `test.py` remain for upstream compatibility. They are
not the recommended starting point for understanding the Hybrid work.

## Evaluation and diagnostics

- `eval_intrusive_metrics.py`: intrusive waveform metrics from paired SCP
  manifests.
- `eval_model_generation.py`: teacher/free semantic generation diagnostics.
- `eval_token_validation.py`: historical UniSE token-validation analysis.
- `eval_token_similarity.py`: semantic token agreement analysis.
- `eval_tokenizer_oracle.py`: tokenizer/oracle path diagnostics.
- `score_unise_avqi_comparison.py`: score a prepared UniSE AVQI comparison;
  optional listening exports are deterministically capped at 3--5 paired IDs.
- `transition_margin_probe.py`: prepares or runs the bounded historical
  control/treatment contract. The saved project never authorized its GPU run.

These scripts can write under `outputs/` or `runs/` when explicitly invoked.

The retained data-preparation helpers are
`prepare_hybrid_unise_stream_protocol.py`, `export_tau_fixed_augmented.py`, and
`split_pair_manifest_by_tau_task.py`. They require explicit external data paths
and may write manifests or generated data products.

## Cluster-specific helpers

Files under `cluster/` are retained research infrastructure:

- `cluster/slurm.sh` can call `sbatch` when run outside an allocation and
  requires the explicit `CONFIRM_SLURM_SUBMIT=1` gate. Its historical
  `rolling_cache_smoke` task is retained behind the same gate.
- `cluster/infer_directory.sh` contains a Triton-oriented directory workflow.

They are intentionally absent from the root Quick Start. Review every path and
resource request before adapting them.

## Non-project utilities

A previously tracked Codex credential-configuration helper was removed from the
current project tree because it was unrelated to Hybrid-UniSE and handled API
keys. It remains recoverable from Git history.
