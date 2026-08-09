# Configuration guide

Configuration files are grouped by intent so that a reader can distinguish a
portable example from a historical cluster run.

## Root baseline files

| File | Purpose | Portability |
|---|---|---|
| `config.yaml` | Preserved upstream UniSE default | Requires upstream BiCodec/WavLM assets |
| `simulation_train.yaml` | Preserved simulation recipe used by several loaders | Requires external speech/noise/RIR sources |
| `generated/` | Runtime-generated configurations | Ignored except for the directory placeholder |

## Examples

| File | Purpose | Safe expectation |
|---|---|---|
| `examples/hybrid_unise_smoke.yaml` | Tiny random-input tensor-flow smoke | No quality claim; deterministic token stub |
| `examples/hybrid_unise_example.yaml` | Readable architecture-scale Hybrid config | No dataset, weight, or checkpoint included |
The two files above are the only configs used in the root README's local
inspection commands.

## Templates

`templates/hybrid_unise_native_multisr_template.yaml` documents the native
multi-rate data contract. Its `/path/to/...` values must be replaced before
validation or use; that deliberate fail-closed behavior distinguishes it from
the self-contained examples.

## Historical experiments

Files under `experiments/` preserve actual research intent and may contain
absolute Triton storage paths. They are not portable defaults.

| File or family | Status |
|---|---|
| `hybrid_unise_urgent2026.yaml` | Main staged Hybrid engineering config |
| `hybrid_unise_rolling_cache_example.yaml` | Rolling-cache experiment config with external archives |
| `unise/hybrid_unise_webdataset_stream_v1.yaml` | Historical UniSE streaming-data experiment |
| `unise/rolling_cache_unise_prototype.yaml` | Historical UniSE rolling-cache prototype |
| `unise/tau_fixed_aug_*.yaml` | Historical UniSE augmentation and adapter/LoRA comparisons |
| `tau_fixed_*.yaml` | TAU fixed-pair comparisons and batch/step variants |
| `tau_sd_unise.yaml` | TAU speech-degradation experiment config |
| `unise/use_simulation_phone_room_16k.yaml` | Simulation recipe used by the historical UniSE cache/streaming experiments |
| `transition_margin_probe_pair_v1.yaml` | Preregistered control/treatment contract; never submitted to GPU |

Do not infer that a historical config produced a successful checkpoint merely
because the file is present.

Files under `experiments/unise/` use the preserved upstream UniSE training
schema. They are YAML-checked historical inputs, not Hybrid configs accepted by
`scripts/validate_hybrid_config.py`.

## Archive

`archive/config_sr_smoke.generated.yaml` is a retained generated snapshot from
an upstream smoke workflow. It is not a maintained input.

## Path semantics

Most paths are interpreted relative to the repository working directory, not
relative to the YAML file. Historical absolute paths document the original
environment and must be reviewed before reuse.

No configuration in this directory bundles a dataset, checkpoint, or pretrained
model.
