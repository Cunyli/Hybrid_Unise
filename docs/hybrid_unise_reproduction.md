# Hybrid-UniSE Reproduction Notes

This repository now has a separate `model_type: hybrid_unise` path for
reproducing "A Hybrid Discriminative and Generative System for Universal Speech
Enhancement" (arXiv:2601.19113) without replacing the original UniSE/BiCodec
path.

## Entry Points

- Training/test config: `conf/hybrid_unise_urgent2026.yaml`
- Native multi-rate training example: `conf/hybrid_unise_native_multisr_example.yaml`
- Lightning module: `model/hybrid_model.py`
- Programmatic inference API: `model/hybrid_inference.py`
- Directory inference: `scripts/infer_hybrid_directory.py`
- CPU-oriented contract tests: `tests/test_hybrid_unise.py`

## Implemented Paper Data Flow

- Discriminative branch:
  degraded waveform at the batch sample rate -> SFI STFT -> TF-GridNet-style
  wrapper -> complex enhanced spectrum -> iSTFT.
- Generative refinement branch:
  degraded waveform resampled to 16 kHz -> WavLM-style conditioner and adapter;
  clean waveform during training -> X-Codec first-RVQ token interface; decoder-only
  LM returns token logits, NLL, target-aligned last hidden states and masks; DPRNN
  refinement consumes LM hidden states through cross-attention and predicts a
  complex mask.
- Fusion branch:
  discriminative and generative spectra are aligned on the original-rate SFI
  grid; a lightweight USES-style network predicts a sigmoid mask in `[0, 1]`;
  final spectrum is `M * Y_disc + (1 - M) * Y_gen`.

## Implementation Choices

The short paper does not publish enough detail to exactly recreate every
submodule, and this repository does not include verified public implementations
or weights for several parts. These are intentionally marked as implementation
choices in config/code:

- TF-GridNet wrapper internals in `model/hybrid_discriminative.py`.
- X-Codec first-RVQ tokenizer backend. The default `deterministic_stub` preserves
  shape and training contracts but is not a verified X-Codec model. A verified
  backend can be plugged in as `xcodec.backend: module:callable`; the callable is
  invoked as `(clean_wav_16k, sample_rate=16000)` and can return first-RVQ token
  IDs shaped `[B, T]`, a 3D RVQ tensor, `(tokens, mask)`, or a dict with
  `tokens` and optional `mask`/`lengths`. If a 3D tensor is returned,
  `xcodec.rvq_axis` selects the RVQ dimension and the first RVQ layer is used.
  Padding masks are passed into LM training and DPRNN cross-attention so padded
  token positions do not enter NLL/accuracy or key/value attention.
- DPRNN block count, channel count, normalization and exact segmentation.
- Linear resampling utility for bootstrap CPU tests.
- PMSQE and SQA losses are disabled by default through `external_losses`.
  Enabling either one requires a `module:callable` import path to a verified
  implementation/weights. The code intentionally fails fast instead of replacing
  them with a surrogate loss.

## Stage Training

Set `stage` in `conf/hybrid_unise_urgent2026.yaml`:

- `disc`: trains only the discriminative branch with MR-STFT loss.
- `gen`: trains WavLM adapter, LM and DPRNN refinement; WavLM encoder and
  X-Codec interface remain frozen.
- `fusion`: freezes both branches and trains only fusion.
- `joint`: optional low-learning-rate joint fine-tuning.

By default, `fusion_use_teacher_forcing: false` keeps fusion-stage training
consistent with inference: the generative branch runs autoregressively instead
of seeing clean target tokens. The smoke config sets it to `true` only to expose
token logits for shape checks.

Checkpoints record `hybrid_stage` and the main architecture config. Loading a
checkpoint as a strict resume into a different stage fails fast.

For staged training, use `stage_init_checkpoint` to initialize a new stage from
a previous-stage checkpoint without treating it as an optimizer/trainer resume.
For example, train `stage: disc`, then set `stage: gen` and
`stage_init_checkpoint: /path/to/disc.ckpt`; after that, set `stage: fusion` and
initialize from the selected gen/joint checkpoint. Keep `resume` for same-stage
interrupted-run recovery only; the config validator rejects using both fields at
the same time.

The training wrapper can also apply these overrides without editing the YAML:

```bash
python scripts/train_hybrid.py --config conf/hybrid_unise_urgent2026.yaml --stage disc
python scripts/train_hybrid.py --config conf/hybrid_unise_urgent2026.yaml --stage gen --stage_init_checkpoint /path/to/disc.ckpt
python scripts/train_hybrid.py --config conf/hybrid_unise_urgent2026.yaml --stage fusion --stage_init_checkpoint /path/to/gen.ckpt
```

## Data Contract

Hybrid SR/USE batches can use explicit dict format:

```python
{
    "degraded_wav": Tensor[B, T],
    "clean_wav": Tensor[B, T],
    "sample_rate": LongTensor[B],
    "length": LongTensor[B],
    "utterance_id": list[str],
}
```

SFI STFT requires a single sample rate per batch. If a dataset contains mixed
rates, bucket or sample by rate before batching.

`TrainDataLoadIter` supports `sample_rates: [...]` and samples one rate per
batch, keeping each STFT grid internally consistent. For Hybrid-UniSE dict
batches, set `modes: [se]` so the native simulation path does not emit TSE/RTSE
batches into the SR/USE-only hybrid model path.

The USE fixed-pair loader depends on the external `FixedPairDataset` and
materializes pairs at `target_sample_rate`; set that field explicitly in config
when training fixed-pair stages.

For `ValDataLoadIter`, set `target_sample_rate: null` to keep input files at
their original sample rate for hybrid inference. The generative branch still
runs internally at 16 kHz.

## Commands

Train:

```bash
python scripts/validate_hybrid_config.py conf/hybrid_unise_urgent2026.yaml
python scripts/train_hybrid.py --config conf/hybrid_unise_urgent2026.yaml
```

Test through the Lightning data module:

```bash
python scripts/test_hybrid.py \
  --config conf/hybrid_unise_urgent2026.yaml \
  --stage fusion \
  --ckpt_path /path/to/checkpoint.ckpt \
  --save_enhanced outputs/hybrid_unise_test
```

Directory inference:

```bash
python scripts/infer_hybrid_directory.py \
  --config conf/hybrid_unise_urgent2026.yaml \
  --stage fusion \
  --checkpoint /path/to/checkpoint.ckpt \
  --input-root /path/to/noisy \
  --reference-root /path/to/clean \
  --output-root outputs/hybrid_unise
```

The programmatic API in `model/hybrid_inference.py` exposes the same checkpoint
stage override through `stage="fusion"` or another trained stage.

The directory inference output layout is:

- `outputs/hybrid_unise/wav/*.wav`
- `outputs/hybrid_unise/inf.scp`
- `outputs/hybrid_unise/ref.scp` with entries only when `--reference-root` is provided and matching clean files exist
- optional `disc/` and `gen/` directories with `--save-intermediates`

## Verification

The repository includes tests for:

- SFI STFT/iSTFT parameterization and round trip at 8/16/24/32/48 kHz.
- Mixed-rate batch rejection.
- Fusion mask extremes.
- LM teacher-forcing hidden-state/token alignment.
- Tiny hybrid shape smoke path.

No-torch static checks:

```bash
python scripts/check_hybrid_artifacts.py
python scripts/audit_hybrid_requirements.py
python scripts/validate_hybrid_config.py conf/hybrid_unise_urgent2026.yaml conf/hybrid_unise_smoke.yaml
```

In the current base shell, `pytest` and `torch` are not importable. On this
machine, real `import torch` timed out after 45 seconds in these existing conda
environments: `urgent2026_baseline_track1`, `simulation`, `reuse`, and
`semambapp`; `unise` was also retried with a 120-second timeout and still did
not complete. A faulthandler retry showed the hang inside
`from torch._C import *` during PyTorch C-extension initialization; `pase` shows
the same pattern, while `avqi` has no torch installed. A separate CPU-only
micromamba environment creation attempt for `hybrid_unise_cpu` was stopped after
roughly 2.5 minutes with no solver output. A later isolated Python 3.13 venv at
`/scratch/work/lil14/.venvs/hybrid_unise_cpu` downloaded the CPU torch wheel but
was stopped after the pip install stayed in the wheel installation phase for
more than 15 minutes. Runtime checks require a working project environment
before running:

```bash
python -m pytest -q tests/test_hybrid_unise.py
```

A tiny no-dataset smoke path is also available:

```bash
PYTHONPATH=/scratch/work/lil14/.overlays/hybrid_unise_cpu_torch \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
micromamba run -n unise python scripts/validate_hybrid_config.py conf/hybrid_unise_smoke.yaml

PYTHONPATH=/scratch/work/lil14/.overlays/hybrid_unise_cpu_torch \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
micromamba run -n unise python scripts/smoke_hybrid_forward.py --config conf/hybrid_unise_smoke.yaml
```

The smoke config uses `lm.max_position_embeddings: 256` to keep CPU tests small.
Use `conf/hybrid_unise_urgent2026.yaml` or raise this value for longer
directory inference runs.

## Real X-Codec Backend

The current `unise` environment has `transformers==4.46.3`, which does not
include `XcodecModel`. The non-destructive runtime overlay at
`/scratch/work/lil14/.overlays/hybrid_unise_cpu_torch` now includes
`transformers==4.57.6` and `tokenizers==0.22.2`; with the usual `PYTHONPATH`
prefix, the API check passes:

```bash
PYTHONPATH=/scratch/work/lil14/.overlays/hybrid_unise_cpu_torch \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
micromamba run -n unise python scripts/check_xcodec_backend.py
```

The public checkpoint `hf-audio/xcodec-wavlm-more-data` has been downloaded to
`pretrained/xcodec/hf-audio_xcodec-wavlm-more-data`, and the main configs now
use the real backend:

```yaml
xcodec:
  model_path: ./pretrained/xcodec/hf-audio_xcodec-wavlm-more-data
  backend: model.xcodec_backends:TransformersXCodecFirstRVQ
  backend_kwargs:
    encode_method: encode
    token_attr: audio_codes
    trust_remote_code: true
  rvq_axis: 1
```

Actual token extraction was verified with:

```bash
PYTHONPATH=/scratch/work/lil14/.overlays/hybrid_unise_cpu_torch \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
micromamba run -n unise python scripts/check_xcodec_backend.py \
  --model-path pretrained/xcodec/hf-audio_xcodec-wavlm-more-data \
  --vocab-size 65536
```

For short random 16 kHz waveforms, first-RVQ token IDs stayed below 1024 in the
local smoke check, so `xcodec.vocab_size: 1024` remains the current training
choice. Increase it if a broader corpus check reports larger valid token IDs.

## PMSQE

`model.quality_losses:AsteroidPMSQELoss` wraps Asteroid's `SingleSrcPMSQE` and
accepts waveforms by computing the required STFT power spectrum internally. It
has been smoke-tested at 16 kHz in the `unise` overlay runtime, including a tiny
Hybrid-UniSE gen-stage loss pass. Enable it with:

```yaml
external_losses:
  pmsqe:
    enabled: true
    import_path: model.quality_losses:AsteroidPMSQELoss
```

## SQA Baseline

The paper's SQA term references a multi-metric quality model ensemble
(MOS/DNSMOS/ScoreQ/UTMOS/NISQA). That exact ensemble is not available here. The
current implementation provides an implementation-choice baseline:
`model.quality_losses:TorchaudioSquimObjectiveLoss`, backed by TorchAudio
`SQUIM_OBJECTIVE`. It predicts a quality vector and penalizes distance between
the enhanced and clean predictions.

Enable it for fusion or joint stages:

```yaml
loss_weights:
  fusion:
    sqa: 0.01

external_losses:
  sqa:
    enabled: true
    import_path: model.quality_losses:TorchaudioSquimObjectiveLoss
    kwargs:
      mode: l1
      weights: [1.0, 1.0, 0.05]
```

It has been smoke-tested as part of a tiny Hybrid-UniSE fusion loss pass. SQUIM
expects 16 kHz audio, so use it with 16 kHz fixed-pair training or add explicit
resampling before enabling it on native multi-rate fusion batches.

## Train, Validation, Test

`scripts/train_hybrid.py` validates the config, applies optional stage
overrides, then calls `train.py`. `train.py` builds `DataModule` from
`dataset_config`:

- `train_kwargs`: used by `trainer.fit` for training batches.
- `val_kwargs`: used by `trainer.fit` for validation batches.
- `test_kwargs`: used by `trainer.test` and `scripts/test_hybrid.py`.

For large compressed dry-data pools, use:

```yaml
dataset_config:
  train_kwargs:
    dataset_type: use_simulation_rolling_cache
    batch_format: dict
```

This materializes a bounded tar-shard cache from archive members and avoids full
archive extraction. For validation and test, prefer fixed-pair or directory
datasets so metrics are stable across runs.

Recommended staged command sequence:

```bash
export HYBRID_ENV="PYTHONPATH=/scratch/work/lil14/.overlays/hybrid_unise_cpu_torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

env $HYBRID_ENV micromamba run -n unise python scripts/train_hybrid.py \
  --config conf/hybrid_unise_rolling_cache_example.yaml --stage disc

env $HYBRID_ENV micromamba run -n unise python scripts/train_hybrid.py \
  --config conf/hybrid_unise_rolling_cache_example.yaml --stage gen \
  --stage_init_checkpoint /path/to/disc.ckpt

env $HYBRID_ENV micromamba run -n unise python scripts/train_hybrid.py \
  --config conf/hybrid_unise_rolling_cache_example.yaml --stage fusion \
  --stage_init_checkpoint /path/to/gen.ckpt
```

For test/inference from a trained fusion checkpoint:

```bash
env $HYBRID_ENV micromamba run -n unise python scripts/test_hybrid.py \
  --config conf/hybrid_unise_urgent2026.yaml \
  --stage fusion \
  --ckpt_path /path/to/fusion.ckpt \
  --save_enhanced outputs/hybrid_unise_test
```
