# Architecture and implementation choices

## Scope

This repository contains two related code paths:

1. the preserved upstream UniSE/BiCodec baseline; and
2. an independent, paper-inspired Hybrid-UniSE extension.

The Hybrid extension follows the high-level idea of combining a discriminative
spectral estimate with a semantic-generative estimate. It does not claim exact
module parity with the referenced paper.

## Code boundary

| Area | Primary code | Role |
|---|---|---|
| Upstream UniSE model | `model/model.py`, `model/llm/`, `model/bicodec/` | Preserved decoder-only LM baseline |
| Hybrid composition | `model/hybrid_model.py` | Stage control, losses, checkpoints, and branch composition |
| Spectral front end | `model/audio/stft.py` | Multi-rate SFI STFT/iSTFT |
| Discriminative path | `model/hybrid_discriminative.py` | TF-GridNet-style spectral estimate |
| Semantic LM | `model/hybrid_lm.py` | Conditioning, teacher forcing, free generation, and semantic objectives |
| Token interface | `model/hybrid_xcodec.py`, `model/xcodec_backends.py` | First-RVQ or diagnostic-token targets |
| Refinement | `model/hybrid_refinement.py` | LM-conditioned DPRNN-style generative estimate |
| Fusion | `model/hybrid_fusion.py` | Learned complex-spectral blend |
| Data | `dataloader/` | Native simulation, fixed-pair, WebDataset, and rolling-cache routes |

## Hybrid data flow

1. The degraded waveform is transformed on an SFI grid whose window and hop are
   expressed in milliseconds.
2. The discriminative branch predicts an enhanced complex spectrum at the
   original sample rate.
3. A parallel path resamples the degraded waveform to 16 kHz and produces
   WavLM-style conditioning features.
4. During training, the clean waveform supplies semantic token targets through
   a first-RVQ-compatible or diagnostic-token interface.
5. An autoregressive LM predicts semantic tokens and exposes target-aligned
   hidden states.
6. A DPRNN-style refinement branch conditions on those hidden states and
   predicts a generative spectrum.
7. A sigmoid mask blends the discriminative and generative spectra before
   inverse STFT.

## Explicit implementation choices

| Area | Implementation in this repository | Boundary |
|---|---|---|
| Discriminative network | TF-GridNet-style wrapper | Not claimed to match unpublished internals |
| Semantic conditioner | WavLM-compatible encoder plus adapter | Pretrained weights are not included |
| Clean token target | First-RVQ adapter; deterministic stub for smoke checks | Stub is not a learned codec |
| Diagnostic token target | WavLM layer plus KMeans codebook interface | Research alternative, not paper-exact |
| Semantic LM | Transformer encoder or LLaMA-style causal path | Exact paper LM details are unavailable |
| Refinement | DPRNN-style spectral mapper | Segmentation, masking, and scaling include local choices |
| Quality losses | Optional PMSQE/SQUIM adapters | Not the paper's complete SQA ensemble |
| Resampling | Linear bootstrap utility | Practical engineering choice |

## Stage behavior

The Hybrid Lightning module exposes four stages:

- `disc`: discriminative branch only;
- `gen`: semantic conditioner, LM, and optional refinement;
- `fusion`: frozen component estimates with trainable fusion;
- `joint`: optional joint optimization.

Checkpoint metadata binds the requested stage and major architecture fields.
The latest engineering snapshot also binds a canonical semantic-objective
identity so that a resume cannot silently change the objective.

## Experimental objective hooks

The saved source contains default-neutral hooks for history corruption,
transition weighting, prefix-only auxiliary loss, and a
transition-predecessor margin. These hooks are retained because they are part of
the engineering record.

Only their implementation and wiring were tested. No predecessor-margin
treatment run established an improvement.

## Known architectural boundary

Historical diagnostics indicate two separable problems:

- free-running semantic generation rapidly diverged from teacher-forced
  targets; and
- even oracle LM hidden states did not remove a substantial output-level
  attenuation gap.

A semantic-token improvement alone would therefore not establish final speech
quality.
