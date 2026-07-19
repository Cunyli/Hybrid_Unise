# Architecture and implementation choices

## Scope

The implementation follows the high-level idea of combining a discriminative
spectral estimate with a semantic-generative estimate. It does not claim exact
module parity with the referenced paper.

## Data flow

1. The degraded waveform is transformed on an SFI grid whose window and hop are
   expressed in milliseconds.
2. The discriminative branch predicts an enhanced complex spectrum at the
   original sample rate.
3. A parallel path resamples the degraded waveform to 16 kHz and produces
   WavLM-style conditioning features.
4. During training, the clean waveform supplies semantic token targets through
   a first-RVQ-compatible tokenizer interface.
5. An autoregressive LM predicts semantic tokens and exposes hidden states.
6. A DPRNN-style refinement branch conditions on those hidden states and
   predicts a generative spectrum.
7. A sigmoid mask blends the discriminative and generative spectra before
   inverse STFT.

## Explicit implementation choices

| Area | Public implementation | Boundary |
|---|---|---|
| Discriminative network | TF-GridNet-style wrapper | Not claimed to match unpublished internals |
| Semantic conditioner | WavLM-compatible encoder plus adapter | Pretrained weights are not included |
| Clean token target | First-RVQ adapter; deterministic stub for smoke checks | Stub is not a learned codec |
| Diagnostic token target | WavLM layer plus KMeans codebook interface | Research alternative, not paper-exact |
| Semantic LM | Transformer encoder or LLaMA-style causal path | Exact paper LM details are unavailable |
| Refinement | DPRNN-style spectral mapper | Several segmentation/mask choices are local |
| Quality losses | Optional PMSQE/SQUIM adapters | Not the paper's complete SQA ensemble |
| Resampling | Linear utility in the bootstrap path | A practical engineering choice |

## Stage behavior

The Lightning module exposes four stages:

- `disc`: discriminative branch only;
- `gen`: semantic conditioner, LM, and optional refinement;
- `fusion`: frozen component estimates with trainable fusion;
- `joint`: optional joint optimization.

Checkpoint metadata binds the requested stage and major architecture fields.
The public inference entry additionally requires strict state-dict matching.

## Semantic objective extensions

The saved source snapshot contains experimental hooks for history corruption,
transition weighting, and a transition-predecessor margin. Their neutral
defaults preserve the base objective. These hooks are retained as part of the
engineering record, but no treatment run established that they improve the
model.
