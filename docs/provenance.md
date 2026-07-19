# Provenance and external assets

## Code origin

The project began as a private engineering branch derived from the
Apache-2.0-licensed
[Alibaba unified-audio / QuarkAudio-UniSE](https://github.com/alibaba/unified-audio/tree/main/QuarkAudio-UniSE)
tree. The public snapshot was curated from a saved internal source snapshot
into a new, unrelated Git history.

The clean history is intentional: the private development history contained
machine-specific paths, dataset routes, experiment receipts, upstream demo
assets, and unrelated utilities. Removing that history does not remove upstream
attribution; attribution is retained in the root `NOTICE` and `LICENSE`.

## What is excluded

This repository does not redistribute:

- upstream demo audio or architecture images;
- checkpoints or model weights;
- training, validation, or test datasets;
- private manifests, cluster launchers, paths, job receipts, or listener IDs;
- the machine-level Conda environment export;
- stopped experiment launch packages.

As a result, there is intentionally no audio gallery. The upstream audio found
in the original source tree predates the Hybrid study and must not be presented
as this project's output.

## External model assets

The source exposes optional interfaces to external model families, but no asset
is downloaded or bundled.

| Asset or family | Use | License boundary |
|---|---|---|
| [`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus) | Optional WavLM conditioner | Model card lists CC BY-SA 3.0 |
| X-Codec-compatible model | Optional first-RVQ token backend | Check the selected model card; licenses vary |
| [`hf-audio/xcodec-wavlm-more-data`](https://huggingface.co/hf-audio/xcodec-wavlm-more-data) | Asset used in a private diagnostic branch | Model card lists CC BY 4.0; not included |
| TorchAudio SQUIM | Optional quality-loss baseline | Distributed separately under its own terms |
| Asteroid PMSQE | Optional perceptual-loss adapter | Distributed separately under its own terms |

Licenses and model cards can change. A downstream user must verify the exact
asset revision and terms before use. The Apache-2.0 code license does not
relicense model weights or datasets.

## Remote code and revisions

The public X-Codec adapter defaults `trust_remote_code` to `false` and accepts an
optional immutable `revision`. Enabling remote code without reviewing and
pinning it is outside this repository's safe default.
