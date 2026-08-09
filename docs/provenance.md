# Repository provenance and asset boundaries

## Code origin

The project began from Alibaba's
[unified-audio / QuarkAudio-UniSE](https://github.com/alibaba/unified-audio/tree/main/QuarkAudio-UniSE)
codebase.

The current repository intentionally retains the upstream UniSE/BiCodec path
and adds substantial independent Hybrid-UniSE work. It is not an official
Alibaba repository and is not an official implementation of the Hybrid-UniSE
paper.

The canonical development lineage is preserved in Git:

```text
73110b6  initial QuarkAudio-UniSE import
920d181  Hybrid-UniSE reproduction pipeline
cf88924  semantic-transition stabilization snapshot
c1772cf  auditable transition-margin Phase A contract
```

A separate one-commit curated view was temporarily created during cleanup. Its
commit is preserved as an archive branch, but the complete lineage above
remains the canonical project.

## Included reference assets

| Path | Origin/status | How it may be described |
|---|---|---|
| `assets/upstream/audio_samples/` | Inherited from the upstream source tree | Upstream UniSE demo material; not Hybrid project output |
| `assets/upstream/figures/unise_architecture.png` | Inherited upstream figure | Upstream reference figure |
| `assets/archive/simulation_non_silence_debug.png` | Retained development diagnostic | Debug artifact, not an evaluation result |

The upstream audio includes folders named `enhanced`. That label reflects the
upstream demo layout and must not be interpreted as evidence produced by this
Hybrid implementation.

## External assets not included

The repository does not include:

- Hybrid training checkpoints;
- external pretrained model weights;
- training, validation, or test datasets;
- private manifests referenced by cluster-bound experiment configurations;
- a project-produced enhanced-audio result set.

The placeholder directories `checkpoints/`, `pretrained/`, `outputs/`,
`logs/`, `runs/`, and `tmp/` keep runtime artifacts out of source code.

## External model families

The source exposes optional interfaces to external model families:

| Asset or family | Intended use | Boundary |
|---|---|---|
| Microsoft WavLM | Semantic conditioner or diagnostic feature extractor | Weights are external |
| X-Codec-compatible model | First-RVQ token backend | Model card and code trust must be reviewed |
| TorchAudio SQUIM | Optional quality-loss baseline | Not the paper's complete SQA ensemble |
| Asteroid PMSQE | Optional perceptual-loss adapter | Separate dependency and license |

Some historical configs identify exact local copies or enable remote model
code. Those paths document the experiment environment; they are not bundled
assets or portable defaults.

## Sharing boundary

External datasets and model weights remain separate from this repository.
Before sharing the inherited demo media beyond research review, verify its
upstream terms independently.
