# Asset guide

Assets are separated from source code and generated outputs.

## `upstream/`

- `audio_samples/` contains the SR, SS, and TSE demo layout inherited from
  QuarkAudio-UniSE.
- `figures/unise_architecture.png` is the inherited UniSE architecture figure.

These files are reference material. In particular, audio under folders named
`enhanced` was not produced by the Hybrid-UniSE implementation in this
repository.

## `archive/`

`simulation_non_silence_debug.png` is a retained development plot from the
simulation utility. It is not a benchmark figure or project result.

Project-generated artifacts belong in `outputs/` or a run-specific
`runs/<run_id>/outputs/`, both ignored by Git.
