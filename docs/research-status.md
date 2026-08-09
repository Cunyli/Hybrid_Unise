# Research status and stopped directions

This page separates verified engineering observations from interpretation.
Nothing below is a paper benchmark.

## Claim matrix

| Claim | Status | Evidence boundary |
|---|---|---|
| Hybrid branches and tensor routes are implemented | Verified engineering fact | Source, configuration, and contract tests |
| Objective/checkpoint identity is fail-closed | Verified for the saved Phase A implementation | Focused unit tests and static review |
| WavLM-KMeans semantic generation is effective | Not established | Tiny diagnostics showed severe free-run failure |
| Transition-predecessor margin improves generation | Unknown | Static implementation only; no treatment run |
| Final Hybrid audio quality is acceptable | Not established | No compatible result checkpoint or manifest |
| Paper results were reproduced | No | Architecture, data, assets, and metrics are not paper-exact |

## Engineering verification retained with the project

The organized tree was rechecked locally and recorded:

- 195 local tests passing with real PyTorch/Transformers and a minimal
  Lightning import stub used by the local harness;

Earlier saved verification also recorded:

- 60 targeted static/CPU/offline checks passing in an isolated Triton copy;
- an earlier broader remote run with 814 passes and three frozen, pre-existing
  ABA-related failures.

These counts establish software-contract coverage at those snapshots. They do
not establish convergence, perceptual quality, or paper fidelity.

## Limited semantic-token diagnostics

The main observed failure was in free-running semantic generation:

| Observation | Historical diagnostic |
|---|---|
| Step 20 | Single-token collapse |
| Step 200 | Small attractor rather than healthy sequence generation |
| Frozen panel size | 4 `healthy_low` examples |
| Free/oracle positional agreement | Approximately 1.35% |
| First error positions | 2 / 2 / 4 / 2 |
| First-error pattern | 4/4 copied the predecessor |
| Confound | All four predecessor tokens were token 5 |

Interpretation: the evidence is consistent with a predecessor or onset
attractor, but the panel is too small and token 5 is fully confounded with the
observed predecessor errors. It does not identify a general causal mechanism.

## Closed decoding direction

A screen of ten history-corruption and run-penalty settings produced zero
complete candidates. The clean-speech distribution gate could pass while the
speech-variation run-length/transition gate failed. This direction was stopped
rather than widened into another search.

## Transition-margin candidate

A transition-predecessor margin was implemented with a neutral default and an
auditable control-versus-treatment contract. The contract fixed the source
checkpoint, data order, seed, objective identity, time limits, and stop gates.

No GPU pair was submitted. There is no treatment checkpoint and no GO result.
The package is retained under `conf/experiments/`, `scripts/`,
`tests/`, and `docs/archive/` as a falsifiable experiment record only.

## Independent refinement limitation

Historical diagnostics suggested that oracle LM hidden states were better than
free-running hidden states, yet the oracle-output path remained approximately
5.94 dB below the clean signal level. This is evidence of a separate refinement
or scale problem. Improving semantic token prediction alone would not establish
final audio quality.

## What was not established

- paper-exact architecture or training data;
- reproduction of the paper's reported metrics;
- generalization beyond the tiny diagnostic panel;
- a successful predecessor-margin treatment;
- perceptually acceptable Hybrid enhanced audio;
- a release-ready trained Hybrid checkpoint.

## Stop boundary

Further variant patching, paper-exact auditing, and GPU validation were stopped.
The current repository is the final organized engineering record for review.
Any future experiment should begin with a new, explicit data, asset, metric,
budget, and stopping contract rather than being implied by this repository.
