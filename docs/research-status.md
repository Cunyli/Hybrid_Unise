# Research status and stopped directions

This page separates verified development observations from interpretation.
None of the entries below is a paper benchmark.

## Verified engineering facts

- The public source contains discriminative, semantic-generative, refinement,
  and fusion paths.
- The saved private development snapshot passed 158 local tests. Separate
  isolated static/CPU checks also passed, but those environments and assets are
  not distributed here.
- No trained Hybrid checkpoint, evaluation manifest, or project-produced audio
  sample is included in this public repository.
- The transition-predecessor margin is present as a default-off objective hook.
  No margin treatment checkpoint was trained.

The historical test count establishes only that the inspected implementation
met its engineering contracts at that snapshot. The smaller public suite is the
only suite a visitor can run from this repository alone.

## Limited semantic-token diagnostics

The main failure was in free-running semantic generation:

| Observation | Historical diagnostic |
|---|---|
| Step 20 | Single-token collapse |
| Step 200 | Small attractor rather than healthy sequence generation |
| Frozen panel size | 4 `healthy_low` examples |
| Free/oracle positional agreement | Approximately 1.35% |
| First error positions | 2 / 2 / 4 / 2 |
| First-error pattern | 4/4 copied the predecessor |
| Confound | All four predecessor tokens were token 5 |

Interpretation: the evidence is consistent with a predecessor/onset attractor,
but the panel is too small and token 5 is fully confounded with the observed
predecessor errors. It does not identify a general causal mechanism.

## Closed decoding direction

A screen of ten history-corruption/run-penalty decoding settings produced zero
complete candidates. The clean-speech distribution gate could pass while the
speech-variation run-length/transition gate failed. This direction was stopped
rather than widened into another search.

## Independent refinement limitation

Historical diagnostics suggested that oracle LM hidden states were better than
free-running hidden states, yet the oracle-output path remained approximately
5.94 dB below the clean signal level. This indicates a separate refinement or
scale problem. Improving semantic token prediction alone would not establish
final audio quality.

## What was not established

- paper-exact architecture or training data;
- reproduction of the paper's reported metrics;
- generalization beyond the tiny diagnostic panel;
- a successful predecessor-margin treatment;
- perceptually acceptable enhanced audio;
- a release-ready trained checkpoint.

## Stop decision

Further variant patching, paper-exact auditing, and GPU validation were stopped.
The remaining output is this bounded engineering record. Any future experiment
should begin as a new project with its own data, asset, metric, and stopping
contract rather than being implied by this archive.
