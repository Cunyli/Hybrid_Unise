# Transition-predecessor-margin paired kill test v1

> **Archived static contract; no GPU run occurred.** The current project stops
> before submission. The launch conditions below are retained for auditability
> and are not active authorization to resume this experiment.

## Decision boundary

This test answers only one question: does a fixed predecessor-margin auxiliary
loss materially reduce the already-observed free-run transition stall relative
to an otherwise identical control? A `GO` keeps this candidate for a later
minimal patch. It does not claim usable final audio, solve refinement
attenuation, or reproduce the Hybrid-UniSE paper.

Phase A ends after static preparation. Running either arm requires fresh user
authorization, a read-only queue/topology check, and a verified preflight
receipt. No second hypothesis or long run follows either decision.

Static preparation deliberately reports `ready_to_launch=false` until a
self-hashed `transition_margin_topology/v1` receipt proves a one-GPU topology
check performed within the same 60-minute pair window. That dynamic receipt belongs
to Phase B and is not fabricated during this local Phase A.

The Phase A command is non-training:

```bash
python -m scripts.transition_margin_probe prepare \
  --spec conf/experiments/transition_margin_probe_pair_v1.yaml \
  --output-dir /tmp/transition_margin_probe_pair_v1
```

It writes two arm configs, a preflight receipt, immutable `not_run` launch and
result receipts, and `queue_topology_receipt.template.json`. Re-running
preparation may refresh untouched `not_run` templates, but it refuses to
overwrite or silently rebind any started arm. The runner contains no `sbatch`
or scheduler submission command.

## Preregistered pair

- Source checkpoint:
  `latest_epoch=00-step=000200.ckpt`, SHA256
  `55e941154e650567572787b0ae4c121a2eed109a3aa30818dcdb4f661956adf6`.
- Code ancestry: `cf88924d56ecb8c53ea45bdf4005300e8a4c07cf` must be an
  ancestor; the run worktree must be clean.
- Order: control, then treatment, each in a fresh process on exactly one GPU.
- Budget: 100 optimizer updates per arm, at most 30 minutes per arm and 60
  minutes total. A timeout is a failed arm and therefore `NO-GO`; it is not a
  reason to retry.
- Shared trainer/data seed: `3407`; identical manifest SHAs, batch size,
  worker count, shard order, buffers, sample count, validation recipes, model,
  optimizer, learning rate, and source checkpoint.
- Control: `transition_predecessor_margin=1.0`, weight `0.0`.
- Treatment: `transition_predecessor_margin=1.0`, weight `0.25`.
- No resume, history corruption, decoding penalty, waveform loss, skipped bad
  samples, or refinement training.

The treatment weight is fixed at `0.25` because it scales the active hinge
gradient to one quarter of that auxiliary term's unweighted value. This is a
conservative one-shot value, not a tuned optimum or a claimed bound relative
to the differently reduced CE gradient.

The complete immutable paths and SHA256 values are in
`conf/experiments/transition_margin_probe_pair_v1.yaml`. Preparation must produce
preflight, launch, and result receipts that all bind the canonical objective
JSON/SHA for both arms.

## Evaluation panel and metrics

Use this exact frozen `healthy_low` panel:

| UID | Slice | Step-200 first error | First-error predecessor |
| --- | --- | ---: | ---: |
| `tau_selected_phone_room_00092_V02_cs` | CS | 2 | 5 |
| `tau_selected_phone_room_00093_V02_sv` | SV | 2 | 5 |
| `tau_selected_phone_room_00094_V06_cs` | CS | 4 | 5 |
| `tau_selected_phone_room_00095_V06_sv` | SV | 2 | 5 |

The panel manifest SHA256 is
`ca7ae088c47a3bd2027222212b76617d7475a1df70af4b133cbb438ddffdd2ed`;
the step-200 drift rows SHA256 is
`6e7ed86fa627f7867b9692d886ba606e6ade04c389d3e066544d9d078d367ec5`.
The four saved tensor paths, sizes, and SHA256 values in the pair spec bind the
`token_targets` clean-oracle sequences. Report overall, `CS`, and `SV`
separately; do not collapse the result to AVQI or training loss.

Before any GPU run, one read-only inspection must confirm that the frozen panel
can supply the degraded/clean input, sample rate, valid length, and oracle token
mask needed for both teacher-forced and deterministic free-run evaluation. The
current local contract has verified only the `token_targets` field. If that
input schema cannot be bound without changing the panel, the pair is not
launchable and the result is `NO-GO` without training.

Metric conventions are fixed as follows: token positions are zero-based; a
row with no free-run error uses its valid-token length as the first-error
position; positional agreement, predecessor-copy rate, teacher accuracy, and
teacher NLL are pooled from their raw numerators/denominators within overall,
`CS`, and `SV`; the median first-error position is computed across the four
row-level positions. Free-run predecessor copies are counted only where the
oracle target changes and both oracle positions are valid, using
`free_token[t] == oracle_token[t-1]`. The teacher-forced predecessor metric is
not substituted for this free-run gate.

All three primary gates must pass:

1. Free-run positional agreement against clean-oracle targets improves by at
   least `+5.0` absolute percentage points overall and by at least `+2.0`
   points in both `CS` and `SV`, relative to the paired control.
2. Median first-error position moves at least `+2` tokens later, with a later
   first error on at least `3/4` rows.
3. Over all valid transition positions (the denominator used by
   `transition_predecessor_rate`), predecessor-copy rate falls by at least
   `25%` relative and `5.0` absolute percentage points overall; neither `CS`
   nor `SV` may worsen by more than `5.0` points.

Guardrails:

- Teacher-forced accuracy may fall by at most `2.0` absolute points in either
  `CS` or `SV`; teacher-forced NLL may rise by at most `0.10` in either slice.
- Losses and gradients must remain finite, both arms must complete exactly 100
  updates, and artifact/objective/config receipts must match.
- One listener, identified as `project_owner_01`, rates the same four rows in
  one session on one fixed device and volume. Do not loudness-normalize.
- For each UID, label the arm with the lexicographically smaller SHA256 of
  `pair_id + "\\0" + uid + "\\0" + arm` as `A`, the other as `B`; keep the
  mapping sealed until the rating receipt is written.
- Present noisy and clean references plus opaque `A`/`B`. Permit at most two
  plays per clip. Rate attenuation, burst, instability, and intelligibility as
  `a_worse`, `tie`, or `b_worse`.
- After unsealing, treatment being worse in any category on any row is a
  guardrail failure. A tie is not a failure.

## Decision rule

`GO` requires every primary gate and every guardrail. Flat or mixed movement,
an `SV` regression, a receipt/config/artifact mismatch, inability to form the
exact pair within the 60-minute GPU budget, or any explanation that requires a
second new hypothesis is `NO-GO`. On `NO-GO`, stop this bugfix line.
