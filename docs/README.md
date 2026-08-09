# Documentation guide

The root [README](../README.md) gives the shortest project overview. The files
in this directory separate current claims from historical implementation notes.

## Current project documents

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | Maps the upstream baseline and Hybrid implementation to code |
| [Research status](research-status.md) | Separates verified engineering evidence from scientific conclusions |
| [Provenance](provenance.md) | Records upstream code, assets, external dependencies, and Git-history boundaries |

## Historical archive

Files under [archive/](archive/) are retained because they explain how the work
was built or bounded. They are not current reproduction instructions.

| Document | Status |
|---|---|
| [Hybrid-UniSE engineering log](archive/hybrid_unise_engineering_log.md) | Historical implementation and environment notes; contains machine-specific paths |
| [Transition-margin kill-test contract](archive/transition_margin_kill_test_v1.md) | Static paired-probe contract; no GPU treatment was run |

For runnable entry points, see the [configuration guide](../conf/README.md) and
[script guide](../scripts/README.md).
