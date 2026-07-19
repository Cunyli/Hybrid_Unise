# Security notes

This repository loads machine-learning artifacts and can optionally delegate to
Hugging Face model code. Treat both as executable trust boundaries.

- Use checkpoints only from a trusted source. The public loader requests
  tensor-only deserialization and strict state-dict matching.
- `trust_remote_code` defaults to `false`. If a backend requires it, inspect the
  referenced repository and pin an immutable revision before opting in.
- Model weights, datasets, credentials, and environment files must remain
  outside Git. The root `.gitignore` covers common artifact and secret names.
- Do not publish configs containing home directories, cluster paths, job IDs,
  dataset manifests, or listener identifiers.

No security support contact is declared in this archival snapshot. Please use
the repository issue tracker if one is enabled by a future maintainer.
