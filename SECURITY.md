# Security notes

This is a research repository with checkpoint loading, optional remote model
code, machine-bound configurations, and cluster launch helpers. Review those
trust boundaries before running anything.

## Checkpoints and model artifacts

- Load checkpoints only from a trusted source. Parts of the preserved research
  code use PyTorch deserialization modes that may execute serialized Python
  objects.
- Architecture and objective checks reduce accidental mismatch; they do not
  make an untrusted checkpoint safe.
- Keep model weights in `pretrained/` or `checkpoints/`, both of which are
  ignored by Git.

## Remote model code

- Some historical X-Codec configurations enable Hugging Face
  `trust_remote_code`.
- Inspect the model repository and pin an immutable revision before enabling
  remote code in a new environment.
- The portable example keeps external backends disabled.

## Cluster scripts

- `scripts/cluster/slurm.sh` can call `sbatch` when it is executed outside an
  existing Slurm allocation.
- Cluster scripts and files under `conf/experiments/` are retained as
  historical research infrastructure, not as safe Quick Start commands.
- Review paths, resource requests, checkpoint locations, and output locations
  before adapting them.

## Credentials and private paths

- Do not commit API keys, tokens, passwords, `.env` files, SSH keys, dataset
  manifests, or private checkpoints.
- The root `.gitignore` excludes common credential and model-artifact names.
- Historical experiment configs contain internal storage paths. They are useful
  for provenance but should be sanitized before sharing beyond the intended
  research-review audience.

No dedicated security support contact is declared for this research archive.
