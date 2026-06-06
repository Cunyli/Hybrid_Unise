import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate then train Hybrid-UniSE")
    parser.add_argument("--config", default="conf/hybrid_unise_urgent2026.yaml")
    parser.add_argument("--stage", choices=("disc", "gen", "fusion", "joint"), default=None)
    parser.add_argument("--stage_init_checkpoint", default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args, passthrough = parser.parse_known_args()

    config_path = Path(args.config)
    config_dir = config_path.expanduser().resolve().parent
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    config["_config_dir"] = str(config_dir)
    if config.get("model_type") != "hybrid_unise":
        raise ValueError("scripts/train_hybrid.py requires model_type: hybrid_unise")
    if args.stage is not None:
        config["stage"] = args.stage
    if args.stage_init_checkpoint is not None:
        stage_init_checkpoint = Path(args.stage_init_checkpoint).expanduser()
        if not stage_init_checkpoint.is_absolute():
            stage_init_checkpoint = (Path.cwd() / stage_init_checkpoint).resolve()
        config["stage_init_checkpoint"] = str(stage_init_checkpoint)

    run_config_path = config_path
    temp_config = None
    if args.stage is not None or args.stage_init_checkpoint is not None:
        temp_config = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="hybrid_unise_", delete=False)
        yaml.safe_dump(config, temp_config, sort_keys=False)
        temp_config.close()
        run_config_path = Path(temp_config.name)
    try:
        if not args.skip_validation:
            subprocess.run(
                [sys.executable, "scripts/validate_hybrid_config.py", str(run_config_path)],
                check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/audit_hybrid_requirements.py"],
                check=True,
            )

        command = [sys.executable, "train.py", "--config", str(run_config_path), *passthrough]
        print("Running", " ".join(command))
        return subprocess.run(command, check=False).returncode
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    finally:
        if temp_config is not None:
            Path(temp_config.name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
