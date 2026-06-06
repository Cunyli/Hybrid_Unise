import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate then test Hybrid-UniSE")
    parser.add_argument("--config", default="conf/hybrid_unise_urgent2026.yaml")
    parser.add_argument("--save_enhanced", default=None)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--stage", choices=("disc", "gen", "fusion", "joint"), default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    if config.get("model_type") != "hybrid_unise":
        raise ValueError("scripts/test_hybrid.py requires model_type: hybrid_unise")
    if args.stage is not None:
        config["stage"] = args.stage
    if args.ckpt_path is not None:
        config["ckpt_path"] = args.ckpt_path
    if args.save_enhanced is not None:
        config["save_enhanced"] = args.save_enhanced
    config["stage_init_checkpoint"] = None

    run_config_path = config_path
    temp_config = None
    if args.stage is not None or args.ckpt_path is not None or args.save_enhanced is not None:
        temp_config = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="hybrid_unise_test_", delete=False)
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

        command = [sys.executable, "test.py", "--config", str(run_config_path)]
        if args.save_enhanced is not None:
            command.extend(["--save_enhanced", args.save_enhanced])
        if args.ckpt_path is not None:
            command.extend(["--ckpt_path", args.ckpt_path])
        if args.stage is not None:
            command.extend(["--stage", args.stage])
        print("Running", " ".join(command))
        return subprocess.run(command, check=False).returncode
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    finally:
        if temp_config is not None:
            Path(temp_config.name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
