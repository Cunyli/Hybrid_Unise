import importlib.util
from pathlib import Path

import yaml


def load_validate_config():
    script_path = Path(__file__).with_name("validate_hybrid_config.py")
    spec = importlib.util.spec_from_file_location("validate_hybrid_config", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_config


validate_config = load_validate_config()


REQUIRED_FILES = [
    "README.md",
    "checkpoints/README.md",
    "pretrained/README.md",
    "outputs/README.md",
    "conf/README.md",
    "conf/experiments/hybrid_unise_urgent2026.yaml",
    "conf/examples/hybrid_unise_smoke.yaml",
    "conf/experiments/hybrid_unise_rolling_cache_example.yaml",
    "model/audio/stft.py",
    "model/hybrid_model.py",
    "model/hybrid_discriminative.py",
    "model/hybrid_lm.py",
    "model/hybrid_xcodec.py",
    "model/xcodec_backends.py",
    "model/quality_losses.py",
    "model/hybrid_refinement.py",
    "model/hybrid_fusion.py",
    "model/hybrid_losses.py",
    "model/hybrid_inference.py",
    "scripts/infer_hybrid_directory.py",
    "scripts/train_hybrid.py",
    "scripts/test_hybrid.py",
    "scripts/smoke_hybrid_forward.py",
    "scripts/validate_hybrid_config.py",
    "scripts/audit_hybrid_requirements.py",
    "scripts/check_xcodec_backend.py",
    "tests/test_hybrid_unise.py",
    "tests/test_rolling_cache.py",
    "docs/README.md",
    "docs/architecture.md",
    "docs/research-status.md",
    "docs/provenance.md",
    "docs/archive/hybrid_unise_engineering_log.md",
    "scripts/README.md",
]

REQUIRED_TRAIN_CONFIG_KEYS = [
    "model_type",
    "stage",
    "sfi",
    "discriminative",
    "wavlm",
    "xcodec",
    "lm",
    "refinement",
    "fusion",
    "loss_weights",
    "external_losses",
    "lm_stft_alignment_mode",
    "lm_stft_alignment_tolerance",
    "dataset_config",
]

REQUIRED_SMOKE_CONFIG_KEYS = [
    key for key in REQUIRED_TRAIN_CONFIG_KEYS if key != "dataset_config"
]


def main() -> int:
    errors: list[str] = []
    for file_name in REQUIRED_FILES:
        if not Path(file_name).is_file():
            errors.append(f"missing required file: {file_name}")

    for config_name in (
        "conf/experiments/hybrid_unise_urgent2026.yaml",
        "conf/examples/hybrid_unise_smoke.yaml",
        "conf/experiments/hybrid_unise_rolling_cache_example.yaml",
    ):
        path = Path(config_name)
        if not path.is_file():
            continue
        config = yaml.safe_load(path.read_text())
        required_keys = (
            REQUIRED_SMOKE_CONFIG_KEYS
            if config_name.endswith("_smoke.yaml")
            else REQUIRED_TRAIN_CONFIG_KEYS
        )
        for key in required_keys:
            if key not in config:
                errors.append(f"{config_name} missing key: {key}")
        errors.extend(f"{config_name}: {error}" for error in validate_config(config, path))

    engineering_log = Path("docs/archive/hybrid_unise_engineering_log.md")
    doc_text = engineering_log.read_text() if engineering_log.is_file() else ""
    doc_text_lower = doc_text.lower()
    for required_text in (
        "implementation choice",
        "PMSQE and SQA losses are disabled",
        "fusion_use_teacher_forcing: false",
        "python scripts/smoke_hybrid_forward.py",
        "python scripts/audit_hybrid_requirements.py",
    ):
        if required_text.lower() not in doc_text_lower:
            errors.append(
                "docs/archive/hybrid_unise_engineering_log.md "
                f"missing text: {required_text}"
            )

    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print("Hybrid-UniSE artifact check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
