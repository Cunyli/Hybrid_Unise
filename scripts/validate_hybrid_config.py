import argparse
from pathlib import Path

import yaml


VALID_STAGES = {"disc", "gen", "fusion", "joint"}


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def is_module_callable(value) -> bool:
    return isinstance(value, str) and ":" in value and all(value.split(":", 1))


def validate_config(config: dict, path: Path) -> list[str]:
    errors: list[str] = []
    config_dir = Path(config.get("_config_dir", path.parent)).expanduser()
    require(config.get("model_type") == "hybrid_unise", "model_type must be hybrid_unise", errors)
    require(config.get("stage") in VALID_STAGES, f"stage must be one of {sorted(VALID_STAGES)}", errors)
    require(
        not (config.get("resume") and config.get("stage_init_checkpoint")),
        "resume and stage_init_checkpoint are mutually exclusive",
        errors,
    )
    stage_init_checkpoint = config.get("stage_init_checkpoint")
    if stage_init_checkpoint:
        if isinstance(stage_init_checkpoint, str) and stage_init_checkpoint.startswith("/path/to/"):
            errors.append(f"stage_init_checkpoint still contains placeholder path {stage_init_checkpoint!r}")
        else:
            stage_init_path = Path(stage_init_checkpoint).expanduser()
            if not stage_init_path.is_absolute():
                stage_init_path = config_dir / stage_init_path
            if not stage_init_path.is_file():
                errors.append(f"stage_init_checkpoint does not exist: {stage_init_checkpoint!r}")

    sfi = config.get("sfi") or {}
    supported_sample_rates = sfi.get("supported_sample_rates") or []
    require(bool(supported_sample_rates), "sfi.supported_sample_rates must not be empty", errors)
    require(16000 in supported_sample_rates, "sfi.supported_sample_rates must include 16000 for the generative branch", errors)
    require(float(sfi.get("window_ms", 0.0)) > 0.0, "sfi.window_ms must be positive", errors)
    require(float(sfi.get("hop_ms", 0.0)) > 0.0, "sfi.hop_ms must be positive", errors)

    xcodec = config.get("xcodec") or {}
    xcodec_backend = xcodec.get("backend", "deterministic_stub")
    require(
        xcodec_backend == "deterministic_stub" or is_module_callable(xcodec_backend),
        "xcodec.backend must be deterministic_stub or module:callable",
        errors,
    )
    require(int(xcodec.get("vocab_size", 0)) > 0, "xcodec.vocab_size must be positive", errors)
    rvq_axis = int(xcodec.get("rvq_axis", -1))
    require(rvq_axis in {-2, -1, 1, 2}, "xcodec.rvq_axis must select a non-batch dimension in a 3D RVQ tensor", errors)

    lm = config.get("lm") or {}
    hidden_size = int(lm.get("hidden_size", 0))
    num_heads = int(lm.get("num_attention_heads", 0))
    require(hidden_size > 0, "lm.hidden_size must be positive", errors)
    require(num_heads > 0, "lm.num_attention_heads must be positive", errors)
    if hidden_size > 0 and num_heads > 0:
        require(hidden_size % num_heads == 0, "lm.hidden_size must be divisible by lm.num_attention_heads", errors)

    refinement = config.get("refinement") or {}
    ref_channels = int(refinement.get("channels", 0))
    ref_heads = int(refinement.get("num_heads", 0))
    require(ref_channels > 0, "refinement.channels must be positive", errors)
    require(ref_heads > 0, "refinement.num_heads must be positive", errors)
    if ref_channels > 0 and ref_heads > 0:
        require(ref_channels % ref_heads == 0, "refinement.channels must be divisible by refinement.num_heads", errors)

    external_losses = config.get("external_losses") or {}
    for loss_name in ("pmsqe", "sqa"):
        loss_config = external_losses.get(loss_name) or {}
        if bool(loss_config.get("enabled", False)):
            require(
                is_module_callable(loss_config.get("import_path")),
                f"external_losses.{loss_name}.import_path must be module:callable when enabled",
                errors,
            )

    resampling = config.get("resampling") or {}
    require(
        resampling.get("backend", "linear") == "linear",
        "resampling.backend must be linear unless a verified backend is wired in code",
        errors,
    )
    require(
        config.get("lm_stft_alignment_mode", "interpolate") in {"interpolate", "strict"},
        "lm_stft_alignment_mode must be interpolate or strict",
        errors,
    )
    require(
        int(config.get("lm_stft_alignment_tolerance", 2)) >= 0,
        "lm_stft_alignment_tolerance must be non-negative",
        errors,
    )

    dataset_config = config.get("dataset_config") or {}
    for split in ("train_kwargs", "val_kwargs", "test_kwargs"):
        kwargs = dataset_config.get(split)
        if kwargs is None:
            continue
        batch_format = kwargs.get("batch_format", "tuple")
        require(batch_format in {"tuple", "dict"}, f"dataset_config.{split}.batch_format must be tuple or dict", errors)
        dataset_type = kwargs.get("dataset_type")
        modes = kwargs.get("modes")
        if modes is not None:
            mode_values = modes if isinstance(modes, list) else [modes]
            invalid_modes = set(mode_values) - {"se", "tse", "rtse"}
            require(not invalid_modes, f"dataset_config.{split}.modes contains unsupported modes {sorted(invalid_modes)}", errors)
            if batch_format == "dict":
                require(
                    set(mode_values) == {"se"},
                    f"dataset_config.{split}.modes must be [se] for hybrid dict SR/USE batches",
                    errors,
                )
        elif batch_format == "dict" and split != "test_kwargs" and dataset_type is None:
            errors.append(f"dataset_config.{split}.modes must be [se] for hybrid dict SR/USE batches")
        if dataset_type is None and split != "test_kwargs":
            sample_rates = kwargs.get("sample_rates")
            if sample_rates is not None:
                sample_rate_values = sample_rates if isinstance(sample_rates, list) else [sample_rates]
                for sample_rate in sample_rate_values:
                    require(
                        int(sample_rate) in supported_sample_rates,
                        f"dataset_config.{split}.sample_rates contains unsupported rate {sample_rate}",
                        errors,
                    )
        if dataset_type == "use_simulation_rolling_cache":
            for required_key in ("dry_data_root", "use_simulation_root", "simulation_config", "cache_dir"):
                require(
                    required_key in kwargs,
                    f"dataset_config.{split}.{required_key} is required for use_simulation_rolling_cache",
                    errors,
                )
            require(
                batch_format == "dict",
                f"dataset_config.{split}.batch_format should be dict for hybrid rolling-cache batches",
                errors,
            )
        target_sample_rate = kwargs.get("target_sample_rate")
        if target_sample_rate is not None:
            require(
                int(target_sample_rate) in supported_sample_rates,
                f"dataset_config.{split}.target_sample_rate is not supported by SFI",
                errors,
            )
        for key, value in kwargs.items():
            if isinstance(value, str) and value.startswith("/path/to/"):
                errors.append(f"dataset_config.{split}.{key} still contains placeholder path {value!r}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Hybrid-UniSE YAML config without importing torch")
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()

    all_errors: list[str] = []
    for config_path in args.configs:
        with config_path.open("r") as handle:
            config = yaml.safe_load(handle)
        errors = validate_config(config, config_path)
        if errors:
            all_errors.extend(f"{config_path}: {error}" for error in errors)
        else:
            print(f"OK {config_path}")

    if all_errors:
        for error in all_errors:
            print(f"ERROR {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
