import argparse
import math
from pathlib import Path

import yaml


VALID_STAGES = {"disc", "gen", "fusion", "joint"}
VALID_COMPONENT_INIT_KEYS = {"discriminative", "disc", "generative", "gen", "fusion"}
LM_OBJECTIVE_ALLOWED_KEYS = {
    "history_embedding_dropout_prob",
    "history_corruption_replacement_fraction",
    "transition_stall_history_corruption",
    "prefix_only_aux_weight",
    "transition_ce_multiplier",
    "normalize_transition_weights_per_sample",
    "transition_predecessor_margin",
    "transition_predecessor_margin_weight",
}


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def is_module_callable(value) -> bool:
    return isinstance(value, str) and ":" in value and all(value.split(":", 1))


def finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def resolve_config_path(value, config_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path


def validate_config(config: dict, path: Path) -> list[str]:
    errors: list[str] = []
    config_dir = Path(config.get("_config_dir", path.parent)).expanduser()
    require(config.get("model_type") == "hybrid_unise", "model_type must be hybrid_unise", errors)
    require(config.get("stage") in VALID_STAGES, f"stage must be one of {sorted(VALID_STAGES)}", errors)
    if "max_steps" in config:
        require(int(config["max_steps"]) > 0, "max_steps must be positive when provided", errors)
    require(
        not (config.get("resume") and config.get("stage_init_checkpoint")),
        "resume and stage_init_checkpoint are mutually exclusive",
        errors,
    )
    allow_refinement_mask_variant = config.get(
        "stage_init_allow_refinement_mask_variant",
        False,
    )
    require(
        isinstance(allow_refinement_mask_variant, bool),
        "stage_init_allow_refinement_mask_variant must be a bool",
        errors,
    )
    require(
        allow_refinement_mask_variant is not True,
        "stage_init_allow_refinement_mask_variant is unsafe and unsupported",
        errors,
    )
    component_init_value = config.get("component_init_checkpoints")
    if component_init_value is None:
        component_init_checkpoints = {}
    elif isinstance(component_init_value, dict):
        component_init_checkpoints = component_init_value
    else:
        errors.append("component_init_checkpoints must be a mapping")
        component_init_checkpoints = {}
    require(
        not (config.get("resume") and component_init_checkpoints),
        "resume and component_init_checkpoints are mutually exclusive",
        errors,
    )
    require(
        not (config.get("stage_init_checkpoint") and component_init_checkpoints),
        "stage_init_checkpoint and component_init_checkpoints are mutually exclusive",
        errors,
    )
    stage_init_checkpoint = config.get("stage_init_checkpoint")
    if stage_init_checkpoint:
        if stage_init_checkpoint == "auto":
            stage_init_dir = config.get("stage_init_checkpoint_dir")
            if not stage_init_dir:
                errors.append("stage_init_checkpoint_dir is required when stage_init_checkpoint is 'auto'")
            else:
                stage_init_root = resolve_config_path(stage_init_dir, config_dir)
                if not stage_init_root.exists():
                    errors.append(f"stage_init_checkpoint_dir does not exist: {stage_init_dir!r}")
                else:
                    patterns = config.get("stage_init_checkpoint_patterns") or ["version_*/latest_*.ckpt"]
                    candidates = []
                    for pattern in patterns:
                        candidates.extend(stage_init_root.glob(str(pattern)))
                    if not any(candidate.is_file() for candidate in candidates):
                        errors.append(
                            "stage_init_checkpoint auto found no checkpoint candidates under "
                            f"{stage_init_dir!r} with patterns {patterns!r}"
                        )
        elif isinstance(stage_init_checkpoint, str) and stage_init_checkpoint.startswith("/path/to/"):
            errors.append(f"stage_init_checkpoint still contains placeholder path {stage_init_checkpoint!r}")
        else:
            stage_init_path = resolve_config_path(stage_init_checkpoint, config_dir)
            if not stage_init_path.is_file():
                errors.append(f"stage_init_checkpoint does not exist: {stage_init_checkpoint!r}")
    for component, checkpoint in component_init_checkpoints.items():
        if component not in VALID_COMPONENT_INIT_KEYS:
            errors.append(
                f"component_init_checkpoints contains unsupported key {component!r}; "
                f"expected one of {sorted(VALID_COMPONENT_INIT_KEYS)}"
            )
            continue
        checkpoint_path = resolve_config_path(checkpoint, config_dir)
        if not checkpoint_path.is_file():
            errors.append(f"component_init_checkpoints.{component} does not exist: {checkpoint!r}")

    sfi = config.get("sfi") or {}
    supported_sample_rates = sfi.get("supported_sample_rates") or []
    require(bool(supported_sample_rates), "sfi.supported_sample_rates must not be empty", errors)
    require(16000 in supported_sample_rates, "sfi.supported_sample_rates must include 16000 for the generative branch", errors)
    require(float(sfi.get("window_ms", 0.0)) > 0.0, "sfi.window_ms must be positive", errors)
    require(float(sfi.get("hop_ms", 0.0)) > 0.0, "sfi.hop_ms must be positive", errors)

    wavlm = config.get("wavlm") or {}
    wavlm_layer_mode = str(wavlm.get("layer_mode", "mean"))
    require(
        wavlm_layer_mode in {"mean", "last", "layer"},
        "wavlm.layer_mode must be mean, last, or layer",
        errors,
    )
    if wavlm_layer_mode == "layer":
        require("layer_index" in wavlm, "wavlm.layer_index is required when wavlm.layer_mode is layer", errors)

    xcodec = config.get("xcodec") or {}
    xcodec_target_type = str(xcodec.get("target_type", "xcodec_rvq"))
    require(
        xcodec_target_type in {"xcodec_rvq", "wavlm_kmeans_vq"},
        "xcodec.target_type must be xcodec_rvq or wavlm_kmeans_vq",
        errors,
    )
    xcodec_backend = xcodec.get("backend", "deterministic_stub")
    require(int(xcodec.get("vocab_size", 0)) > 0, "xcodec.vocab_size must be positive", errors)
    if xcodec_target_type == "xcodec_rvq":
        require(
            xcodec_backend == "deterministic_stub" or is_module_callable(xcodec_backend),
            "xcodec.backend must be deterministic_stub or module:callable",
            errors,
        )
        rvq_axis = int(xcodec.get("rvq_axis", -1))
        require(rvq_axis in {-2, -1, 1, 2}, "xcodec.rvq_axis must select a non-batch dimension in a 3D RVQ tensor", errors)
        require(int(xcodec.get("rvq_index", 0)) >= 0, "xcodec.rvq_index must be non-negative", errors)
    if xcodec_target_type == "wavlm_kmeans_vq":
        require("codebook_path" in xcodec, "xcodec.codebook_path is required for wavlm_kmeans_vq", errors)
        require(
            "wavlm_model_path" in xcodec or "pretrained_name_or_path" in wavlm,
            "xcodec.wavlm_model_path or wavlm.pretrained_name_or_path is required for wavlm_kmeans_vq",
            errors,
        )
        require(
            str(xcodec.get("feature_normalization", "standardize_l2")) in {"none", "l2", "standardize_l2"},
            "xcodec.feature_normalization must be none, l2, or standardize_l2",
            errors,
        )
        require(int(xcodec.get("layer_index", 11)) >= -128, "xcodec.layer_index is out of expected range", errors)
        require(int(xcodec.get("codec_hop_length", xcodec.get("hop_length", 320))) > 0, "xcodec.codec_hop_length must be positive", errors)

    lm = config.get("lm") or {}
    lm_architecture = lm.get("architecture", "transformer_encoder")
    require(
        lm_architecture in {"transformer_encoder", "llama"},
        "lm.architecture must be transformer_encoder or llama",
        errors,
    )
    hidden_size = int(lm.get("hidden_size", 0))
    num_heads = int(lm.get("num_attention_heads", 0))
    require(hidden_size > 0, "lm.hidden_size must be positive", errors)
    require(num_heads > 0, "lm.num_attention_heads must be positive", errors)
    if hidden_size > 0 and num_heads > 0:
        require(hidden_size % num_heads == 0, "lm.hidden_size must be divisible by lm.num_attention_heads", errors)
    label_smoothing_value = lm.get("label_smoothing", 0.0)
    label_smoothing = (
        None
        if isinstance(label_smoothing_value, bool)
        else finite_float(label_smoothing_value)
    )
    require(
        label_smoothing is not None and 0.0 <= label_smoothing <= 1.0,
        "lm.label_smoothing must be finite and in [0, 1]",
        errors,
    )

    lm_objective_value = config.get("lm_objective", {})
    if lm_objective_value is None:
        lm_objective_value = {}
    if not isinstance(lm_objective_value, dict):
        errors.append("lm_objective must be a mapping")
        lm_objective = {}
    else:
        lm_objective = lm_objective_value
        unknown_lm_objective_keys = sorted(
            set(lm_objective) - LM_OBJECTIVE_ALLOWED_KEYS
        )
        if unknown_lm_objective_keys:
            errors.append(
                "lm_objective contains unsupported keys "
                f"{unknown_lm_objective_keys}; expected only "
                f"{sorted(LM_OBJECTIVE_ALLOWED_KEYS)}"
            )
    history_dropout = finite_float(
        lm_objective.get("history_embedding_dropout_prob", 0.0)
    )
    history_replacement_fraction = finite_float(
        lm_objective.get("history_corruption_replacement_fraction", 0.0)
    )
    transition_stall_history_corruption = lm_objective.get(
        "transition_stall_history_corruption",
        False,
    )
    prefix_only_aux_weight = finite_float(
        lm_objective.get("prefix_only_aux_weight", 0.0)
    )
    transition_ce_multiplier = finite_float(
        lm_objective.get("transition_ce_multiplier", 1.0)
    )
    transition_predecessor_margin_value = lm_objective.get(
        "transition_predecessor_margin",
        1.0,
    )
    transition_predecessor_margin_weight_value = lm_objective.get(
        "transition_predecessor_margin_weight",
        0.0,
    )
    transition_predecessor_margin = (
        None
        if isinstance(transition_predecessor_margin_value, bool)
        else finite_float(transition_predecessor_margin_value)
    )
    transition_predecessor_margin_weight = (
        None
        if isinstance(transition_predecessor_margin_weight_value, bool)
        else finite_float(transition_predecessor_margin_weight_value)
    )
    normalize_transition_weights_per_sample = lm_objective.get(
        "normalize_transition_weights_per_sample",
        False,
    )
    require(
        history_dropout is not None and 0.0 <= history_dropout < 1.0,
        "lm_objective.history_embedding_dropout_prob must be finite and in [0, 1)",
        errors,
    )
    require(
        history_replacement_fraction is not None
        and 0.0 <= history_replacement_fraction <= 1.0,
        "lm_objective.history_corruption_replacement_fraction must be finite "
        "and in [0, 1]",
        errors,
    )
    require(
        history_replacement_fraction is None
        or history_replacement_fraction == 0.0
        or (history_dropout is not None and history_dropout > 0.0),
        "lm_objective.history_corruption_replacement_fraction requires a "
        "positive history_embedding_dropout_prob",
        errors,
    )
    require(
        isinstance(transition_stall_history_corruption, bool),
        "lm_objective.transition_stall_history_corruption must be a bool",
        errors,
    )
    require(
        transition_stall_history_corruption is not True
        or (
            history_replacement_fraction is not None
            and history_replacement_fraction > 0.0
        ),
        "lm_objective.transition_stall_history_corruption requires a positive "
        "history_corruption_replacement_fraction",
        errors,
    )
    require(
        prefix_only_aux_weight is not None and prefix_only_aux_weight >= 0.0,
        "lm_objective.prefix_only_aux_weight must be finite and non-negative",
        errors,
    )
    require(
        transition_ce_multiplier is not None and transition_ce_multiplier >= 1.0,
        "lm_objective.transition_ce_multiplier must be finite and at least 1.0",
        errors,
    )
    require(
        transition_predecessor_margin is not None
        and transition_predecessor_margin >= 0.0,
        "lm_objective.transition_predecessor_margin must be finite and non-negative",
        errors,
    )
    require(
        transition_predecessor_margin_weight is not None
        and transition_predecessor_margin_weight >= 0.0,
        "lm_objective.transition_predecessor_margin_weight must be finite "
        "and non-negative",
        errors,
    )
    require(
        isinstance(normalize_transition_weights_per_sample, bool),
        "lm_objective.normalize_transition_weights_per_sample must be a bool",
        errors,
    )
    lm_objective_enabled = (
        (history_dropout is not None and history_dropout > 0.0)
        or (
            history_replacement_fraction is not None
            and history_replacement_fraction > 0.0
        )
        or transition_stall_history_corruption is True
        or (prefix_only_aux_weight is not None and prefix_only_aux_weight > 0.0)
        or (
            transition_ce_multiplier is not None
            and transition_ce_multiplier != 1.0
        )
        or normalize_transition_weights_per_sample is True
        or (
            transition_predecessor_margin_weight is not None
            and transition_predecessor_margin_weight > 0.0
        )
    )
    require(
        not lm_objective_enabled or config.get("stage") == "gen",
        "A non-default lm_objective is currently supported only for stage=gen",
        errors,
    )

    refinement = config.get("refinement") or {}
    refinement_architecture = refinement.get("architecture", "legacy")
    require(
        refinement_architecture in {"legacy", "paper_dprnn", "paper_dprnn_identity_mask"},
        "refinement.architecture must be legacy, paper_dprnn, or paper_dprnn_identity_mask",
        errors,
    )
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

    gen_weights = (config.get("loss_weights") or {}).get("gen", {})
    gen_refinement_has_loss = (
        float(gen_weights.get("complex", 0.1)) != 0.0
        or float(gen_weights.get("mag", 0.9)) != 0.0
        or (
            float(gen_weights.get("pmsqe", 0.01)) != 0.0
            and bool((external_losses.get("pmsqe") or {}).get("enabled", False))
        )
    )
    if config.get("stage") == "gen" and bool(config.get("gen_train_refinement", False)):
        require(
            gen_refinement_has_loss,
            "gen_train_refinement=true requires a non-zero gen complex, mag, or enabled pmsqe loss",
            errors,
        )
    if (
        config.get("stage") == "gen"
        and history_dropout is not None
        and history_dropout > 0.0
    ):
        require(
            not gen_refinement_has_loss,
            "lm_objective.history_embedding_dropout_prob must be 0 when generative waveform losses are enabled",
            errors,
        )

    resampling = config.get("resampling") or {}
    require(
        resampling.get("backend", "linear") == "linear",
        "resampling.backend must be linear unless a verified backend is wired in code",
        errors,
    )
    require(
        config.get("lm_stft_alignment_mode", "interpolate") in {"interpolate", "strict", "raw_cross_attn"},
        "lm_stft_alignment_mode must be interpolate, strict, or raw_cross_attn",
        errors,
    )
    require(
        int(config.get("lm_stft_alignment_tolerance", 2)) >= 0,
        "lm_stft_alignment_tolerance must be non-negative",
        errors,
    )

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
