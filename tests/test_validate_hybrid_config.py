from model.hybrid_objective import (
    LM_OBJECTIVE_ALLOWED_KEYS as RUNTIME_LM_OBJECTIVE_ALLOWED_KEYS,
)
from scripts.validate_hybrid_config import (
    LM_OBJECTIVE_ALLOWED_KEYS as VALIDATOR_LM_OBJECTIVE_ALLOWED_KEYS,
    validate_config,
)


def base_config():
    return {
        "model_type": "hybrid_unise",
        "stage": "disc",
        "sfi": {
            "window_ms": 20.0,
            "hop_ms": 10.0,
            "supported_sample_rates": [16000],
        },
        "xcodec": {
            "backend": "deterministic_stub",
            "vocab_size": 32,
        },
        "lm": {
            "hidden_size": 16,
            "num_attention_heads": 4,
        },
        "refinement": {
            "channels": 8,
            "num_heads": 2,
        },
        "external_losses": {
            "pmsqe": {"enabled": False},
            "sqa": {"enabled": False},
        },
        "dataset_config": {
            "train_kwargs": {
                "batch_format": "dict",
                "modes": ["se"],
                "sample_rates": [16000],
            },
        },
    }


def test_lm_objective_allowlists_match_runtime():
    assert (
        VALIDATOR_LM_OBJECTIVE_ALLOWED_KEYS
        == RUNTIME_LM_OBJECTIVE_ALLOWED_KEYS
    )


def test_validate_hybrid_config_accepts_minimal_valid_config(tmp_path):
    assert validate_config(base_config(), tmp_path / "valid.yaml") == []


def test_validate_hybrid_config_accepts_gen_lm_objective_without_waveform_loss(tmp_path):
    config = base_config()
    config["stage"] = "gen"
    config["loss_weights"] = {
        "gen": {"nll": 1.0, "complex": 0.0, "mag": 0.0, "pmsqe": 0.0}
    }
    config["lm_objective"] = {
        "history_embedding_dropout_prob": 0.5,
        "history_corruption_replacement_fraction": 0.5,
        "transition_stall_history_corruption": True,
        "prefix_only_aux_weight": 1.0,
        "transition_ce_multiplier": 3.0,
        "normalize_transition_weights_per_sample": True,
        "transition_predecessor_margin": 1.5,
        "transition_predecessor_margin_weight": 0.25,
    }

    assert validate_config(config, tmp_path / "valid.yaml") == []


def test_validate_hybrid_config_rejects_non_mapping_lm_objective(tmp_path):
    config = base_config()
    config["lm_objective"] = []

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert "lm_objective must be a mapping" in errors


def test_validate_hybrid_config_rejects_unknown_lm_objective_key(tmp_path):
    config = base_config()
    config["lm_objective"] = {
        "transition_predecessor_margin_weigth": 0.25,
    }

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any(
        "lm_objective contains unsupported keys" in error
        and "transition_predecessor_margin_weigth" in error
        for error in errors
    )


def test_validate_hybrid_config_can_skip_only_external_path_checks(tmp_path):
    config = base_config()
    config["stage_init_checkpoint"] = str(tmp_path / "missing.ckpt")

    checked_errors = validate_config(config, tmp_path / "checked.yaml")
    static_errors = validate_config(
        config,
        tmp_path / "static.yaml",
        check_external_paths=False,
    )

    assert any("stage_init_checkpoint does not exist" in error for error in checked_errors)
    assert static_errors == []


def test_validate_hybrid_config_rejects_invalid_lm_objective_values(tmp_path):
    invalid_values = {
        "history_embedding_dropout_prob": (-0.1, 1.0, float("nan")),
        "history_corruption_replacement_fraction": (
            -0.1,
            1.1,
            float("nan"),
        ),
        "transition_stall_history_corruption": (0, 1, "true", None, []),
        "prefix_only_aux_weight": (-0.1, float("inf"), "bad"),
        "transition_ce_multiplier": (0.5, float("nan"), "bad"),
        "normalize_transition_weights_per_sample": (0, 1, "true", None, []),
        "transition_predecessor_margin": (
            -0.1,
            float("nan"),
            "bad",
            True,
        ),
        "transition_predecessor_margin_weight": (
            -0.1,
            float("inf"),
            "bad",
            False,
        ),
    }
    for field, values in invalid_values.items():
        for value in values:
            config = base_config()
            config["lm_objective"] = {field: value}

            errors = validate_config(config, tmp_path / "invalid.yaml")

            assert any(f"lm_objective.{field}" in error for error in errors)


def test_validate_hybrid_config_requires_dropout_for_history_replacement(tmp_path):
    config = base_config()
    config["stage"] = "gen"
    config["lm_objective"] = {
        "history_embedding_dropout_prob": 0.0,
        "history_corruption_replacement_fraction": 0.5,
    }

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("requires a positive history_embedding_dropout_prob" in error for error in errors)


def test_validate_hybrid_config_requires_replacement_for_transition_stall(tmp_path):
    config = base_config()
    config["stage"] = "gen"
    config["lm_objective"] = {"transition_stall_history_corruption": True}

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any(
        "requires a positive history_corruption_replacement_fraction" in error
        for error in errors
    )


def test_validate_hybrid_config_rejects_transition_stall_outside_gen(tmp_path):
    config = base_config()
    config["lm_objective"] = {
        "history_embedding_dropout_prob": 0.5,
        "history_corruption_replacement_fraction": 0.5,
        "transition_stall_history_corruption": True,
    }

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("supported only for stage=gen" in error for error in errors)


def test_validate_hybrid_config_rejects_non_default_lm_objective_outside_gen(tmp_path):
    config = base_config()
    config["lm_objective"] = {"transition_ce_multiplier": 3.0}

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("supported only for stage=gen" in error for error in errors)


def test_validate_hybrid_config_allows_metric_only_margin_outside_gen(tmp_path):
    config = base_config()
    config["lm_objective"] = {
        "transition_predecessor_margin": 2.0,
        "transition_predecessor_margin_weight": 0.0,
    }

    assert validate_config(config, tmp_path / "valid.yaml") == []


def test_validate_hybrid_config_rejects_weighted_margin_outside_gen(tmp_path):
    config = base_config()
    config["lm_objective"] = {
        "transition_predecessor_margin_weight": 0.5,
    }

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("supported only for stage=gen" in error for error in errors)


def test_validate_hybrid_config_allows_weighted_margin_with_waveform_loss(tmp_path):
    config = base_config()
    config["stage"] = "gen"
    config["loss_weights"] = {
        "gen": {"nll": 1.0, "complex": 0.1, "mag": 0.0, "pmsqe": 0.0}
    }
    config["lm_objective"] = {
        "transition_predecessor_margin_weight": 0.5,
    }

    assert validate_config(config, tmp_path / "valid.yaml") == []


def test_validate_hybrid_config_rejects_invalid_label_smoothing(tmp_path):
    for value in (True, -0.1, 1.1, float("nan"), "bad"):
        config = base_config()
        config["lm"]["label_smoothing"] = value

        errors = validate_config(config, tmp_path / "invalid.yaml")

        assert "lm.label_smoothing must be finite and in [0, 1]" in errors


def test_validate_hybrid_config_rejects_history_dropout_with_waveform_loss(tmp_path):
    config = base_config()
    config["stage"] = "gen"
    config["gen_train_refinement"] = False
    config["loss_weights"] = {
        "gen": {"nll": 1.0, "complex": 0.1, "mag": 0.0, "pmsqe": 0.0}
    }
    config["lm_objective"] = {"history_embedding_dropout_prob": 0.5}

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("generative waveform losses" in error for error in errors)


def test_validate_hybrid_config_requires_positive_explicit_max_steps(tmp_path):
    valid = base_config()
    valid["max_steps"] = 50
    assert validate_config(valid, tmp_path / "valid.yaml") == []

    invalid = base_config()
    invalid["max_steps"] = 0
    errors = validate_config(invalid, tmp_path / "invalid.yaml")
    assert "max_steps must be positive when provided" in errors


def test_validate_hybrid_config_rejects_non_mapping_component_init(tmp_path):
    config = base_config()
    config["component_init_checkpoints"] = ["bad.ckpt"]

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert "component_init_checkpoints must be a mapping" in errors


def test_validate_hybrid_config_rejects_zero_loss_trainable_refinement(tmp_path):
    config = base_config()
    config["stage"] = "gen"
    config["gen_train_refinement"] = True
    config["loss_weights"] = {
        "gen": {"nll": 1.0, "complex": 0.0, "mag": 0.0, "pmsqe": 0.0}
    }

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("gen_train_refinement=true" in error for error in errors)


def test_validate_hybrid_config_accepts_paper_aligned_gen_architecture(tmp_path):
    config = base_config()
    config["lm"]["architecture"] = "llama"
    config["refinement"]["architecture"] = "paper_dprnn"
    config["lm_stft_alignment_mode"] = "raw_cross_attn"
    config["wavlm"] = {"layer_mode": "layer", "layer_index": 11}
    config["xcodec"]["rvq_index"] = 7

    assert validate_config(config, tmp_path / "valid.yaml") == []


def test_validate_hybrid_config_accepts_wavlm_kmeans_vq_target(tmp_path):
    config = base_config()
    config["wavlm"] = {"pretrained_name_or_path": "./pretrained/wavlm/microsoft_wavlm-base-plus"}
    config["xcodec"] = {
        "target_type": "wavlm_kmeans_vq",
        "vocab_size": 128,
        "codebook_path": "./runs/semantic_vq_probe/outputs/codebook.npz",
        "layer_index": 11,
        "feature_normalization": "standardize_l2",
        "codec_hop_length": 320,
    }

    assert validate_config(config, tmp_path / "valid.yaml") == []


def test_validate_hybrid_config_accepts_identity_centered_paper_dprnn(tmp_path):
    config = base_config()
    config["lm"]["architecture"] = "llama"
    config["refinement"]["architecture"] = "paper_dprnn_identity_mask"
    config["lm_stft_alignment_mode"] = "raw_cross_attn"

    assert validate_config(config, tmp_path / "valid.yaml") == []


def test_validate_hybrid_config_rejects_bad_lm_architecture(tmp_path):
    config = base_config()
    config["lm"]["architecture"] = "rnn"
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("lm.architecture" in error for error in errors)


def test_validate_hybrid_config_rejects_bad_wavlm_layer_mode(tmp_path):
    config = base_config()
    config["wavlm"] = {"layer_mode": "middle"}

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("wavlm.layer_mode" in error for error in errors)


def test_validate_hybrid_config_rejects_missing_wavlm_layer_index(tmp_path):
    config = base_config()
    config["wavlm"] = {"layer_mode": "layer"}

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("wavlm.layer_index" in error for error in errors)


def test_validate_hybrid_config_rejects_bad_rvq_index(tmp_path):
    config = base_config()
    config["xcodec"]["rvq_index"] = -1

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("xcodec.rvq_index" in error for error in errors)


def test_validate_hybrid_config_rejects_bad_xcodec_target_type(tmp_path):
    config = base_config()
    config["xcodec"]["target_type"] = "wavtokenizer"

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("xcodec.target_type" in error for error in errors)


def test_validate_hybrid_config_rejects_missing_wavlm_kmeans_codebook(tmp_path):
    config = base_config()
    config["wavlm"] = {"pretrained_name_or_path": "./pretrained/wavlm/microsoft_wavlm-base-plus"}
    config["xcodec"] = {
        "target_type": "wavlm_kmeans_vq",
        "vocab_size": 128,
    }

    errors = validate_config(config, tmp_path / "invalid.yaml")

    assert any("xcodec.codebook_path" in error for error in errors)


def test_validate_hybrid_config_rejects_bad_refinement_architecture(tmp_path):
    config = base_config()
    config["refinement"]["architecture"] = "flat_lstm"
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("refinement.architecture" in error for error in errors)


def test_validate_hybrid_config_rejects_bad_external_loss(tmp_path):
    config = base_config()
    config["external_losses"]["pmsqe"] = {"enabled": True}
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("external_losses.pmsqe.import_path" in error for error in errors)


def test_validate_hybrid_config_rejects_unsupported_sample_rate(tmp_path):
    config = base_config()
    config["dataset_config"]["train_kwargs"]["sample_rates"] = [8000]
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("unsupported rate 8000" in error for error in errors)


def test_validate_hybrid_config_rejects_unwired_resampling_backend(tmp_path):
    config = base_config()
    config["resampling"] = {"backend": "soxr"}
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("resampling.backend" in error for error in errors)


def test_validate_hybrid_config_rejects_bad_alignment_mode(tmp_path):
    config = base_config()
    config["lm_stft_alignment_mode"] = "nearest"
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("lm_stft_alignment_mode" in error for error in errors)


def test_validate_hybrid_config_rejects_resume_and_stage_init_together(tmp_path):
    config = base_config()
    config["resume"] = "auto"
    config["stage_init_checkpoint"] = "disc.ckpt"
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("stage_init_checkpoint" in error for error in errors)


def test_validate_hybrid_config_rejects_unsafe_refinement_mask_variant_flag(
    tmp_path,
):
    for value in (True, "false", 1):
        config = base_config()
        config["stage_init_allow_refinement_mask_variant"] = value

        errors = validate_config(config, tmp_path / "bad.yaml")

        assert any(
            "stage_init_allow_refinement_mask_variant" in error
            for error in errors
        )


def test_validate_hybrid_config_rejects_missing_stage_init_checkpoint(tmp_path):
    config = base_config()
    config["stage_init_checkpoint"] = str(tmp_path / "missing.ckpt")
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("stage_init_checkpoint does not exist" in error for error in errors)


def test_validate_hybrid_config_resolves_stage_init_relative_to_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    checkpoint = config_dir / "disc.ckpt"
    checkpoint.write_bytes(b"placeholder")
    config = base_config()
    config["_config_dir"] = str(config_dir)
    config["stage_init_checkpoint"] = "disc.ckpt"

    assert validate_config(config, tmp_path / "generated.yaml") == []


def test_validate_hybrid_config_rejects_resume_and_component_init_together(tmp_path):
    config = base_config()
    config["resume"] = "auto"
    config["component_init_checkpoints"] = {"discriminative": str(tmp_path / "disc.ckpt")}
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("component_init_checkpoints" in error and "resume" in error for error in errors)


def test_validate_hybrid_config_rejects_stage_init_and_component_init_together(tmp_path):
    config = base_config()
    config["stage_init_checkpoint"] = str(tmp_path / "disc.ckpt")
    config["component_init_checkpoints"] = {"generative": str(tmp_path / "gen.ckpt")}
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("component_init_checkpoints" in error and "stage_init_checkpoint" in error for error in errors)


def test_validate_hybrid_config_rejects_missing_component_init_checkpoint(tmp_path):
    config = base_config()
    config["component_init_checkpoints"] = {"generative": str(tmp_path / "missing.ckpt")}
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("component_init_checkpoints.generative does not exist" in error for error in errors)


def test_validate_hybrid_config_resolves_component_init_relative_to_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    checkpoint = config_dir / "gen.ckpt"
    checkpoint.write_bytes(b"placeholder")
    config = base_config()
    config["_config_dir"] = str(config_dir)
    config["component_init_checkpoints"] = {"generative": "gen.ckpt"}

    assert validate_config(config, tmp_path / "generated.yaml") == []


def test_validate_hybrid_config_rejects_unknown_component_init_key(tmp_path):
    config = base_config()
    config["component_init_checkpoints"] = {"encoder": str(tmp_path / "model.ckpt")}
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("unsupported key 'encoder'" in error for error in errors)


def test_validate_hybrid_config_rejects_incomplete_webdataset_stream(tmp_path):
    config = base_config()
    config["dataset_config"]["train_kwargs"] = {
        "dataset_type": "hybrid_unise_webdataset_stream",
        "batch_format": "tuple",
        "simulation_config": "./conf/simulation_train.yaml",
    }
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("split_root is required" in error for error in errors)


def test_validate_hybrid_config_rejects_incomplete_fixed_recipe(tmp_path):
    config = base_config()
    config["dataset_config"]["val_kwargs"] = {
        "dataset_type": "hybrid_unise_webdataset_fixed_recipe",
        "batch_format": "tuple",
        "recipe_manifest": "valid/fixed_recipes.jsonl",
    }
    errors = validate_config(config, tmp_path / "bad.yaml")
    assert any("simulation_config is required" in error for error in errors)
