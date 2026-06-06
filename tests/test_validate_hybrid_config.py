from scripts.validate_hybrid_config import validate_config


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


def test_validate_hybrid_config_accepts_minimal_valid_config(tmp_path):
    assert validate_config(base_config(), tmp_path / "valid.yaml") == []


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
