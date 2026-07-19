from pathlib import Path

import torch
import yaml

from model import build_model
from model.audio import SFIConfig, sfi_istft, sfi_stft
from model.hybrid_fusion import blend_spectra
from model.hybrid_lm import _semantic_loss_metrics
from model.hybrid_model import LM_OBJECTIVE_ALLOWED_KEYS as RUNTIME_OBJECTIVE_KEYS
from scripts.validate_hybrid_config import (
    LM_OBJECTIVE_ALLOWED_KEYS as VALIDATOR_OBJECTIVE_KEYS,
    validate_config,
)


def test_model_factory_is_hybrid_only():
    try:
        build_model({"model_type": "unise"})
    except ValueError as exc:
        assert "only supports model_type: hybrid_unise" in str(exc)
    else:
        raise AssertionError("non-hybrid model type should fail closed")


def test_lm_objective_schema_is_shared_and_rejects_typos():
    assert RUNTIME_OBJECTIVE_KEYS == VALIDATOR_OBJECTIVE_KEYS
    config_path = Path("conf/hybrid_unise_smoke.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["lm_objective"] = {
        "transition_predecessor_margin_weigth": 0.25,
    }

    errors = validate_config(config, config_path)

    assert any("lm_objective contains unsupported keys" in error for error in errors)


def test_sfi_round_trip_shape():
    config = SFIConfig(
        window_ms=20.0,
        hop_ms=10.0,
        supported_sample_rates=(16000,),
    )
    waveform = torch.randn(1, 3200)
    spectrum, _ = sfi_stft(waveform, 16000, config)
    recovered = sfi_istft(spectrum, 16000, config, length=waveform.size(-1))

    assert recovered.shape == waveform.shape
    assert torch.mean(torch.abs(recovered - waveform)) < 1.0e-4


def test_fusion_mask_extremes_select_the_expected_branch():
    discriminative = torch.randn(1, 4, 5, dtype=torch.complex64)
    generative = torch.randn(1, 4, 5, dtype=torch.complex64)

    assert torch.equal(
        blend_spectra(discriminative, generative, torch.ones(1, 4, 5)),
        discriminative,
    )
    assert torch.equal(
        blend_spectra(discriminative, generative, torch.zeros(1, 4, 5)),
        generative,
    )


def test_zero_predecessor_margin_weight_preserves_ce_objective():
    logits = torch.randn(1, 5, 4)
    targets = torch.tensor([[0, 0, 1, 1, 2]])
    target_mask = torch.ones_like(targets, dtype=torch.bool)

    metrics = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.0,
        ignore_index=99,
        transition_loss_weight=1.0,
        transition_predecessor_margin=3.0,
        transition_predecessor_margin_weight=0.0,
    )

    assert metrics["objective_loss"] is metrics["ce_objective_loss"]
