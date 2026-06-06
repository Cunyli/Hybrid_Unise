import math

import torch

from model.audio import SFIConfig, sfi_istft, sfi_stft, stft_params
from dataloader.data_module import make_sr_batch
from model.hybrid_fusion import blend_spectra
from model.hybrid_lm import HybridSemanticLM
from model.hybrid_model import (
    HybridUniSELightning,
    hybrid_architecture_config,
    validate_hybrid_architecture_metadata,
    validate_hybrid_checkpoint_metadata,
)
from model.hybrid_xcodec import XCodecFirstRVQTokenizer


class FakeRVQBackend:
    def __call__(self, clean_wav_16k, sample_rate=16000):
        _ = sample_rate
        tokens = torch.zeros(clean_wav_16k.size(0), 5, 3, dtype=torch.long, device=clean_wav_16k.device)
        tokens[:, :, 0] = torch.arange(5, device=clean_wav_16k.device)
        tokens[:, :, 1] = 99
        mask = torch.tensor([[True, True, True, False, False]], device=clean_wav_16k.device).expand(clean_wav_16k.size(0), -1)
        return {"tokens": tokens, "mask": mask}


class FakeRVQBackendWith3DMask:
    def __call__(self, clean_wav_16k, sample_rate=16000):
        _ = sample_rate
        tokens = torch.zeros(clean_wav_16k.size(0), 2, 4, dtype=torch.long, device=clean_wav_16k.device)
        tokens[:, 0, :] = torch.arange(4, device=clean_wav_16k.device)
        tokens[:, 1, :] = 99
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[:, 0, -1] = False
        return tokens, mask


class FakeInvalidRVQBackend:
    def __call__(self, clean_wav_16k, sample_rate=16000):
        _ = sample_rate
        return torch.tensor([[0, 999]], device=clean_wav_16k.device)


class FakeAllPaddingRVQBackend:
    def __call__(self, clean_wav_16k, sample_rate=16000):
        _ = sample_rate
        tokens = torch.zeros(clean_wav_16k.size(0), 3, dtype=torch.long, device=clean_wav_16k.device)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        return tokens, mask


class FakeConfigurableRVQBackend:
    def __init__(self, model_path=None, offset=0):
        self.model_path = model_path
        self.offset = int(offset)

    def __call__(self, clean_wav_16k, sample_rate=16000):
        _ = sample_rate
        tokens = torch.arange(4, device=clean_wav_16k.device).unsqueeze(0) + self.offset
        return {"tokens": tokens.expand(clean_wav_16k.size(0), -1)}


def test_sfi_params_and_round_trip_supported_sample_rates():
    config = SFIConfig(window_ms=20.0, hop_ms=10.0, supported_sample_rates=(8000, 16000, 24000, 32000, 48000))
    for sample_rate in config.supported_sample_rates:
        params = stft_params(sample_rate, config)
        assert params.win_length == round(sample_rate * 0.020)
        assert params.hop_length == round(sample_rate * 0.010)
        assert params.n_bins == params.n_fft // 2 + 1

        time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        wav = torch.sin(2 * math.pi * 440.0 * time).unsqueeze(0)
        spec, _ = sfi_stft(wav, sample_rate, config)
        recovered = sfi_istft(spec, sample_rate, config, length=wav.size(-1))
        assert recovered.shape == wav.shape
        assert torch.mean(torch.abs(recovered - wav)) < 1e-4


def test_sfi_rejects_mixed_sample_rate_batch():
    config = SFIConfig()
    wav = torch.zeros(2, 1600)
    sample_rate = torch.tensor([16000, 48000])
    try:
        sfi_stft(wav, sample_rate, config)
    except ValueError as exc:
        assert "one sample rate per batch" in str(exc)
    else:
        raise AssertionError("mixed-rate batch should fail fast")


def test_sfi_handles_shorter_than_window_waveform():
    config = SFIConfig(window_ms=20.0, hop_ms=10.0, supported_sample_rates=(16000,))
    wav = torch.randn(1, 80)
    spec, params = sfi_stft(wav, 16000, config)
    assert params.win_length == 320
    assert spec.size(1) == params.n_bins
    recovered = sfi_istft(spec, 16000, config, length=wav.size(-1))
    assert recovered.shape == wav.shape


def test_fusion_extreme_masks_return_expected_branch():
    disc = torch.randn(2, 17, 9, dtype=torch.complex64)
    gen = torch.randn(2, 17, 9, dtype=torch.complex64)

    assert torch.equal(blend_spectra(disc, gen, torch.ones_like(disc.real)), disc)
    assert torch.equal(blend_spectra(disc, gen, torch.zeros_like(disc.real)), gen)


def test_lm_teacher_forcing_returns_target_aligned_hidden_states():
    lm = HybridSemanticLM(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
    )
    prefix = torch.randn(2, 5, 16)
    targets = torch.randint(0, 32, (2, 7))
    output = lm(prefix, targets)

    assert output["logits"].shape == (2, 7, 32)
    assert output["targets"].shape == targets.shape
    assert output["hidden_states"].shape == (2, 7, 16)
    assert output["hidden_mask"].shape == (2, 7)
    assert output["hidden_mask"].all()
    assert output["loss"].ndim == 0


def test_lm_teacher_forcing_ignores_padding_tokens():
    lm = HybridSemanticLM(
        vocab_size=8,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
    )
    prefix = torch.randn(1, 3, 16)
    targets = torch.tensor([[1, 2, lm.pad_token_id, lm.pad_token_id]])
    mask = torch.tensor([[True, True, False, False]])
    output = lm(prefix, targets, target_mask=mask)

    assert output["logits"].shape == (1, 4, 8)
    assert torch.equal(output["hidden_mask"], mask)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["accuracy"])


def test_xcodec_selects_first_rvq_layer_and_preserves_padding_mask(monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    monkeypatch.setattr(hybrid_xcodec.importlib, "import_module", lambda _name: type("M", (), {"backend": FakeRVQBackend()})())
    tokenizer = XCodecFirstRVQTokenizer(vocab_size=128, backend="fake.module:backend", rvq_axis=-1)
    batch = tokenizer.encode_first_rvq_batch(torch.zeros(1, 1600))

    assert batch.tokens.shape == (1, 5)
    assert torch.equal(batch.tokens[0, :3], torch.tensor([0, 1, 2]))
    assert torch.equal(batch.mask, torch.tensor([[True, True, True, False, False]]))
    assert torch.equal(batch.tokens[0, 3:], torch.tensor([tokenizer.vocab_size + 1, tokenizer.vocab_size + 1]))


def test_xcodec_selects_first_rvq_for_3d_mask(monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    monkeypatch.setattr(
        hybrid_xcodec.importlib,
        "import_module",
        lambda _name: type("M", (), {"backend": FakeRVQBackendWith3DMask()})(),
    )
    tokenizer = XCodecFirstRVQTokenizer(vocab_size=128, backend="fake.module:backend", rvq_axis=1)
    batch = tokenizer.encode_first_rvq_batch(torch.zeros(1, 1600))

    assert torch.equal(batch.tokens, torch.tensor([[0, 1, 2, tokenizer.vocab_size + 1]]))
    assert torch.equal(batch.mask, torch.tensor([[True, True, True, False]]))


def test_xcodec_rejects_out_of_vocab_valid_tokens(monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    monkeypatch.setattr(
        hybrid_xcodec.importlib,
        "import_module",
        lambda _name: type("M", (), {"backend": FakeInvalidRVQBackend()})(),
    )
    tokenizer = XCodecFirstRVQTokenizer(vocab_size=8, backend="fake.module:backend")
    try:
        tokenizer.encode_first_rvq_batch(torch.zeros(1, 1600))
    except ValueError as exc:
        assert "outside [0, vocab_size)" in str(exc)
    else:
        raise AssertionError("out-of-vocab X-Codec tokens should fail fast")


def test_xcodec_rejects_all_padding_token_mask(monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    monkeypatch.setattr(
        hybrid_xcodec.importlib,
        "import_module",
        lambda _name: type("M", (), {"backend": FakeAllPaddingRVQBackend()})(),
    )
    tokenizer = XCodecFirstRVQTokenizer(vocab_size=8, backend="fake.module:backend")
    try:
        tokenizer.encode_first_rvq_batch(torch.zeros(1, 1600))
    except ValueError as exc:
        assert "at least one valid token" in str(exc)
    else:
        raise AssertionError("all-padding X-Codec masks should fail fast")


def test_hybrid_accepts_explicit_sr_dict_batch_contract():
    batch = {
        "mode": "se",
        "degraded_wav": torch.randn(1, 1600),
        "clean_wav": torch.randn(1, 1600),
        "sample_rate": torch.tensor([16000]),
        "length": torch.tensor([1600]),
        "utterance_id": ["utt"],
        "source_path": ["noisy.wav"],
        "clean_path": ["clean.wav"],
    }
    normalized = HybridUniSELightning._normalize_batch(batch)
    assert normalized["degraded_wav"].shape == (1, 1600)
    assert normalized["sample_rate"].item() == 16000
    assert normalized["utterance_id"] == ["utt"]
    assert normalized["clean_path"] == ["clean.wav"]


def test_make_sr_batch_dict_contract_and_tuple_compatibility():
    degraded = torch.randn(2, 160)
    clean = torch.randn(2, 160)
    sample_rate = torch.tensor([16000, 16000])
    length = torch.tensor([160, 160])
    names = ["a", "b"]

    batch = make_sr_batch(
        "se",
        degraded,
        clean,
        sample_rate,
        length,
        names,
        batch_format="dict",
        source_path=["a_noisy.wav", "b_noisy.wav"],
        clean_path=["a_clean.wav", "b_clean.wav"],
    )
    assert batch["degraded_wav"] is degraded
    assert batch["clean_wav"] is clean
    assert batch["sample_rate"] is sample_rate
    assert batch["utterance_id"] == names
    assert batch["source_path"][0] == "a_noisy.wav"

    tuple_batch = make_sr_batch("se", degraded, clean, sample_rate, length, names)
    assert tuple_batch[0] == "se"
    assert tuple_batch[1] is None
    assert tuple_batch[2] is degraded
    assert tuple_batch[3] is clean
    assert tuple_batch[4] is None
    assert tuple_batch[5] is sample_rate
    assert tuple_batch[6] is length
    assert tuple_batch[7] == names


def test_tiny_hybrid_forward_shapes_and_mask_range():
    model = HybridUniSELightning(
        {
            "model_type": "hybrid_unise",
            "stage": "fusion",
            "fusion_use_teacher_forcing": True,
            "sfi": {"supported_sample_rates": [16000], "window_ms": 20.0, "hop_ms": 10.0},
            "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
            "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
            "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
            "lm": {
                "hidden_size": 16,
                "num_layers": 1,
                "num_attention_heads": 4,
                "dropout": 0.0,
                "max_position_embeddings": 256,
            },
            "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
            "fusion": {"channels": 8},
            "mrstft_loss": {"fft_sizes": [64], "hop_ratio": 0.25},
            "opt": {"lr": 1e-3},
        }
    )
    wav = torch.randn(1, 3200)
    clean = torch.randn(1, 3200)
    output = model(wav, torch.tensor([16000]), clean_wav=clean)

    assert output.final_wav.shape == wav.shape
    assert output.length is None
    assert output.disc_wav.shape == wav.shape
    assert output.gen_wav.shape == wav.shape
    assert output.gen_wav_16k.shape == wav.shape
    assert output.gen_spec_16k is not None
    assert output.fusion_mask.min() >= 0
    assert output.fusion_mask.max() <= 1
    assert output.token_logits.shape[:2] == output.token_targets.shape
    assert output.lm_hidden_states.shape[:2] == output.token_targets.shape
    assert output.lm_hidden_mask.shape == output.token_targets.shape
    assert output.aligned_lm_hidden_states.shape[1] == output.gen_spec_16k.shape[-1]
    assert output.aligned_lm_hidden_mask.shape[1] == output.gen_spec_16k.shape[-1]


def test_lm_stft_alignment_check_fails_on_large_mismatch():
    model = HybridUniSELightning(
        {
            "model_type": "hybrid_unise",
            "stage": "disc",
            "sfi": {"supported_sample_rates": [16000]},
            "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
            "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
            "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
            "lm": {"hidden_size": 16, "num_layers": 1, "num_attention_heads": 4},
            "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
            "lm_stft_alignment_tolerance": 1,
        }
    )
    try:
        model._validate_lm_stft_alignment(torch.randn(1, 4, 16), stft_frames=8)
    except ValueError as exc:
        assert "not aligned" in str(exc)
    else:
        raise AssertionError("large LM/STFT mismatch should fail")


def test_lm_hidden_alignment_interpolates_to_stft_frames():
    model = HybridUniSELightning(
        {
            "model_type": "hybrid_unise",
            "stage": "disc",
            "sfi": {"supported_sample_rates": [16000]},
            "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
            "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
            "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
            "lm": {"hidden_size": 16, "num_layers": 1, "num_attention_heads": 4},
            "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
            "lm_stft_alignment_mode": "interpolate",
        }
    )
    hidden, mask = model._align_lm_hidden_to_stft(
        torch.randn(2, 3, 16),
        torch.ones(2, 3, dtype=torch.bool),
        stft_frames=7,
    )
    assert hidden.shape == (2, 7, 16)
    assert mask.shape == (2, 7)


def test_external_loss_requires_import_path_when_enabled():
    try:
        HybridUniSELightning(
            {
                "model_type": "hybrid_unise",
                "stage": "disc",
                "sfi": {"supported_sample_rates": [16000]},
                "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
                "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
                "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
                "lm": {"hidden_size": 16, "num_layers": 1, "num_attention_heads": 4},
                "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
                "external_losses": {"pmsqe": {"enabled": True}},
            }
        )
    except ValueError as exc:
        assert "PMSQE" in str(exc)
        assert "import_path" in str(exc)
    else:
        raise AssertionError("enabled PMSQE loss without import_path should fail")


def test_xcodec_backend_requires_module_callable_format():
    try:
        XCodecFirstRVQTokenizer(backend="not_a_valid_backend")
    except ValueError as exc:
        assert "module:callable" in str(exc)
    else:
        raise AssertionError("invalid X-Codec backend should fail")


def test_xcodec_backend_class_receives_model_path_and_kwargs(monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    monkeypatch.setattr(
        hybrid_xcodec.importlib,
        "import_module",
        lambda _name: type("M", (), {"Backend": FakeConfigurableRVQBackend})(),
    )
    tokenizer = XCodecFirstRVQTokenizer(
        vocab_size=16,
        backend="fake.module:Backend",
        model_path="/tmp/xcodec",
        backend_kwargs={"offset": 2},
    )
    batch = tokenizer.encode_first_rvq_batch(torch.zeros(1, 1600))

    assert tokenizer.backend_impl.model_path == "/tmp/xcodec"
    assert torch.equal(batch.tokens, torch.tensor([[2, 3, 4, 5]]))


def test_unknown_resampling_backend_fails_fast():
    try:
        HybridUniSELightning(
            {
                "model_type": "hybrid_unise",
                "stage": "disc",
                "sfi": {"supported_sample_rates": [16000]},
                "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
                "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
                "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
                "lm": {"hidden_size": 16, "num_layers": 1, "num_attention_heads": 4},
                "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
                "resampling": {"backend": "soxr"},
            }
        )
    except ValueError as exc:
        assert "resampling.backend='linear'" in str(exc)
    else:
        raise AssertionError("unknown resampling backend should fail")


def test_hybrid_checkpoint_stage_metadata_must_match():
    validate_hybrid_checkpoint_metadata({"hybrid_stage": "disc"}, "disc")
    try:
        validate_hybrid_checkpoint_metadata({"hybrid_stage": "gen"}, "fusion")
    except ValueError as exc:
        assert "gen" in str(exc)
        assert "fusion" in str(exc)
    else:
        raise AssertionError("mismatched hybrid checkpoint stage should fail")


def test_hybrid_checkpoint_architecture_metadata_must_match():
    config = {
        "sfi": {"supported_sample_rates": [16000]},
        "lm": {"hidden_size": 16},
    }
    architecture = hybrid_architecture_config(config)
    validate_hybrid_architecture_metadata({"hybrid_architecture_config": architecture}, architecture)
    mismatched = hybrid_architecture_config({"sfi": {"supported_sample_rates": [8000]}, "lm": {"hidden_size": 16}})
    try:
        validate_hybrid_architecture_metadata({"hybrid_architecture_config": mismatched}, architecture)
    except ValueError as exc:
        assert "architecture" in str(exc)
    else:
        raise AssertionError("mismatched hybrid architecture should fail")


def test_hybrid_architecture_metadata_ignores_training_only_fields():
    base = {
        "sfi": {"supported_sample_rates": [16000]},
        "lm": {"hidden_size": 16},
        "loss_weights": {"fusion": {"l1": 0.5}},
        "fusion_use_teacher_forcing": False,
    }
    changed_training_only = {
        "sfi": {"supported_sample_rates": [16000]},
        "lm": {"hidden_size": 16},
        "loss_weights": {"fusion": {"l1": 0.1}},
        "fusion_use_teacher_forcing": True,
    }

    assert hybrid_architecture_config(base) == hybrid_architecture_config(changed_training_only)


def test_hybrid_checkpoint_architecture_metadata_ignores_legacy_extra_training_fields():
    architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "lm": {"hidden_size": 16},
        }
    )
    legacy_checkpoint_architecture = dict(
        architecture,
        loss_weights={"fusion": {"l1": 0.5}},
        fusion_use_teacher_forcing=True,
    )

    validate_hybrid_architecture_metadata(
        {"hybrid_architecture_config": legacy_checkpoint_architecture},
        architecture,
    )


def test_stage_freezing_matches_training_plan():
    base_config = {
        "model_type": "hybrid_unise",
        "sfi": {"supported_sample_rates": [16000]},
        "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
        "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
        "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
        "lm": {"hidden_size": 16, "num_layers": 1, "num_attention_heads": 4},
        "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
        "fusion": {"channels": 8},
    }

    disc_model = HybridUniSELightning(dict(base_config, stage="disc"))
    assert any(parameter.requires_grad for parameter in disc_model.discriminative.parameters())
    assert not any(parameter.requires_grad for parameter in disc_model.lm.parameters())
    assert not any(parameter.requires_grad for parameter in disc_model.fusion.parameters())

    gen_model = HybridUniSELightning(dict(base_config, stage="gen"))
    assert not any(parameter.requires_grad for parameter in gen_model.discriminative.parameters())
    assert any(parameter.requires_grad for parameter in gen_model.conditioner.adapter.parameters())
    assert not any(parameter.requires_grad for parameter in gen_model.conditioner.encoder.parameters())
    assert any(parameter.requires_grad for parameter in gen_model.lm.parameters())
    assert any(parameter.requires_grad for parameter in gen_model.refinement.parameters())
    assert not any(parameter.requires_grad for parameter in gen_model.fusion.parameters())

    fusion_model = HybridUniSELightning(dict(base_config, stage="fusion"))
    assert not any(parameter.requires_grad for parameter in fusion_model.discriminative.parameters())
    assert not any(parameter.requires_grad for parameter in fusion_model.lm.parameters())
    assert not any(parameter.requires_grad for parameter in fusion_model.refinement.parameters())
    assert any(parameter.requires_grad for parameter in fusion_model.fusion.parameters())


def test_fusion_stage_defaults_to_no_teacher_forcing(monkeypatch):
    model = HybridUniSELightning(
        {
            "model_type": "hybrid_unise",
            "stage": "fusion",
            "sfi": {"supported_sample_rates": [16000]},
            "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
            "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
            "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
            "lm": {"hidden_size": 16, "num_layers": 1, "num_attention_heads": 4},
            "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
            "fusion": {"channels": 8},
        }
    )
    captured = {}

    def fake_disc(wav, sample_rate):
        spec = torch.zeros(wav.size(0), 161, 3, dtype=torch.complex64)
        return wav, spec, spec

    def fake_gen(degraded_wav, clean_wav, sample_rate, length=None, do_sample=False):
        _ = length
        captured["clean_wav"] = clean_wav
        spec = torch.zeros(degraded_wav.size(0), 161, 3, dtype=torch.complex64)
        return degraded_wav, spec, degraded_wav, spec, {
            "loss": None,
            "logits": None,
            "targets": torch.zeros(degraded_wav.size(0), 3, dtype=torch.long),
            "hidden_states": torch.zeros(degraded_wav.size(0), 3, 16),
            "hidden_mask": torch.ones(degraded_wav.size(0), 3, dtype=torch.bool),
        }

    monkeypatch.setattr(model, "_disc", fake_disc)
    monkeypatch.setattr(model, "_gen", fake_gen)
    model(torch.zeros(1, 320), torch.tensor([16000]), clean_wav=torch.ones(1, 320))
    assert captured["clean_wav"] is None


def test_disc_forward_default_does_not_run_generation(monkeypatch):
    model = HybridUniSELightning(
        {
            "model_type": "hybrid_unise",
            "stage": "disc",
            "sfi": {"supported_sample_rates": [16000]},
            "discriminative": {"embedding": 8, "lstm_hidden": 4, "num_blocks": 1},
            "wavlm": {"use_pretrained": False, "freeze": True, "feature_dim": 8},
            "xcodec": {"backend": "deterministic_stub", "vocab_size": 32},
            "lm": {"hidden_size": 16, "num_layers": 1, "num_attention_heads": 4},
            "refinement": {"channels": 8, "hidden": 4, "num_blocks": 1, "num_heads": 2},
            "fusion": {"channels": 8},
        }
    )

    def fail_gen(*_args, **_kwargs):
        raise AssertionError("_gen should not run for default disc forward")

    monkeypatch.setattr(model, "_gen", fail_gen)
    output = model(torch.zeros(1, 320), torch.tensor([16000]))
    assert output.gen_wav is None
    assert output.token_targets is None
