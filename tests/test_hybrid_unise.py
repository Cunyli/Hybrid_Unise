import io
import math
import random
import tarfile

import numpy as np
import soundfile as sf
import torch

from dataloader.hybrid_webdataset_protocol import AudioDecodeError, WebDatasetAudioReader
from dataloader.simulation.rir_utils import direct_path_delay, get_rir_start_sample, shift_by_delay
from dataloader.simulation.simulate import simulate_data
from model.xcodec_backends import TransformersXCodecFirstRVQ
from model.audio import SFIConfig, sfi_istft, sfi_stft, stft_params
from dataloader.data_module import DataModule, make_sr_batch
from model.hybrid_fusion import blend_spectra
from model.hybrid_lm import (
    HybridLlamaSemanticLM,
    HybridSemanticLM,
    WavLMConditioner,
    _semantic_loss_metrics,
    _semantic_token_embeddings,
)
from model.hybrid_model import (
    HybridUniSELightning,
    hybrid_architecture_config,
    validate_hybrid_architecture_metadata,
    validate_hybrid_checkpoint_metadata,
)
from model.hybrid_refinement import IdentityCenteredPaperDPRNNRefinementBranch
from model.hybrid_xcodec import WavLMKMeansVQTokenizer, XCodecFirstRVQTokenizer


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


class FakeWavLMEncoder(torch.nn.Module):
    _CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
    _CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        assert add_adapter in {None, False}
        output_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
        for kernel_size, stride in zip(self._CONV_KERNEL, self._CONV_STRIDE):
            output_lengths = torch.div(
                output_lengths - kernel_size,
                stride,
                rounding_mode="floor",
            ) + 1
        return output_lengths

    def __init__(self):
        super().__init__()
        self.config = type("FakeWavLMConfig", (), {"hidden_size": 4})()
        self.last_attention_mask = None

    def forward(self, wav_16k, attention_mask=None, output_hidden_states=True):
        assert output_hidden_states
        self.last_attention_mask = attention_mask
        batch = wav_16k.size(0)
        frame_count = int(
            self._get_feat_extract_output_lengths(
                torch.tensor([wav_16k.size(-1)]),
                add_adapter=False,
            ).item()
        )
        hidden_states = tuple(
            torch.full(
                (batch, frame_count, 4),
                float(layer),
                device=wav_16k.device,
            )
            for layer in range(4)
        )
        return type("FakeWavLMOutput", (), {"hidden_states": hidden_states})()


def tiny_hybrid_config(stage="disc", **overrides):
    config = {
        "model_type": "hybrid_unise",
        "stage": stage,
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
    config.update(overrides)
    return config


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


def test_identity_centered_paper_dprnn_zero_decoder_preserves_input_spectrum():
    branch = IdentityCenteredPaperDPRNNRefinementBranch(
        channels=4,
        hidden=4,
        lm_hidden=8,
        num_heads=2,
        num_blocks=1,
        dprnn_window_frames=2,
        dprnn_hop_frames=1,
    )
    degraded = torch.randn(1, 5, 7, dtype=torch.complex64)
    lm_hidden = torch.randn(1, 4, 8)

    enhanced = branch(degraded, lm_hidden)

    assert torch.allclose(enhanced, degraded)
    output_head = branch.decoder[-1]
    assert torch.equal(output_head.weight, torch.zeros_like(output_head.weight))
    assert torch.equal(output_head.bias, torch.zeros_like(output_head.bias))

    enhanced.abs().mean().backward()
    assert output_head.weight.grad is not None
    assert torch.isfinite(output_head.weight.grad).all()
    assert torch.count_nonzero(output_head.weight.grad) > 0


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


def test_llama_lm_teacher_forcing_returns_target_aligned_hidden_states():
    lm = HybridLlamaSemanticLM(
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


def test_llama_teacher_forcing_is_causally_shifted():
    torch.manual_seed(7)
    lm = HybridLlamaSemanticLM(
        vocab_size=16,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
    ).eval()
    prefix = torch.randn(1, 4, 16)
    targets = torch.tensor([[1, 2, 3, 4, 5]])
    changed_targets = targets.clone()
    changed_targets[0, 3] = 9

    baseline = lm(prefix, targets)
    changed = lm(prefix, changed_targets)

    assert torch.equal(baseline["logits"][:, :4], changed["logits"][:, :4])
    assert torch.max(torch.abs(baseline["logits"][:, 4:] - changed["logits"][:, 4:])) > 0


def test_llama_generate_matches_teacher_forcing_on_its_own_greedy_tokens():
    torch.manual_seed(7)
    lm = HybridLlamaSemanticLM(
        vocab_size=16,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
    ).eval()
    prefix = torch.randn(1, 4, 16)

    generated = lm.generate(prefix, max_tokens=5, do_sample=False)
    replayed = lm(prefix, generated["tokens"])

    assert torch.equal(generated["tokens"], replayed["logits"].argmax(dim=-1))
    assert torch.equal(generated["hidden_states"], replayed["hidden_states"])
    assert torch.equal(generated["hidden_mask"], replayed["hidden_mask"])


def test_llama_logits_and_loss_depend_on_acoustic_prefix():
    torch.manual_seed(7)
    lm = HybridLlamaSemanticLM(
        vocab_size=16,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
    ).eval()
    prefix = torch.randn(1, 4, 16, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4, 5]])

    output = lm(prefix, targets)
    zero_prefix_output = lm(torch.zeros_like(prefix), targets)
    output["loss"].backward()

    assert not torch.equal(output["logits"], zero_prefix_output["logits"])
    assert prefix.grad is not None
    assert torch.isfinite(prefix.grad).all()
    assert prefix.grad.abs().sum() > 0


def test_lms_ignore_padded_prefix_frames_and_compact_token_positions():
    for lm_class in (HybridSemanticLM, HybridLlamaSemanticLM):
        torch.manual_seed(7)
        lm = lm_class(
            vocab_size=16,
            hidden_size=16,
            num_layers=1,
            num_attention_heads=4,
            dropout=0.0,
            max_position_embeddings=64,
        ).eval()
        valid_prefix = torch.randn(1, 2, 16)
        padded_prefix = torch.cat([valid_prefix, torch.randn(1, 2, 16)], dim=1)
        padded_prefix_mask = torch.tensor([[True, True, False, False]])
        targets = torch.tensor([[1, 2, 3, 4]])

        cropped_output = lm(valid_prefix, targets)
        padded_output = lm(
            padded_prefix,
            targets,
            prefix_mask=padded_prefix_mask,
        )
        cropped_generated = lm.generate(valid_prefix, max_tokens=4)
        padded_generated = lm.generate(
            padded_prefix,
            max_tokens=4,
            prefix_mask=padded_prefix_mask,
        )

        torch.testing.assert_close(
            cropped_output["logits"],
            padded_output["logits"],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            cropped_output["hidden_states"],
            padded_output["hidden_states"],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        assert torch.equal(
            cropped_generated["tokens"],
            padded_generated["tokens"],
        )
        torch.testing.assert_close(
            cropped_generated["hidden_states"],
            padded_generated["hidden_states"],
            atol=1.0e-5,
            rtol=1.0e-5,
        )


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


def test_semantic_loss_metrics_weight_transitions_without_crossing_padding():
    logits = torch.zeros(1, 6, 4)
    logits[0, 0, 0] = 3.0
    logits[0, 1, 1] = 3.0
    logits[0, 2, 1] = 2.0
    logits[0, 3, 1] = 2.0
    logits[0, 4, 0] = 4.0
    targets = torch.tensor([[0, 0, 1, 1, 2, 3]])
    target_mask = torch.tensor([[True, True, True, True, True, False]])

    metrics = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.0,
        ignore_index=99,
        transition_loss_weight=3.0,
    )
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    expected_nll = token_losses[0, :5].mean()
    expected_objective = (
        token_losses[0, 0]
        + token_losses[0, 1]
        + 3.0 * token_losses[0, 2]
        + token_losses[0, 3]
        + 3.0 * token_losses[0, 4]
    ) / 9.0

    torch.testing.assert_close(metrics["loss"], expected_nll)
    torch.testing.assert_close(metrics["objective_loss"], expected_objective)
    torch.testing.assert_close(metrics["transition_fraction"], torch.tensor(0.4))
    torch.testing.assert_close(metrics["initial_accuracy"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["repeat_accuracy"], torch.tensor(0.5))
    torch.testing.assert_close(metrics["transition_accuracy"], torch.tensor(0.5))


def test_semantic_loss_metrics_normalizes_transition_weights_per_sample_with_padding():
    torch.manual_seed(41)
    logits = torch.randn(2, 5, 4)
    targets = torch.tensor(
        [
            [0, 0, 1, 1, 2],
            [1, 2, 2, 3, 3],
        ]
    )
    target_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    metrics = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.0,
        ignore_index=99,
        transition_loss_weight=3.0,
        normalize_transition_weights_per_sample=True,
    )
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    raw_weights = torch.tensor(
        [
            [1.0, 1.0, 3.0, 1.0, 3.0],
            [1.0, 3.0, 1.0, 0.0, 0.0],
        ]
    )
    per_sample_scale = torch.tensor([[5.0 / 9.0], [3.0 / 5.0]])
    expected_objective = (
        token_losses * raw_weights * per_sample_scale
    ).sum() / 8.0

    torch.testing.assert_close(metrics["objective_loss"], expected_objective)
    torch.testing.assert_close(metrics["transition_fraction"], torch.tensor(3.0 / 8.0))


def test_semantic_loss_metrics_per_sample_normalization_removes_transition_fraction_bias():
    targets = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
        ]
    )
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    logits = torch.zeros(2, 4, 2)
    logits[0, :, 0] = 3.0
    for index, target in enumerate(targets[1]):
        logits[1, index, target] = -1.0

    legacy = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.0,
        ignore_index=99,
        transition_loss_weight=4.0,
    )
    normalized = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.0,
        ignore_index=99,
        transition_loss_weight=4.0,
        normalize_transition_weights_per_sample=True,
    )

    torch.testing.assert_close(normalized["objective_loss"], normalized["loss"])
    assert legacy["objective_loss"] > normalized["objective_loss"]


def test_semantic_loss_metrics_per_sample_normalization_preserves_single_sample_objective():
    torch.manual_seed(42)
    logits = torch.randn(1, 6, 4)
    targets = torch.tensor([[0, 0, 1, 1, 2, 3]])
    target_mask = torch.tensor([[True, True, True, True, True, False]])

    legacy = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.1,
        ignore_index=99,
        transition_loss_weight=3.0,
    )
    normalized = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.1,
        ignore_index=99,
        transition_loss_weight=3.0,
        normalize_transition_weights_per_sample=True,
    )

    torch.testing.assert_close(
        normalized["objective_loss"],
        legacy["objective_loss"],
    )


def test_semantic_loss_metrics_normalized_objective_has_finite_gradients():
    torch.manual_seed(43)
    logits = torch.randn(2, 4, 3, requires_grad=True)
    targets = torch.tensor([[0, 1, 1, 2], [2, 1, 0, 0]])
    target_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )

    metrics = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        label_smoothing=0.1,
        ignore_index=99,
        transition_loss_weight=3.0,
        normalize_transition_weights_per_sample=True,
    )
    metrics["objective_loss"].backward()

    assert torch.isfinite(metrics["objective_loss"])
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[1, 2:], torch.zeros_like(logits.grad[1, 2:]))


def test_semantic_loss_metrics_normalization_is_exactly_legacy_when_disabled_or_unit_weight():
    torch.manual_seed(47)
    logits = torch.randn(2, 5, 4)
    targets = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 2, 3, 0]])
    target_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    common = {
        "label_smoothing": 0.1,
        "ignore_index": 99,
    }

    default_legacy = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        transition_loss_weight=3.0,
        **common,
    )
    explicit_legacy = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        transition_loss_weight=3.0,
        normalize_transition_weights_per_sample=False,
        **common,
    )
    unit_weight_legacy = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        transition_loss_weight=1.0,
        **common,
    )
    unit_weight_normalized = _semantic_loss_metrics(
        logits,
        targets,
        target_mask,
        transition_loss_weight=1.0,
        normalize_transition_weights_per_sample=True,
        **common,
    )

    for key in default_legacy:
        assert torch.equal(default_legacy[key], explicit_legacy[key])
        assert torch.equal(unit_weight_legacy[key], unit_weight_normalized[key])


def test_semantic_loss_metrics_rejects_non_contiguous_target_mask():
    try:
        _semantic_loss_metrics(
            torch.zeros(1, 3, 4),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[True, False, True]]),
            label_smoothing=0.0,
            ignore_index=99,
            transition_loss_weight=1.0,
        )
    except ValueError as exc:
        assert "contiguous right padding" in str(exc)
    else:
        raise AssertionError("A non-contiguous target mask should fail fast")


def test_history_embedding_dropout_preserves_sos_and_padding_without_rescaling():
    embedding = torch.nn.Embedding(8, 4)
    torch.nn.init.ones_(embedding.weight)
    token_input = torch.zeros(8, 33, dtype=torch.long)
    token_input_mask = torch.ones_like(token_input, dtype=torch.bool)
    token_input_mask[:, -4:] = False

    rng_before = torch.get_rng_state().clone()
    eval_embeddings = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=False,
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(eval_embeddings, torch.ones_like(eval_embeddings))

    torch.manual_seed(17)
    dropped = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
    )
    assert torch.equal(dropped[:, 0], torch.ones_like(dropped[:, 0]))
    assert torch.equal(dropped[~token_input_mask], torch.ones_like(dropped[~token_input_mask]))
    valid_history = dropped[:, 1:-4]
    assert (valid_history == 0.0).all(dim=-1).any()
    assert (valid_history == 1.0).all(dim=-1).any()

    zero_history = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=True,
        history_embedding_dropout_prob=0.5,
        training=True,
    )
    assert torch.equal(zero_history[:, 0], torch.ones_like(zero_history[:, 0]))
    assert torch.equal(zero_history[:, 1:], torch.zeros_like(zero_history[:, 1:]))


def test_zero_replacement_fraction_preserves_dropout_output_and_rng_state():
    embedding = torch.nn.Embedding(10, 3)
    token_input = torch.tensor([[8, 0, 1, 2, 3, 9], [8, 4, 5, 3, 9, 9]])
    token_input_mask = torch.tensor(
        [[True, True, True, True, True, False], [True, True, True, True, False, False]]
    )

    torch.manual_seed(29)
    implicit_zero = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
    )
    implicit_rng_state = torch.get_rng_state().clone()

    torch.manual_seed(29)
    explicit_zero = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
        history_corruption_replacement_fraction=0.0,
        transition_stall_history_corruption=False,
    )

    assert torch.equal(explicit_zero, implicit_zero)
    assert torch.equal(torch.get_rng_state(), implicit_rng_state)

    torch.manual_seed(29)
    _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
        history_corruption_replacement_fraction=1.0,
        semantic_vocab_size=8,
    )
    assert torch.equal(torch.get_rng_state(), implicit_rng_state)


def test_history_corruption_replaces_with_different_non_special_tokens():
    embedding = torch.nn.Embedding(10, 1)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(1, 11, dtype=torch.float32).view(-1, 1))
    token_input = torch.tensor(
        [[8, 0, 1, 2, 3, 9], [8, 4, 5, 3, 9, 9]],
        dtype=torch.long,
    )
    token_input_mask = torch.tensor(
        [[True, True, True, True, True, False], [True, True, True, True, False, False]]
    )
    original_embeddings = embedding(token_input)

    torch.manual_seed(31)
    corrupted = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.9,
        training=True,
        history_corruption_replacement_fraction=1.0,
        semantic_vocab_size=8,
    )

    assert torch.equal(corrupted[:, 0], original_embeddings[:, 0])
    assert torch.equal(corrupted[~token_input_mask], original_embeddings[~token_input_mask])
    changed = (corrupted != original_embeddings).squeeze(-1)
    assert changed.any()
    assert not changed[:, 0].any()
    assert not changed[~token_input_mask].any()
    replacement_eligible = torch.zeros_like(token_input_mask)
    replacement_eligible[:, 1:-1] = token_input_mask[:, 2:]
    replaced = changed & replacement_eligible
    assert replaced.any()
    replacement_tokens = corrupted.squeeze(-1)[replaced].long() - 1
    assert ((0 <= replacement_tokens) & (replacement_tokens < 8)).all()
    assert (replacement_tokens != token_input[replaced]).all()
    terminal_history = token_input_mask & ~replacement_eligible
    terminal_history[:, 0] = False
    terminal_values = corrupted.squeeze(-1)[terminal_history]
    terminal_original = original_embeddings.squeeze(-1)[terminal_history]
    assert ((terminal_values == 0.0) | (terminal_values == terminal_original)).all()


def test_transition_stall_history_corruption_reuses_selected_replacement_mask(
    monkeypatch,
):
    embedding = torch.nn.Embedding(10, 1)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(1, 11, dtype=torch.float32).view(-1, 1))
    token_input = torch.tensor([[8, 2, 3, 3, 5, 6]], dtype=torch.long)
    token_input_mask = torch.ones_like(token_input, dtype=torch.bool)
    original_embeddings = embedding(token_input)
    rand_calls = []

    def fixed_rand(size, *, device):
        rand_calls.append((size, device))
        return torch.full((size,), 0.25, device=device)

    monkeypatch.setattr(torch, "rand", fixed_rand)

    uniform_corruption = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
        history_corruption_replacement_fraction=1.0,
        transition_stall_history_corruption=False,
        semantic_vocab_size=8,
    )

    transition_stall = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
        history_corruption_replacement_fraction=1.0,
        transition_stall_history_corruption=True,
        semantic_vocab_size=8,
    )

    assert rand_calls == [(5, token_input.device), (5, token_input.device)]
    assert transition_stall.dtype == embedding.weight.dtype
    assert transition_stall.device == token_input.device
    assert torch.equal(transition_stall[:, 0], original_embeddings[:, 0])
    assert torch.equal(transition_stall[:, -1], uniform_corruption[:, -1])
    assert transition_stall[0, -1].item() == 0.0
    assert transition_stall[0, 2].item() == original_embeddings[0, 1].item()
    assert transition_stall[0, 4].item() == original_embeddings[0, 3].item()
    assert transition_stall[0, 3].item() != original_embeddings[0, 3].item()
    assert transition_stall[0, 1].item() == uniform_corruption[0, 1].item()
    assert transition_stall[0, 3].item() == uniform_corruption[0, 3].item()


def test_transition_stall_history_corruption_preserves_eval_and_padding(monkeypatch):
    embedding = torch.nn.Embedding(10, 2)
    token_input = torch.tensor(
        [[8, 1, 2, 3, 4, 5], [8, 2, 4, 9, 9, 9]],
        dtype=torch.long,
    )
    token_input_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, False, False, False]]
    )
    original_embeddings = embedding(token_input)

    rng_before = torch.get_rng_state().clone()
    eval_embeddings = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=False,
        history_corruption_replacement_fraction=0.5,
        transition_stall_history_corruption=True,
        semantic_vocab_size=8,
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(eval_embeddings, original_embeddings)

    rand_calls = []

    def fixed_rand(size, *, device):
        rand_calls.append((size, device))
        return torch.full((size,), 0.25, device=device)

    monkeypatch.setattr(torch, "rand", fixed_rand)
    uniform_train_embeddings = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
        history_corruption_replacement_fraction=1.0,
        transition_stall_history_corruption=False,
        semantic_vocab_size=8,
    )
    train_embeddings = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.5,
        training=True,
        history_corruption_replacement_fraction=1.0,
        transition_stall_history_corruption=True,
        semantic_vocab_size=8,
    )
    assert torch.equal(
        train_embeddings[~token_input_mask],
        original_embeddings[~token_input_mask],
    )
    assert torch.equal(train_embeddings[1, 2], uniform_train_embeddings[1, 2])
    assert torch.equal(
        train_embeddings[~token_input_mask],
        uniform_train_embeddings[~token_input_mask],
    )
    assert rand_calls == [(7, token_input.device), (7, token_input.device)]


def test_transition_stall_history_corruption_does_not_steal_drop_mask(monkeypatch):
    embedding = torch.nn.Embedding(10, 1)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(1, 11, dtype=torch.float32).view(-1, 1))
    token_input = torch.tensor([[8, 1, 2, 3, 4, 5]], dtype=torch.long)
    token_input_mask = torch.ones_like(token_input, dtype=torch.bool)
    original_embeddings = embedding(token_input)

    def fixed_rand(size, *, device):
        assert size == 5
        return torch.tensor(
            [0.2, 0.2, 0.6, 0.9, 0.6],
            device=device,
        )

    monkeypatch.setattr(torch, "rand", fixed_rand)
    corrupted = _semantic_token_embeddings(
        embedding,
        token_input,
        token_input_mask=token_input_mask,
        zero_history=False,
        history_embedding_dropout_prob=0.8,
        training=True,
        history_corruption_replacement_fraction=0.5,
        transition_stall_history_corruption=True,
        semantic_vocab_size=8,
    )

    assert corrupted[0, 2].item() == original_embeddings[0, 1].item()
    assert corrupted[0, 3].item() == 0.0
    assert corrupted[0, 4].item() == original_embeddings[0, 4].item()
    assert corrupted[0, 5].item() == 0.0


def test_transition_stall_history_corruption_is_wired_through_both_lms(monkeypatch):
    def fixed_rand(size, *, device):
        return torch.full((size,), 0.25, device=device)

    monkeypatch.setattr(torch, "rand", fixed_rand)
    for lm_class in (HybridSemanticLM, HybridLlamaSemanticLM):
        torch.manual_seed(71)
        lm = lm_class(
            vocab_size=16,
            hidden_size=16,
            num_layers=1,
            num_attention_heads=4,
            dropout=0.0,
            max_position_embeddings=64,
        ).train()
        prefix = torch.randn(1, 3, 16)
        targets = torch.tensor([[1, 1, 4, 4]], dtype=torch.long)
        forward_rng_state = torch.get_rng_state().clone()

        uniform = lm(
            prefix,
            targets,
            history_embedding_dropout_prob=0.5,
            history_corruption_replacement_fraction=1.0,
            transition_stall_history_corruption=False,
        )
        uniform_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(forward_rng_state)
        transition_stall = lm(
            prefix,
            targets,
            history_embedding_dropout_prob=0.5,
            history_corruption_replacement_fraction=1.0,
            transition_stall_history_corruption=True,
        )

        assert torch.equal(torch.get_rng_state(), uniform_rng_state)
        assert transition_stall["logits"].shape == uniform["logits"].shape
        assert torch.isfinite(transition_stall["logits"]).all()
        assert not torch.equal(
            transition_stall["logits"][:, 3],
            uniform["logits"][:, 3],
        )


def test_transition_stall_history_corruption_requires_replacement_budget():
    embedding = torch.nn.Embedding(10, 2)
    try:
        _semantic_token_embeddings(
            embedding,
            torch.tensor([[8, 1, 2]], dtype=torch.long),
            token_input_mask=torch.ones(1, 3, dtype=torch.bool),
            zero_history=False,
            history_embedding_dropout_prob=0.5,
            training=True,
            history_corruption_replacement_fraction=0.0,
            transition_stall_history_corruption=True,
            semantic_vocab_size=8,
        )
    except ValueError as exc:
        assert "requires a positive history_corruption_replacement_fraction" in str(exc)
    else:
        raise AssertionError("Transition-stall corruption without replacement must fail fast")


def test_lm_objective_defaults_and_eval_dropout_do_not_consume_rng():
    for lm_class in (HybridSemanticLM, HybridLlamaSemanticLM):
        torch.manual_seed(23)
        lm = lm_class(
            vocab_size=16,
            hidden_size=16,
            num_layers=1,
            num_attention_heads=4,
            dropout=0.0,
            max_position_embeddings=64,
        ).eval()
        prefix = torch.randn(2, 4, 16)
        targets = torch.tensor([[1, 1, 2, 2], [3, 4, 4, 5]])

        rng_before = torch.get_rng_state().clone()
        baseline = lm(prefix, targets)
        explicit_legacy = lm(
            prefix,
            targets,
            transition_loss_weight=3.0,
            normalize_transition_weights_per_sample=False,
        )
        normalized = lm(
            prefix,
            targets,
            transition_loss_weight=3.0,
            normalize_transition_weights_per_sample=True,
        )
        explicit_eval_dropout = lm(
            prefix,
            targets,
            history_embedding_dropout_prob=0.5,
            history_corruption_replacement_fraction=0.5,
        )

        assert torch.equal(torch.get_rng_state(), rng_before)
        assert torch.equal(baseline["logits"], explicit_eval_dropout["logits"])
        assert torch.equal(baseline["loss"], baseline["objective_loss"])
        assert torch.isfinite(baseline["transition_accuracy"])
        expected_normalized = _semantic_loss_metrics(
            normalized["logits"],
            normalized["targets"],
            normalized["hidden_mask"],
            label_smoothing=lm.label_smoothing,
            ignore_index=lm.pad_token_id,
            transition_loss_weight=3.0,
            normalize_transition_weights_per_sample=True,
        )
        torch.testing.assert_close(
            normalized["objective_loss"],
            expected_normalized["objective_loss"],
        )
        assert not torch.equal(
            explicit_legacy["objective_loss"],
            normalized["objective_loss"],
        )


def test_gen_lm_objective_is_separate_from_architecture_and_drives_loss(monkeypatch):
    loss_weights = {
        "gen": {"nll": 1.0, "complex": 0.0, "mag": 0.0, "pmsqe": 0.0}
    }
    objective = {
        "history_embedding_dropout_prob": 0.5,
        "history_corruption_replacement_fraction": 0.5,
        "transition_stall_history_corruption": True,
        "prefix_only_aux_weight": 1.0,
        "transition_ce_multiplier": 3.0,
        "normalize_transition_weights_per_sample": True,
    }
    baseline_config = tiny_hybrid_config("gen", loss_weights=loss_weights)
    objective_config = tiny_hybrid_config(
        "gen",
        loss_weights=loss_weights,
        lm_objective=objective,
    )

    assert hybrid_architecture_config(baseline_config) == hybrid_architecture_config(
        objective_config
    )
    model = HybridUniSELightning(objective_config).train()
    assert model.lm_normalize_transition_weights_per_sample is True
    assert model.lm_transition_stall_history_corruption is True
    lm_calls = []
    original_lm_forward = model.lm.forward

    def record_lm_forward(*args, **kwargs):
        lm_calls.append(kwargs.copy())
        return original_lm_forward(*args, **kwargs)

    monkeypatch.setattr(model.lm, "forward", record_lm_forward)
    wav = torch.randn(1, 1600)
    clean = torch.randn(1, 1600)
    output = model(
        wav,
        torch.tensor([16000]),
        clean_wav=clean,
        length=torch.tensor([1600]),
    )
    losses = model._losses(output, clean, torch.tensor([16000]))

    assert len(lm_calls) == 2
    assert lm_calls[0]["normalize_transition_weights_per_sample"] is True
    assert lm_calls[0]["transition_stall_history_corruption"] is True
    assert lm_calls[1]["normalize_transition_weights_per_sample"] is True
    assert "transition_stall_history_corruption" not in lm_calls[1]
    assert lm_calls[1]["zero_history"] is True
    assert output.token_objective_nll is not None
    assert output.token_prefix_only_nll is not None
    assert output.token_prefix_only_weighted_nll is not None
    assert output.token_transition_accuracy is not None
    assert losses["lm_objective"] is output.token_objective_nll
    expected_objective = (
        output.token_weighted_nll + output.token_prefix_only_weighted_nll
    ) / 2.0
    torch.testing.assert_close(output.token_objective_nll, expected_objective)
    training_loss = model._weighted_stage_loss(losses)
    torch.testing.assert_close(training_loss, losses["lm_objective"])
    training_loss.backward()
    assert model.lm.output_head.weight.grad is not None
    assert torch.isfinite(model.lm.output_head.weight.grad).all()
    assert model.lm.embedding.weight.grad is not None
    assert torch.isfinite(model.lm.embedding.weight.grad).all()
    assert model.conditioner.adapter.weight.grad is not None
    assert torch.isfinite(model.conditioner.adapter.weight.grad).all()


def test_gen_transition_predecessor_margin_wires_main_and_prefix_objectives(
    monkeypatch,
):
    config = tiny_hybrid_config(
        "gen",
        loss_weights={
            "gen": {"nll": 1.0, "complex": 0.0, "mag": 0.0, "pmsqe": 0.0}
        },
        lm_objective={
            "prefix_only_aux_weight": 1.0,
            "transition_predecessor_margin": 100.0,
            "transition_predecessor_margin_weight": 0.25,
        },
    )
    model = HybridUniSELightning(config).train()
    lm_calls = []
    original_lm_forward = model.lm.forward

    def record_lm_forward(*args, **kwargs):
        result = original_lm_forward(*args, **kwargs)
        lm_calls.append((kwargs.copy(), result))
        return result

    monkeypatch.setattr(model.lm, "forward", record_lm_forward)
    wav = torch.randn(1, 1600)
    clean = torch.randn(1, 1600)
    output = model(
        wav,
        torch.tensor([16000]),
        clean_wav=clean,
        length=torch.tensor([1600]),
    )
    losses = model._losses(output, clean, torch.tensor([16000]))

    assert len(lm_calls) == 2
    for kwargs, result in lm_calls:
        assert kwargs["transition_predecessor_margin"] == 100.0
        assert kwargs["transition_predecessor_margin_weight"] == 0.25
        assert result["objective_loss"] > result["ce_objective_loss"]
    main_result = lm_calls[0][1]
    prefix_result = lm_calls[1][1]
    assert lm_calls[1][0]["zero_history"] is True
    torch.testing.assert_close(
        output.token_weighted_nll,
        main_result["ce_objective_loss"],
    )
    torch.testing.assert_close(
        output.token_prefix_only_weighted_nll,
        prefix_result["ce_objective_loss"],
    )
    torch.testing.assert_close(
        output.token_objective_nll,
        (main_result["objective_loss"] + prefix_result["objective_loss"]) / 2.0,
    )
    assert (
        output.token_transition_predecessor_margin_loss
        is main_result["transition_predecessor_margin_loss"]
    )
    assert (
        output.token_prefix_only_transition_predecessor_margin_loss
        is prefix_result["transition_predecessor_margin_loss"]
    )
    assert (
        losses["transition_predecessor_rate"]
        is output.token_transition_predecessor_rate
    )
    assert (
        losses["prefix_only_transition_predecessor_rate"]
        is output.token_prefix_only_transition_predecessor_rate
    )


def test_gen_transition_predecessor_margin_is_metric_only_in_eval(monkeypatch):
    config = tiny_hybrid_config(
        "gen",
        loss_weights={
            "gen": {"nll": 1.0, "complex": 0.0, "mag": 0.0, "pmsqe": 0.0}
        },
        lm_objective={
            "transition_predecessor_margin": 2.0,
            "transition_predecessor_margin_weight": 0.5,
        },
    )
    model = HybridUniSELightning(config).eval()
    lm_calls = []
    original_lm_forward = model.lm.forward

    def record_lm_forward(*args, **kwargs):
        result = original_lm_forward(*args, **kwargs)
        lm_calls.append((kwargs.copy(), result))
        return result

    monkeypatch.setattr(model.lm, "forward", record_lm_forward)
    wav = torch.randn(1, 1600)
    clean = torch.randn(1, 1600)
    output = model(
        wav,
        torch.tensor([16000]),
        clean_wav=clean,
        length=torch.tensor([1600]),
    )

    assert len(lm_calls) == 1
    kwargs, result = lm_calls[0]
    assert kwargs["transition_predecessor_margin"] == 2.0
    assert kwargs["transition_predecessor_margin_weight"] == 0.0
    assert result["objective_loss"] is result["ce_objective_loss"]
    assert output.token_objective_nll is None
    assert output.token_weighted_nll is result["ce_objective_loss"]


def test_gen_nll_weight_scales_full_margin_training_objective():
    model = HybridUniSELightning(
        tiny_hybrid_config(
            "gen",
            loss_weights={
                "gen": {
                    "nll": 2.0,
                    "complex": 0.0,
                    "mag": 0.0,
                    "pmsqe": 0.0,
                }
            },
            lm_objective={"transition_predecessor_margin_weight": 0.25},
        )
    )
    training_objective = torch.tensor(3.0)
    losses = {
        "lm_objective": training_objective,
        "complex": torch.tensor(0.0),
        "mag": torch.tensor(0.0),
    }

    torch.testing.assert_close(
        model._weighted_stage_loss(losses),
        2.0 * training_objective,
    )


def test_hybrid_model_rejects_invalid_transition_predecessor_margin_values():
    invalid_values = {
        "transition_predecessor_margin": (True, -0.1, float("nan")),
        "transition_predecessor_margin_weight": (False, -0.1, float("inf")),
    }
    for field, values in invalid_values.items():
        for value in values:
            config = tiny_hybrid_config("gen", lm_objective={field: value})
            try:
                HybridUniSELightning(config)
            except ValueError as exc:
                assert field in str(exc)
            else:
                raise AssertionError(f"Invalid {field}={value!r} should fail fast")


def test_hybrid_model_rejects_invalid_label_smoothing():
    for value in (True, -0.1, 1.1, float("nan")):
        config = tiny_hybrid_config("gen")
        config["lm"]["label_smoothing"] = value
        try:
            HybridUniSELightning(config)
        except ValueError as exc:
            assert "lm.label_smoothing" in str(exc)
        else:
            raise AssertionError(
                f"Invalid lm.label_smoothing={value!r} should fail fast"
            )


def test_history_dropout_rejects_enabled_generative_waveform_losses():
    config = tiny_hybrid_config(
        "gen",
        loss_weights={
            "gen": {"nll": 1.0, "complex": 0.1, "mag": 0.0, "pmsqe": 0.0}
        },
        gen_train_refinement=False,
        lm_objective={"history_embedding_dropout_prob": 0.5},
    )

    try:
        HybridUniSELightning(config)
    except ValueError as exc:
        assert "generative waveform losses" in str(exc)
    else:
        raise AssertionError("History dropout with waveform loss should fail fast")


def test_hybrid_model_rejects_non_bool_transition_weight_normalization():
    config = tiny_hybrid_config(
        "gen",
        lm_objective={"normalize_transition_weights_per_sample": 1},
    )

    try:
        HybridUniSELightning(config)
    except ValueError as exc:
        assert "normalize_transition_weights_per_sample must be a bool" in str(exc)
    else:
        raise AssertionError("Non-bool transition weight normalization should fail fast")


def test_hybrid_model_rejects_invalid_transition_stall_history_corruption():
    invalid_type = tiny_hybrid_config(
        "gen",
        lm_objective={"transition_stall_history_corruption": 1},
    )
    try:
        HybridUniSELightning(invalid_type)
    except ValueError as exc:
        assert "transition_stall_history_corruption must be a bool" in str(exc)
    else:
        raise AssertionError("Non-bool transition-stall corruption should fail fast")

    missing_replacement = tiny_hybrid_config(
        "gen",
        lm_objective={"transition_stall_history_corruption": True},
    )
    try:
        HybridUniSELightning(missing_replacement)
    except ValueError as exc:
        assert "requires a positive history_corruption_replacement_fraction" in str(exc)
    else:
        raise AssertionError("Transition-stall corruption without replacement should fail fast")


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


def test_xcodec_selects_configured_rvq_index(monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    monkeypatch.setattr(hybrid_xcodec.importlib, "import_module", lambda _name: type("M", (), {"backend": FakeRVQBackend()})())
    tokenizer = XCodecFirstRVQTokenizer(vocab_size=128, backend="fake.module:backend", rvq_axis=-1, rvq_index=1)
    batch = tokenizer.encode_first_rvq_batch(torch.zeros(1, 1600))

    assert torch.equal(batch.tokens[0, :3], torch.tensor([99, 99, 99]))
    assert torch.equal(batch.mask, torch.tensor([[True, True, True, False, False]]))


def test_xcodec_selects_configured_rvq_index_for_3d_mask(monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    monkeypatch.setattr(
        hybrid_xcodec.importlib,
        "import_module",
        lambda _name: type("M", (), {"backend": FakeRVQBackendWith3DMask()})(),
    )
    tokenizer = XCodecFirstRVQTokenizer(vocab_size=128, backend="fake.module:backend", rvq_axis=1, rvq_index=1)
    batch = tokenizer.encode_first_rvq_batch(torch.zeros(1, 1600))

    assert torch.equal(batch.tokens, torch.tensor([[99, 99, 99, 99]]))
    assert torch.equal(batch.mask, torch.tensor([[True, True, True, True]]))


def test_wavlm_conditioner_selects_configured_hidden_layer():
    conditioner = WavLMConditioner(output_dim=4, use_pretrained=False, feature_dim=4, layer_mode="layer", layer_index=2)
    conditioner.use_pretrained = True
    conditioner.encoder = FakeWavLMEncoder()
    conditioner.adapter = torch.nn.Identity()

    output = conditioner(torch.zeros(2, 1600))

    assert torch.equal(output, torch.full((2, 4, 4), 2.0))


def test_wavlm_conditioner_selects_last_hidden_layer():
    conditioner = WavLMConditioner(output_dim=4, use_pretrained=False, feature_dim=4, layer_mode="last")
    conditioner.use_pretrained = True
    conditioner.encoder = FakeWavLMEncoder()
    conditioner.adapter = torch.nn.Identity()

    output = conditioner(torch.zeros(1, 1600))

    assert torch.equal(output, torch.full((1, 4, 4), 3.0))


def test_wavlm_conditioner_returns_exact_feature_mask_for_padded_batch():
    conditioner = WavLMConditioner(
        output_dim=4,
        use_pretrained=False,
        feature_dim=4,
        layer_mode="layer",
        layer_index=2,
    )
    conditioner.use_pretrained = True
    conditioner.encoder = FakeWavLMEncoder()
    conditioner.adapter = torch.nn.Identity()

    prefix, prefix_mask = conditioner.encode_with_mask(
        torch.zeros(2, 1600),
        lengths_16k=torch.tensor([960, 1600]),
    )

    assert prefix.shape == (2, 4, 4)
    assert torch.equal(
        prefix_mask,
        torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
    )
    assert torch.equal(
        conditioner.encoder.last_attention_mask.sum(dim=1),
        torch.tensor([960, 1600]),
    )


def test_xcodec_token_lengths_use_configured_codec_hop():
    tokenizer = XCodecFirstRVQTokenizer(vocab_size=8, codec_hop_length=320)
    lengths = torch.tensor([320, 321, 16000, 16001])

    assert torch.equal(
        tokenizer.token_lengths_from_waveform_lengths(lengths),
        torch.tensor([1, 2, 50, 51]),
    )


def test_wavlm_kmeans_vq_tokenizer_uses_fixed_codebook_and_mask(tmp_path, monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    codebook_path = tmp_path / "semantic_vq_codebook.npz"
    np.savez(
        codebook_path,
        centers=np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(hybrid_xcodec.AutoModel, "from_pretrained", lambda _path: FakeWavLMEncoder())

    tokenizer = WavLMKMeansVQTokenizer(
        vocab_size=3,
        codebook_path=str(codebook_path),
        wavlm_model_path="fake-wavlm",
        layer_index=2,
        feature_normalization="none",
        codec_hop_length=320,
    )
    batch = tokenizer.encode_first_rvq_batch(
        torch.zeros(2, 1600),
        lengths_16k=torch.tensor([960, 1600]),
    )

    assert torch.equal(
        batch.tokens,
        torch.tensor(
            [
                [
                    1,
                    1,
                    tokenizer.vocab_size + 1,
                    tokenizer.vocab_size + 1,
                ],
                [1, 1, 1, 1],
            ]
        ),
    )
    assert torch.equal(
        batch.mask,
        torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
    )
    assert torch.equal(
        tokenizer.encoder.last_attention_mask.sum(dim=1),
        torch.tensor([960, 1600]),
    )


def test_wavlm_kmeans_vq_token_lengths_use_encoder_feature_timebase(tmp_path, monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    codebook_path = tmp_path / "semantic_vq_codebook.npz"
    np.savez(codebook_path, centers=np.zeros((2, 4), dtype=np.float32))
    monkeypatch.setattr(
        hybrid_xcodec.AutoModel,
        "from_pretrained",
        lambda _path: FakeWavLMEncoder(),
    )
    tokenizer = WavLMKMeansVQTokenizer(
        vocab_size=2,
        codebook_path=str(codebook_path),
        wavlm_model_path="fake-wavlm",
        layer_index=2,
        feature_normalization="none",
        codec_hop_length=320,
    )
    lengths = torch.tensor([400, 640, 960, 16000, 16001, 48000, 79934, 80000, 85200])

    token_lengths = tokenizer.token_lengths_from_waveform_lengths(lengths)

    assert torch.equal(token_lengths, torch.tensor([1, 1, 2, 49, 49, 149, 249, 249, 266]))
    assert tokenizer.max_token_count_from_waveform_lengths(lengths) == 266

    try:
        tokenizer.token_lengths_from_waveform_lengths(torch.tensor([399]))
    except ValueError as exc:
        assert "too short" in str(exc)
    else:
        raise AssertionError("Sub-frame WavLM inputs should fail fast")


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


def test_avqi_gap_tokens_encode_direction_and_value():
    assert HybridUniSELightning._gap_token(0.1234567) == "pos0p123457"
    assert HybridUniSELightning._gap_token(-0.5) == "neg0p500000"


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


def test_webdataset_audio_reader_reports_context_for_invalid_payload():
    reader = WebDatasetAudioReader()
    item = {
        "_shard_dir": "/tmp/shards",
        "shard": "bad.tar",
        "audio_member": "clean/bad.wav",
        "key": "bad",
        "role": "clean",
        "_tar_offset_data": 512,
        "_tar_size": 9,
    }

    try:
        reader.decode(b"not audio", item)
    except AudioDecodeError as exc:
        message = str(exc)
        assert "role=clean" in message
        assert "member=clean/bad.wav" in message
        assert "offset=512" in message
    else:
        raise AssertionError("invalid audio payload should fail with context")


def test_webdataset_audio_reader_falls_back_from_bad_offset_to_tar_member(tmp_path):
    wav = np.zeros(160, dtype=np.float32)
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav, 16000, format="WAV")
    wav_bytes = wav_buffer.getvalue()

    tar_path = tmp_path / "audio.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("clean/ok.wav")
        info.size = len(wav_bytes)
        tar.addfile(info, io.BytesIO(wav_bytes))

    reader = WebDatasetAudioReader(target_sample_rate=16000)
    loaded = reader.read(
        {
            "_shard_dir": str(tmp_path),
            "shard": "audio.tar",
            "audio_member": "clean/ok.wav",
            "key": "ok",
            "role": "clean",
            "_tar_offset_data": 0,
            "_tar_size": 32,
        }
    )

    assert loaded.shape == (1, 160)


def test_tiny_hybrid_forward_shapes_and_mask_range():
    model = HybridUniSELightning(tiny_hybrid_config("fusion", fusion_use_teacher_forcing=True))
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


def test_gen_nll_only_freezes_and_excludes_refinement_from_optimizer():
    model = HybridUniSELightning(
        tiny_hybrid_config(
            "gen",
            loss_weights={
                "gen": {"nll": 1.0, "complex": 0.0, "mag": 0.0, "pmsqe": 0.0}
            },
        )
    )

    optimizer = model.configure_optimizers()
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert not model.gen_train_refinement
    assert not any(parameter.requires_grad for parameter in model.refinement.parameters())
    assert not (
        {id(parameter) for parameter in model.refinement.parameters()}
        & optimizer_parameter_ids
    )
    assert any(parameter.requires_grad for parameter in model.lm.parameters())
    assert any(
        parameter.requires_grad for parameter in model.conditioner.adapter.parameters()
    )


def test_gen_rejects_trainable_refinement_without_refinement_loss():
    config = tiny_hybrid_config(
        "gen",
        loss_weights={
            "gen": {"nll": 1.0, "complex": 0.0, "mag": 0.0, "pmsqe": 0.0}
        },
        gen_train_refinement=True,
    )

    try:
        HybridUniSELightning(config)
    except ValueError as exc:
        assert "gen_train_refinement=true" in str(exc)
    else:
        raise AssertionError("A zero-loss refinement branch should fail fast")


def test_tiny_paper_aligned_gen_forward_shapes():
    model = HybridUniSELightning(
        tiny_hybrid_config(
            "gen",
            lm={
                "architecture": "llama",
                "hidden_size": 16,
                "num_layers": 1,
                "num_attention_heads": 4,
                "dropout": 0.0,
                "max_position_embeddings": 256,
            },
            refinement={
                "architecture": "paper_dprnn",
                "channels": 8,
                "hidden": 4,
                "num_blocks": 1,
                "num_heads": 2,
                "dprnn_window_length": 640,
                "dprnn_hop_length": 320,
                "stft_hop_length": 160,
            },
        )
    )
    wav = torch.randn(1, 1600)
    clean = torch.randn(1, 1600)
    output = model(wav, torch.tensor([16000]), clean_wav=clean, length=torch.tensor([1600]))

    assert output.final_wav.shape == wav.shape
    assert output.gen_wav.shape == wav.shape
    assert output.token_logits.shape[:2] == output.token_targets.shape
    assert output.lm_hidden_states.shape[:2] == output.token_targets.shape
    assert output.aligned_lm_hidden_states.shape[1] == output.gen_spec_16k.shape[-1]
    assert model.refinement.chunk_size == 4
    assert model.refinement.hop_size == 2


def test_tiny_gen_can_use_wavlm_kmeans_vq_target(tmp_path, monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    codebook_path = tmp_path / "semantic_vq_codebook.npz"
    np.savez(
        codebook_path,
        centers=np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(hybrid_xcodec.AutoModel, "from_pretrained", lambda _path: FakeWavLMEncoder())
    config = tiny_hybrid_config(
        "gen",
        xcodec={
            "target_type": "wavlm_kmeans_vq",
            "vocab_size": 4,
            "codebook_path": str(codebook_path),
            "wavlm_model_path": "fake-wavlm",
            "layer_index": 2,
            "feature_normalization": "none",
            "codec_hop_length": 320,
        },
    )
    model = HybridUniSELightning(config)
    wav = torch.randn(1, 960)
    clean = torch.randn(1, 960)
    output = model(wav, torch.tensor([16000]), clean_wav=clean, length=torch.tensor([640]))

    assert output.token_targets.shape == (1, 2)
    assert torch.equal(output.lm_hidden_mask, torch.tensor([[True, False]]))
    assert output.token_targets[0, -1].item() == model.lm.pad_token_id


def test_gen_inference_uses_xcodec_timebase_for_lm_max_tokens(monkeypatch):
    model = HybridUniSELightning(
        tiny_hybrid_config(
            "gen",
            xcodec={"backend": "deterministic_stub", "vocab_size": 32, "codec_hop_length": 320},
        )
    )
    captured = {}

    def fake_generate(
        prefix,
        max_tokens,
        temperature=1.0,
        do_sample=False,
        prefix_mask=None,
    ):
        _ = temperature, do_sample, prefix_mask
        captured["max_tokens"] = max_tokens
        return {
            "tokens": torch.zeros(prefix.size(0), max_tokens, dtype=torch.long, device=prefix.device),
            "hidden_states": torch.zeros(prefix.size(0), max_tokens, 16, device=prefix.device),
            "hidden_mask": torch.ones(prefix.size(0), max_tokens, dtype=torch.bool, device=prefix.device),
        }

    monkeypatch.setattr(model.lm, "generate", fake_generate)
    wav = torch.randn(1, 321)
    output = model(wav, torch.tensor([16000]), length=torch.tensor([321]))

    assert captured["max_tokens"] == 2
    assert output.gen_spec_16k.shape[-1] == 3
    assert output.token_targets.shape == (1, 2)
    assert output.lm_hidden_states.shape[1] == 2
    assert output.aligned_lm_hidden_states.shape[1] == output.gen_spec_16k.shape[-1]


def test_gen_inference_uses_wavlm_feature_timebase_for_lm_max_tokens(
    tmp_path,
    monkeypatch,
):
    import model.hybrid_xcodec as hybrid_xcodec

    codebook_path = tmp_path / "semantic_vq_codebook.npz"
    np.savez(codebook_path, centers=np.zeros((4, 4), dtype=np.float32))
    monkeypatch.setattr(
        hybrid_xcodec.AutoModel,
        "from_pretrained",
        lambda _path: FakeWavLMEncoder(),
    )
    model = HybridUniSELightning(
        tiny_hybrid_config(
            "gen",
            wavlm={
                "pretrained_name_or_path": "fake-wavlm",
                "use_pretrained": True,
                "freeze": True,
                "feature_dim": 4,
                "layer_mode": "layer",
                "layer_index": 2,
            },
            xcodec={
                "target_type": "wavlm_kmeans_vq",
                "vocab_size": 4,
                "codebook_path": str(codebook_path),
                "wavlm_model_path": "fake-wavlm",
                "layer_index": 2,
                "feature_normalization": "none",
                "codec_hop_length": 320,
            },
        )
    )
    captured = {}

    def fake_generate(
        prefix,
        max_tokens,
        temperature=1.0,
        do_sample=False,
        prefix_mask=None,
    ):
        _ = temperature, do_sample
        captured["max_tokens"] = max_tokens
        captured["prefix_mask"] = prefix_mask
        return {
            "tokens": torch.zeros(
                prefix.size(0),
                max_tokens,
                dtype=torch.long,
                device=prefix.device,
            ),
            "hidden_states": torch.zeros(
                prefix.size(0),
                max_tokens,
                16,
                device=prefix.device,
            ),
            "hidden_mask": torch.ones(
                prefix.size(0),
                max_tokens,
                dtype=torch.bool,
                device=prefix.device,
            ),
        }

    monkeypatch.setattr(model.lm, "generate", fake_generate)
    wav = torch.randn(1, 960)
    output = model(wav, torch.tensor([16000]), length=torch.tensor([960]))

    assert captured["max_tokens"] == 2
    assert captured["prefix_mask"].shape == (1, 2)
    assert captured["prefix_mask"].all()
    assert output.token_targets.shape == (1, 2)
    assert output.lm_hidden_mask.shape == (1, 2)
    assert output.lm_hidden_mask.all()


def test_gen_inference_masks_variable_length_wavlm_batch(tmp_path, monkeypatch):
    import model.hybrid_xcodec as hybrid_xcodec

    codebook_path = tmp_path / "semantic_vq_codebook.npz"
    np.savez(codebook_path, centers=np.zeros((4, 4), dtype=np.float32))
    monkeypatch.setattr(
        hybrid_xcodec.AutoModel,
        "from_pretrained",
        lambda _path: FakeWavLMEncoder(),
    )
    model = HybridUniSELightning(
        tiny_hybrid_config(
            "gen",
            wavlm={
                "pretrained_name_or_path": "fake-wavlm",
                "use_pretrained": True,
                "freeze": True,
                "feature_dim": 4,
                "layer_mode": "layer",
                "layer_index": 2,
            },
            xcodec={
                "target_type": "wavlm_kmeans_vq",
                "vocab_size": 4,
                "codebook_path": str(codebook_path),
                "wavlm_model_path": "fake-wavlm",
                "layer_index": 2,
                "feature_normalization": "none",
                "codec_hop_length": 320,
            },
        )
    )
    captured = {}

    def fake_generate(
        prefix,
        max_tokens,
        temperature=1.0,
        do_sample=False,
        prefix_mask=None,
    ):
        _ = temperature, do_sample
        captured["max_tokens"] = max_tokens
        captured["prefix_mask"] = prefix_mask
        return {
            "tokens": torch.zeros(
                prefix.size(0),
                max_tokens,
                dtype=torch.long,
                device=prefix.device,
            ),
            "hidden_states": torch.zeros(
                prefix.size(0),
                max_tokens,
                16,
                device=prefix.device,
            ),
            "hidden_mask": torch.ones(
                prefix.size(0),
                max_tokens,
                dtype=torch.bool,
                device=prefix.device,
            ),
        }

    monkeypatch.setattr(model.lm, "generate", fake_generate)
    wav = torch.randn(2, 1600)
    output = model(
        wav,
        torch.tensor([16000, 16000]),
        length=torch.tensor([960, 1600]),
    )

    assert captured["max_tokens"] == 4
    assert torch.equal(
        captured["prefix_mask"],
        torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
    )
    assert torch.equal(
        output.lm_hidden_mask,
        torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
    )
    assert torch.equal(
        output.token_targets[0, 2:],
        torch.full((2,), model.lm.pad_token_id, dtype=torch.long),
    )


def test_enhance_builds_one_length_per_batch_item():
    model = HybridUniSELightning(tiny_hybrid_config("gen"))
    wav = torch.randn(2, 960)

    output = model.enhance(wav, torch.tensor([16000, 16000]))

    assert output.final_wav.shape == wav.shape
    assert torch.equal(output.length, torch.tensor([960, 960]))


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


def test_lm_hidden_alignment_excludes_masked_tail_values():
    model = HybridUniSELightning(
        tiny_hybrid_config(
            "disc",
            lm_stft_alignment_mode="interpolate",
        )
    )
    lm_hidden = torch.cat(
        [torch.ones(1, 2, 16), torch.full((1, 2, 16), 100.0)],
        dim=1,
    )
    lm_mask = torch.tensor([[True, True, False, False]])

    hidden, mask = model._align_lm_hidden_to_stft(
        lm_hidden,
        lm_mask,
        stft_frames=8,
    )

    assert torch.equal(
        mask,
        torch.tensor([[True, True, True, True, False, False, False, False]]),
    )
    torch.testing.assert_close(hidden[mask], torch.ones_like(hidden[mask]))
    assert torch.equal(hidden[~mask], torch.zeros_like(hidden[~mask]))


def test_lm_hidden_raw_cross_attention_keeps_codec_timebase():
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
            "lm_stft_alignment_mode": "raw_cross_attn",
        }
    )
    lm_hidden = torch.randn(2, 3, 16)
    lm_mask = torch.ones(2, 3, dtype=torch.bool)
    hidden, mask = model._align_lm_hidden_to_stft(lm_hidden, lm_mask, stft_frames=7)

    assert hidden is lm_hidden
    assert mask is lm_mask
    assert hidden.shape == (2, 3, 16)
    assert mask.shape == (2, 3)


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


def test_hybrid_checkpoint_architecture_metadata_rejects_refinement_mask_variant():
    checkpoint_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "lm": {"hidden_size": 16},
            "refinement": {"architecture": "paper_dprnn", "channels": 8, "num_heads": 2},
        }
    )
    current_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "lm": {"hidden_size": 16},
            "refinement": {"architecture": "paper_dprnn_identity_mask", "channels": 8, "num_heads": 2},
        }
    )

    try:
        validate_hybrid_architecture_metadata(
            {"hybrid_architecture_config": checkpoint_architecture},
            current_architecture,
        )
    except ValueError as exc:
        assert "architecture" in str(exc)
    else:
        raise AssertionError("mask variant mismatch should fail by default")

    try:
        validate_hybrid_architecture_metadata(
            {"hybrid_architecture_config": checkpoint_architecture},
            current_architecture,
            allow_refinement_mask_variant=True,
        )
    except ValueError as exc:
        assert "unsafe" in str(exc)
        assert "X*M" in str(exc)
    else:
        raise AssertionError("full-mask checkpoint migration must remain blocked")


def test_identity_refinement_rejects_metadata_less_checkpoint():
    identity_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "lm": {"hidden_size": 16},
            "refinement": {
                "architecture": "paper_dprnn_identity_mask",
                "channels": 8,
                "num_heads": 2,
            },
        }
    )
    try:
        validate_hybrid_architecture_metadata({}, identity_architecture)
    except ValueError as exc:
        assert "require hybrid_architecture_config metadata" in str(exc)
        assert "X*(1+M)" in str(exc)
    else:
        raise AssertionError("identity-mask checkpoint loads must fail closed")

    legacy_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "lm": {"hidden_size": 16},
            "refinement": {"architecture": "paper_dprnn"},
        }
    )
    validate_hybrid_architecture_metadata({}, legacy_architecture)


def test_hybrid_model_rejects_unsafe_refinement_mask_variant_flag():
    for value in (True, "false", 1):
        config = tiny_hybrid_config("gen")
        config["stage_init_allow_refinement_mask_variant"] = value
        try:
            HybridUniSELightning(config)
        except ValueError as exc:
            assert "stage_init_allow_refinement_mask_variant" in str(exc)
        else:
            raise AssertionError(f"Unsafe refinement flag {value!r} must fail")


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


def test_hybrid_checkpoint_architecture_metadata_ignores_wavlm_source_path_alias():
    checkpoint_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "wavlm": {
                "pretrained_name_or_path": "microsoft/wavlm-base-plus",
                "freeze": True,
                "use_pretrained": True,
                "feature_dim": 768,
            },
            "lm": {"hidden_size": 16},
        }
    )
    current_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "wavlm": {
                "pretrained_name_or_path": "./pretrained/wavlm/microsoft_wavlm-base-plus",
                "freeze": True,
                "use_pretrained": True,
                "feature_dim": 768,
            },
            "lm": {"hidden_size": 16},
        }
    )

    validate_hybrid_architecture_metadata(
        {"hybrid_architecture_config": checkpoint_architecture},
        current_architecture,
    )


def test_hybrid_checkpoint_architecture_metadata_ignores_xcodec_wavlm_source_path_alias():
    checkpoint_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "wavlm": {
                "pretrained_name_or_path": "microsoft/wavlm-base-plus",
                "freeze": True,
            },
            "xcodec": {
                "target_type": "wavlm_kmeans_vq",
                "vocab_size": 128,
                "wavlm_model_path": "./pretrained/wavlm/microsoft_wavlm-base-plus",
                "layer_index": 11,
            },
            "lm": {"hidden_size": 16},
        }
    )
    current_architecture = hybrid_architecture_config(
        {
            "sfi": {"supported_sample_rates": [16000]},
            "wavlm": {
                "pretrained_name_or_path": "/frozen/pretrained/wavlm",
                "freeze": True,
            },
            "xcodec": {
                "target_type": "wavlm_kmeans_vq",
                "vocab_size": 128,
                "wavlm_model_path": (
                    "/scratch/work/lil14/Hybrid_Unise_autofix_20260711/"
                    "pretrained/wavlm/microsoft_wavlm-base-plus"
                ),
                "layer_index": 11,
            },
            "lm": {"hidden_size": 16},
        }
    )

    validate_hybrid_architecture_metadata(
        {"hybrid_architecture_config": checkpoint_architecture},
        current_architecture,
    )


def test_hybrid_checkpoint_architecture_metadata_still_rejects_xcodec_shape_mismatch():
    base_xcodec = {
        "target_type": "wavlm_kmeans_vq",
        "vocab_size": 128,
        "codebook_path": "/frozen/codebook.npz",
        "wavlm_model_path": "./pretrained/wavlm",
        "layer_index": 11,
        "feature_normalization": "standardize_l2",
        "codec_hop_length": 320,
    }
    checkpoint_architecture = hybrid_architecture_config({"xcodec": base_xcodec})
    incompatible_values = {
        "vocab_size": 256,
        "codebook_path": "/frozen/other-codebook.npz",
        "layer_index": 10,
        "feature_normalization": "none",
        "codec_hop_length": 640,
    }

    for field, incompatible_value in incompatible_values.items():
        current_xcodec = dict(
            base_xcodec,
            wavlm_model_path="/frozen/pretrained/wavlm",
            **{field: incompatible_value},
        )
        current_architecture = hybrid_architecture_config({"xcodec": current_xcodec})

        try:
            validate_hybrid_architecture_metadata(
                {"hybrid_architecture_config": checkpoint_architecture},
                current_architecture,
            )
        except ValueError as exc:
            assert "architecture" in str(exc)
        else:
            raise AssertionError(f"xcodec {field} mismatch should fail")


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
    base_config = tiny_hybrid_config()

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
    model = HybridUniSELightning(tiny_hybrid_config("disc"))

    def fail_gen(*_args, **_kwargs):
        raise AssertionError("_gen should not run for default disc forward")

    monkeypatch.setattr(model, "_gen", fail_gen)
    output = model(torch.zeros(1, 320), torch.tensor([16000]))
    assert output.gen_wav is None
    assert output.token_targets is None


def test_gen_forward_default_does_not_run_discriminative(monkeypatch):
    model = HybridUniSELightning(tiny_hybrid_config("gen"))

    def fail_disc(*_args, **_kwargs):
        raise AssertionError("_disc should not run for default gen forward")

    monkeypatch.setattr(model, "_disc", fail_disc)
    wav = torch.randn(1, 3200)
    clean = torch.randn(1, 3200)
    output = model(wav, torch.tensor([16000]), clean_wav=clean)
    losses = model._losses(output, clean, torch.tensor([16000]))
    loss = model._weighted_stage_loss(losses)

    assert output.disc_wav is None
    assert output.disc_spec is None
    assert output.gen_wav.shape == wav.shape
    assert "disc_mrstft" not in losses
    assert torch.isfinite(loss)


def test_component_init_checkpoint_loads_selected_branch_weights(tmp_path):
    base_config = tiny_hybrid_config()
    disc_source = HybridUniSELightning(dict(base_config, stage="disc"))
    gen_source = HybridUniSELightning(dict(base_config, stage="gen"))
    for parameter in disc_source.discriminative.parameters():
        parameter.data.fill_(0.125)
    for module in (gen_source.conditioner, gen_source.lm, gen_source.refinement):
        for parameter in module.parameters():
            parameter.data.fill_(0.25)

    disc_ckpt = tmp_path / "disc.ckpt"
    gen_ckpt = tmp_path / "gen.ckpt"
    torch.save(
        {
            "state_dict": disc_source.state_dict(),
            "hybrid_stage": "disc",
            "hybrid_architecture_config": disc_source.architecture_config,
        },
        disc_ckpt,
    )
    torch.save(
        {
            "state_dict": gen_source.state_dict(),
            "hybrid_stage": "gen",
            "hybrid_architecture_config": gen_source.architecture_config,
        },
        gen_ckpt,
    )

    target = HybridUniSELightning(
        dict(
            base_config,
            stage="fusion",
            component_init_checkpoints={
                "discriminative": str(disc_ckpt),
                "generative": str(gen_ckpt),
            },
        )
    )

    for parameter in target.discriminative.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.125))
    for module in (target.conditioner, target.lm, target.refinement):
        for parameter in module.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 0.25))
    assert target.component_init_load_results["discriminative"]["source_stage"] == "disc"
    assert target.component_init_load_results["generative"]["source_stage"] == "gen"


def test_stage_init_and_component_init_are_mutually_exclusive():
    try:
        HybridUniSELightning(
            tiny_hybrid_config(
                "fusion",
                stage_init_checkpoint="disc.ckpt",
                component_init_checkpoints={"generative": "gen.ckpt"},
            )
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("stage_init_checkpoint and component_init_checkpoints should not be mixed")


class MovableDummyModel:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(torch.device(device))
        return self

    def eval(self):
        return self


def test_rir_start_sample_handles_peak_at_last_sample():
    start, end = get_rir_start_sample(torch.tensor([0.0, 0.1, 1.0]).numpy())

    assert start == 2
    assert end == 3


def test_shifted_anechoic_target_uses_direct_path_delay():
    clean = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
    rir = np.zeros((1, 8), dtype=np.float32)
    rir[0, 2] = 0.5
    rir[0, 5] = 1.0

    assert direct_path_delay(rir) == 5
    np.testing.assert_array_equal(
        shift_by_delay(clean, 5),
        np.array([[0, 0, 0, 0, 0, 1, 2, 3]], dtype=np.float32),
    )


def test_simulation_shifted_anechoic_reverb_target_excludes_reflections():
    clean = np.zeros((1, 10), dtype=np.float32)
    clean[0, 0] = 1.0
    noise = np.zeros((1, 10), dtype=np.float32)
    rir = np.zeros((1, 8), dtype=np.float32)
    rir[0, 2] = 1.0
    rir[0, 5] = 0.5
    config = {
        "se_interference": {"sir": [10.0, 10.0]},
        "tse_interference": {"sir": [10.0, 10.0]},
        "reverberation": {"prob": 1.0},
        "target": {"reverb_mode": "shifted_anechoic"},
        "noise": {"prob": 0.0, "snr": [20.0, 20.0]},
        "bandwidth_limitation": {"prob": 0.0, "fs_new": [16000], "res_type": "soxr_hq"},
        "clipping": {"prob": 0.0, "min_quantile": [0.0, 0.0], "max_quantile": [1.0, 1.0]},
        "packet_loss": {
            "prob": 0.0,
            "packet_duration_ms": 20,
            "packet_loss_rate": [0.0, 0.0],
            "max_continuous_packet_loss": 1,
        },
    }

    _noisy, target, _ = simulate_data(
        mode="se",
        speech=clean,
        interf=None,
        noise=noise,
        rir=rir,
        fs=16000,
        config=config,
        py_rng=random.Random(123),
        rng=np.random.default_rng(123),
    )

    assert target[0, 2] > 0.98
    assert target[0, 5] == 0.0


def test_simulation_reverb_target_defaults_to_shifted_anechoic():
    clean = np.zeros((1, 10), dtype=np.float32)
    clean[0, 0] = 1.0
    noise = np.zeros((1, 10), dtype=np.float32)
    rir = np.zeros((1, 8), dtype=np.float32)
    rir[0, 2] = 1.0
    rir[0, 5] = 0.5
    config = {
        "se_interference": {"sir": [10.0, 10.0]},
        "tse_interference": {"sir": [10.0, 10.0]},
        "reverberation": {"prob": 1.0},
        "noise": {"prob": 0.0, "snr": [20.0, 20.0]},
        "bandwidth_limitation": {"prob": 0.0, "fs_new": [16000], "res_type": "soxr_hq"},
        "clipping": {"prob": 0.0, "min_quantile": [0.0, 0.0], "max_quantile": [1.0, 1.0]},
        "packet_loss": {
            "prob": 0.0,
            "packet_duration_ms": 20,
            "packet_loss_rate": [0.0, 0.0],
            "max_continuous_packet_loss": 1,
        },
    }

    _noisy, target, _ = simulate_data(
        mode="se",
        speech=clean,
        interf=None,
        noise=noise,
        rir=rir,
        fs=16000,
        config=config,
        py_rng=random.Random(123),
        rng=np.random.default_rng(123),
    )

    assert target[0, 2] > 0.98
    assert target[0, 5] == 0.0


def test_transformers_xcodec_backend_moves_model_to_input_device():
    backend = TransformersXCodecFirstRVQ.__new__(TransformersXCodecFirstRVQ)
    backend.device = torch.device("cpu")
    backend.model = MovableDummyModel()

    backend._ensure_device(torch.device("meta"))

    assert backend.device == torch.device("meta")
    assert backend.model.devices == [torch.device("meta")]


class DummyEpochDataset:
    def __init__(self):
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class DummyEpochIter:
    def __init__(self):
        self.epoch = None
        self.dataset = DummyEpochDataset()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class DummyTrainer:
    current_epoch = 17


def test_datamodule_syncs_train_iterator_epoch_from_trainer():
    module = DataModule(train_kwargs={}, val_kwargs={}, test_kwargs={})
    module.train_iter = DummyEpochIter()

    module._sync_train_epoch_from_trainer(DummyTrainer())

    assert module.train_iter.epoch == 17
    assert module.train_iter.dataset.epoch == 17
