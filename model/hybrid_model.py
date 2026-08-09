import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn.functional as F

from .audio import SFIConfig, align_length, linear_resample, sfi_istft, sfi_stft
from .hybrid_discriminative import DiscriminativeBranch
from .hybrid_fusion import FusionBranch
from .hybrid_lm import HybridLlamaSemanticLM, HybridSemanticLM, WavLMConditioner
from .hybrid_losses import MultiResolutionSTFTLoss, build_external_loss, complex_mse, magnitude_mse
from .hybrid_objective import (
    lm_objective_identity,
    validate_checkpoint_lm_objective_identity,
)
from .hybrid_refinement import (
    GenerativeRefinementBranch,
    IdentityCenteredPaperDPRNNRefinementBranch,
    PaperDPRNNRefinementBranch,
)
from .hybrid_types import HybridOutput
from .hybrid_xcodec import WavLMKMeansVQTokenizer, XCodecFirstRVQTokenizer


def _cfg(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def validate_hybrid_checkpoint_metadata(checkpoint: dict[str, Any], stage: str) -> None:
    checkpoint_stage = checkpoint.get("hybrid_stage")
    if checkpoint_stage is not None and checkpoint_stage != stage:
        raise ValueError(
            f"Checkpoint was saved for hybrid stage '{checkpoint_stage}', "
            f"but current config requests stage '{stage}'."
        )


def hybrid_architecture_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "sfi": config.get("sfi", {}),
        "discriminative": config.get("discriminative", {}),
        "wavlm": config.get("wavlm", {}),
        "xcodec": config.get("xcodec", {}),
        "lm": config.get("lm", {}),
        "refinement": config.get("refinement", {}),
        "fusion": config.get("fusion", {}),
        "lm_stft_alignment_mode": config.get("lm_stft_alignment_mode", "interpolate"),
        "lm_stft_alignment_tolerance": config.get("lm_stft_alignment_tolerance", 2),
    }


def validate_hybrid_architecture_metadata(
    checkpoint: dict[str, Any],
    expected_architecture: dict[str, Any],
    allow_refinement_mask_variant: bool = False,
) -> None:
    if not isinstance(allow_refinement_mask_variant, bool):
        raise ValueError("allow_refinement_mask_variant must be a bool")
    if allow_refinement_mask_variant:
        raise ValueError(
            "Direct paper_dprnn/refinement identity-mask checkpoint compatibility "
            "is unsafe and unsupported because X*M and X*(1+M) are not "
            "functionally equivalent"
        )
    checkpoint_architecture = checkpoint.get("hybrid_architecture_config")
    if checkpoint_architecture is None:
        expected_refinement = expected_architecture.get("refinement")
        if (
            isinstance(expected_refinement, dict)
            and expected_refinement.get("architecture")
            == "paper_dprnn_identity_mask"
        ):
            raise ValueError(
                "Identity-centered refinement checkpoints require "
                "hybrid_architecture_config metadata so full-mask weights "
                "cannot be reinterpreted as X*(1+M)"
            )
        return
    comparable_checkpoint_architecture = {
        key: checkpoint_architecture.get(key)
        for key in expected_architecture
    }
    normalized_checkpoint = _normalize_hybrid_architecture_for_compare(comparable_checkpoint_architecture)
    normalized_expected = _normalize_hybrid_architecture_for_compare(expected_architecture)
    if normalized_checkpoint == normalized_expected:
        return
    raise ValueError("Checkpoint hybrid architecture config does not match current config.")


def _normalize_hybrid_architecture_for_compare(architecture: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in architecture.items():
        normalized[key] = dict(value) if isinstance(value, dict) else value
    wavlm = normalized.get("wavlm")
    if isinstance(wavlm, dict):
        wavlm = dict(wavlm)
        wavlm.pop("pretrained_name_or_path", None)
        normalized["wavlm"] = wavlm
    xcodec = normalized.get("xcodec")
    if isinstance(xcodec, dict):
        xcodec = dict(xcodec)
        xcodec.pop("wavlm_model_path", None)
        normalized["xcodec"] = xcodec
    return normalized


def _checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    return checkpoint.get("state_dict", checkpoint)


COMPONENT_STATE_PREFIXES = {
    "discriminative": ("discriminative.",),
    "disc": ("discriminative.",),
    "generative": ("conditioner.", "lm.", "refinement."),
    "gen": ("conditioner.", "lm.", "refinement."),
    "fusion": ("fusion.",),
}


def _require_finite(name: str, value: torch.Tensor | None) -> None:
    if value is not None and not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def load_hybrid_checkpoint(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class HybridUniSELightning(pl.LightningModule):
    """Hybrid discriminative/generative/fusion reproduction path.

    This module follows the paper data flow. Missing unpublished or unavailable
    assets are surfaced as implementation choices in config rather than hidden
    behind renamed UniSE/BiCodec components.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.stage = str(config.get("stage", "disc"))
        self.architecture_config = hybrid_architecture_config(config)

        sfi_config = config.get("sfi", {})
        self.sfi = SFIConfig(
            window_ms=float(sfi_config.get("window_ms", 20.0)),
            hop_ms=float(sfi_config.get("hop_ms", 10.0)),
            supported_sample_rates=tuple(int(x) for x in sfi_config.get("supported_sample_rates", [8000, 16000, 24000, 32000, 48000])),
        )
        self.gen_sfi = SFIConfig(window_ms=20.0, hop_ms=10.0, supported_sample_rates=(16000,))

        disc_config = config.get("discriminative", {})
        self.discriminative = DiscriminativeBranch(
            embedding=int(disc_config.get("embedding", 64)),
            lstm_hidden=int(disc_config.get("lstm_hidden", 256)),
            num_blocks=int(disc_config.get("num_blocks", 8)),
            attention_heads=int(disc_config.get("attention_heads", 4)),
            dropout=float(disc_config.get("dropout", 0.0)),
        )

        lm_config = config.get("lm", {})
        xcodec_config = config.get("xcodec", {})
        vocab_size = int(xcodec_config.get("vocab_size", lm_config.get("vocab_size", 1024)))
        hidden_size = int(lm_config.get("hidden_size", 512))
        wavlm_config = config.get("wavlm", {})
        self.conditioner = WavLMConditioner(
            output_dim=hidden_size,
            freeze=bool(wavlm_config.get("freeze", True)),
            pretrained_name_or_path=str(wavlm_config.get("pretrained_name_or_path", "microsoft/wavlm-base-plus")),
            use_pretrained=bool(wavlm_config.get("use_pretrained", False)),
            feature_dim=int(wavlm_config.get("feature_dim", 768)),
            layer_mode=str(wavlm_config.get("layer_mode", "mean")),
            layer_index=int(wavlm_config.get("layer_index", -1)),
        )
        target_type = str(xcodec_config.get("target_type", "xcodec_rvq"))
        if target_type == "xcodec_rvq":
            self.xcodec = XCodecFirstRVQTokenizer(
                vocab_size=vocab_size,
                backend=str(xcodec_config.get("backend", "deterministic_stub")),
                model_path=xcodec_config.get("model_path"),
                rvq_axis=int(xcodec_config.get("rvq_axis", -1)),
                rvq_index=int(xcodec_config.get("rvq_index", 0)),
                codec_hop_length=xcodec_config.get("codec_hop_length", xcodec_config.get("hop_length")),
                backend_kwargs=xcodec_config.get("backend_kwargs", {}),
            )
        elif target_type == "wavlm_kmeans_vq":
            wavlm_model_path = xcodec_config.get("wavlm_model_path", wavlm_config.get("pretrained_name_or_path"))
            self.xcodec = WavLMKMeansVQTokenizer(
                vocab_size=vocab_size,
                codebook_path=str(xcodec_config["codebook_path"]),
                wavlm_model_path=str(wavlm_model_path),
                sample_rate=int(xcodec_config.get("sample_rate", 16000)),
                layer_index=int(xcodec_config.get("layer_index", 11)),
                feature_normalization=str(xcodec_config.get("feature_normalization", "standardize_l2")),
                codec_hop_length=int(xcodec_config.get("codec_hop_length", xcodec_config.get("hop_length", 320))),
                centers_key=str(xcodec_config.get("centers_key", "centers")),
                mean_key=str(xcodec_config.get("mean_key", "mean")),
                std_key=str(xcodec_config.get("std_key", "std")),
            )
        else:
            raise ValueError("xcodec.target_type must be 'xcodec_rvq' or 'wavlm_kmeans_vq'")
        label_smoothing_value = lm_config.get("label_smoothing", 0.0)
        if isinstance(label_smoothing_value, bool):
            raise ValueError("lm.label_smoothing must be finite and in [0, 1]")
        try:
            label_smoothing = float(label_smoothing_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lm.label_smoothing must be finite and in [0, 1]"
            ) from exc
        if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("lm.label_smoothing must be finite and in [0, 1]")
        lm_kwargs = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": int(lm_config.get("num_layers", 12)),
            "num_attention_heads": int(lm_config.get("num_attention_heads", 8)),
            "dropout": float(lm_config.get("dropout", 0.1)),
            "max_position_embeddings": int(lm_config.get("max_position_embeddings", 4096)),
            "label_smoothing": label_smoothing,
        }
        lm_architecture = str(lm_config.get("architecture", "transformer_encoder"))
        if lm_architecture == "transformer_encoder":
            self.lm = HybridSemanticLM(**lm_kwargs)
        elif lm_architecture == "llama":
            self.lm = HybridLlamaSemanticLM(
                **lm_kwargs,
                intermediate_size=lm_config.get("intermediate_size"),
                rms_norm_eps=float(lm_config.get("rms_norm_eps", 1.0e-6)),
            )
        else:
            raise ValueError("lm.architecture must be 'transformer_encoder' or 'llama'")

        refinement_config = config.get("refinement", {})
        refinement_kwargs = {
            "channels": int(refinement_config.get("channels", 64)),
            "hidden": int(refinement_config.get("hidden", 128)),
            "num_blocks": int(refinement_config.get("num_blocks", 4)),
            "lm_hidden": hidden_size,
            "num_heads": int(refinement_config.get("num_heads", 8)),
            "dropout": float(refinement_config.get("dropout", 0.0)),
        }
        refinement_architecture = str(refinement_config.get("architecture", "legacy"))
        if refinement_architecture == "legacy":
            self.refinement = GenerativeRefinementBranch(**refinement_kwargs)
        elif refinement_architecture in {"paper_dprnn", "paper_dprnn_identity_mask"}:
            refinement_class = (
                IdentityCenteredPaperDPRNNRefinementBranch
                if refinement_architecture == "paper_dprnn_identity_mask"
                else PaperDPRNNRefinementBranch
            )
            self.refinement = refinement_class(
                **refinement_kwargs,
                dprnn_window_length=int(refinement_config.get("dprnn_window_length", 640)),
                dprnn_hop_length=int(refinement_config.get("dprnn_hop_length", 320)),
                stft_hop_length=int(refinement_config.get("stft_hop_length", 160)),
                dprnn_window_frames=refinement_config.get("dprnn_window_frames"),
                dprnn_hop_frames=refinement_config.get("dprnn_hop_frames"),
            )
        else:
            raise ValueError(
                "refinement.architecture must be 'legacy', 'paper_dprnn', "
                "or 'paper_dprnn_identity_mask'"
            )

        fusion_config = config.get("fusion", {})
        self.fusion = FusionBranch(channels=int(fusion_config.get("channels", 32)))
        self.mrstft_loss = MultiResolutionSTFTLoss(**config.get("mrstft_loss", {}))
        self.loss_weights = config.get("loss_weights", {})
        lm_objective = config.get("lm_objective", {})
        if lm_objective is None:
            lm_objective = {}
        if not isinstance(lm_objective, dict):
            raise ValueError("lm_objective must be a mapping")
        (
            self.lm_objective_config,
            self.lm_objective_json,
            self.lm_objective_sha256,
        ) = lm_objective_identity(lm_objective)
        lm_objective = self.lm_objective_config
        self.lm_history_embedding_dropout_prob = float(
            lm_objective.get("history_embedding_dropout_prob", 0.0)
        )
        self.lm_history_corruption_replacement_fraction = float(
            lm_objective.get("history_corruption_replacement_fraction", 0.0)
        )
        transition_stall_history_corruption = lm_objective.get(
            "transition_stall_history_corruption",
            False,
        )
        if not isinstance(transition_stall_history_corruption, bool):
            raise ValueError(
                "lm_objective.transition_stall_history_corruption must be a bool"
            )
        self.lm_transition_stall_history_corruption = (
            transition_stall_history_corruption
        )
        self.lm_prefix_only_aux_weight = float(
            lm_objective.get("prefix_only_aux_weight", 0.0)
        )
        self.lm_transition_ce_multiplier = float(
            lm_objective.get("transition_ce_multiplier", 1.0)
        )
        transition_predecessor_margin = lm_objective.get(
            "transition_predecessor_margin",
            1.0,
        )
        transition_predecessor_margin_weight = lm_objective.get(
            "transition_predecessor_margin_weight",
            0.0,
        )
        if isinstance(transition_predecessor_margin, bool):
            raise ValueError(
                "lm_objective.transition_predecessor_margin must be finite "
                "and non-negative"
            )
        if isinstance(transition_predecessor_margin_weight, bool):
            raise ValueError(
                "lm_objective.transition_predecessor_margin_weight must be "
                "finite and non-negative"
            )
        try:
            self.lm_transition_predecessor_margin = float(
                transition_predecessor_margin
            )
            self.lm_transition_predecessor_margin_weight = float(
                transition_predecessor_margin_weight
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lm_objective transition predecessor margin values must be "
                "finite and non-negative"
            ) from exc
        normalize_transition_weights_per_sample = lm_objective.get(
            "normalize_transition_weights_per_sample",
            False,
        )
        if not isinstance(normalize_transition_weights_per_sample, bool):
            raise ValueError(
                "lm_objective.normalize_transition_weights_per_sample must be a bool"
            )
        self.lm_normalize_transition_weights_per_sample = (
            normalize_transition_weights_per_sample
        )
        if (
            not math.isfinite(self.lm_history_embedding_dropout_prob)
            or not 0.0 <= self.lm_history_embedding_dropout_prob < 1.0
        ):
            raise ValueError(
                "lm_objective.history_embedding_dropout_prob must be finite "
                "and in the interval [0, 1)"
            )
        if (
            not math.isfinite(self.lm_history_corruption_replacement_fraction)
            or not 0.0
            <= self.lm_history_corruption_replacement_fraction
            <= 1.0
        ):
            raise ValueError(
                "lm_objective.history_corruption_replacement_fraction must be "
                "finite and in the interval [0, 1]"
            )
        if (
            self.lm_history_corruption_replacement_fraction > 0.0
            and self.lm_history_embedding_dropout_prob == 0.0
        ):
            raise ValueError(
                "lm_objective.history_corruption_replacement_fraction requires "
                "a positive history_embedding_dropout_prob"
            )
        if (
            self.lm_transition_stall_history_corruption
            and self.lm_history_corruption_replacement_fraction == 0.0
        ):
            raise ValueError(
                "lm_objective.transition_stall_history_corruption requires a "
                "positive history_corruption_replacement_fraction"
            )
        if (
            not math.isfinite(self.lm_prefix_only_aux_weight)
            or self.lm_prefix_only_aux_weight < 0.0
        ):
            raise ValueError(
                "lm_objective.prefix_only_aux_weight must be finite "
                "and non-negative"
            )
        if (
            not math.isfinite(self.lm_transition_ce_multiplier)
            or self.lm_transition_ce_multiplier < 1.0
        ):
            raise ValueError(
                "lm_objective.transition_ce_multiplier must be finite "
                "and at least 1.0"
            )
        if (
            not math.isfinite(self.lm_transition_predecessor_margin)
            or self.lm_transition_predecessor_margin < 0.0
        ):
            raise ValueError(
                "lm_objective.transition_predecessor_margin must be finite "
                "and non-negative"
            )
        if (
            not math.isfinite(self.lm_transition_predecessor_margin_weight)
            or self.lm_transition_predecessor_margin_weight < 0.0
        ):
            raise ValueError(
                "lm_objective.transition_predecessor_margin_weight must be "
                "finite and non-negative"
            )
        self.lm_objective_enabled = (
            self.lm_history_embedding_dropout_prob > 0.0
            or self.lm_history_corruption_replacement_fraction > 0.0
            or self.lm_transition_stall_history_corruption
            or self.lm_prefix_only_aux_weight > 0.0
            or self.lm_transition_ce_multiplier != 1.0
            or self.lm_normalize_transition_weights_per_sample
            or self.lm_transition_predecessor_margin_weight > 0.0
        )
        if self.lm_objective_enabled and self.stage != "gen":
            raise ValueError(
                "A non-default lm_objective is currently supported only for "
                "stage='gen'"
            )
        external_losses = config.get("external_losses", {})
        self.pmsqe_loss = build_external_loss("PMSQE", external_losses.get("pmsqe"))
        self.sqa_loss = build_external_loss("SQA", external_losses.get("sqa"))
        gen_loss_weights = self.loss_weights.get("gen", {})
        self.gen_refinement_has_loss = (
            float(gen_loss_weights.get("complex", 0.1)) != 0.0
            or float(gen_loss_weights.get("mag", 0.9)) != 0.0
            or (
                float(gen_loss_weights.get("pmsqe", 0.01)) != 0.0
                and self.pmsqe_loss is not None
            )
        )
        configured_gen_train_refinement = config.get("gen_train_refinement")
        self.gen_train_refinement = (
            self.gen_refinement_has_loss
            if configured_gen_train_refinement is None
            else bool(configured_gen_train_refinement)
        )
        if self.stage == "gen" and self.gen_train_refinement and not self.gen_refinement_has_loss:
            raise ValueError(
                "gen_train_refinement=true requires a non-zero gen complex, mag, "
                "or pmsqe loss weight"
            )
        if (
            self.lm_history_embedding_dropout_prob > 0.0
            and self.stage == "gen"
            and self.gen_refinement_has_loss
        ):
            raise ValueError(
                "lm_objective.history_embedding_dropout_prob must be 0 when "
                "generative waveform losses are enabled because dropped LM "
                "history would also change refinement conditioning"
            )
        self.fusion_use_teacher_forcing = bool(config.get("fusion_use_teacher_forcing", False))
        self.resampling_backend = str(config.get("resampling", {}).get("backend", "linear"))
        if self.resampling_backend != "linear":
            raise ValueError(
                "Only resampling.backend='linear' is implemented in this repository. "
                "Wire a verified backend before selecting another value."
            )
        self.lm_stft_alignment_mode = str(config.get("lm_stft_alignment_mode", "interpolate"))
        if self.lm_stft_alignment_mode not in {"interpolate", "strict", "raw_cross_attn"}:
            raise ValueError("lm_stft_alignment_mode must be 'interpolate', 'strict', or 'raw_cross_attn'")
        self.stage_init_checkpoint = config.get("stage_init_checkpoint")
        allow_refinement_mask_variant = config.get(
            "stage_init_allow_refinement_mask_variant",
            False,
        )
        if not isinstance(allow_refinement_mask_variant, bool):
            raise ValueError(
                "stage_init_allow_refinement_mask_variant must be a bool"
            )
        if allow_refinement_mask_variant:
            raise ValueError(
                "stage_init_allow_refinement_mask_variant is unsafe and "
                "unsupported; full-mask and identity-centered refinement "
                "checkpoints are not functionally equivalent"
            )
        self.stage_init_allow_refinement_mask_variant = False
        component_init_config = config.get("component_init_checkpoints") or {}
        if self.stage_init_checkpoint and component_init_config:
            raise ValueError("stage_init_checkpoint and component_init_checkpoints are mutually exclusive.")
        if self.stage_init_checkpoint:
            stage_init_path = Path(str(self.stage_init_checkpoint)).expanduser()
            if not stage_init_path.is_absolute():
                stage_init_path = Path(str(config.get("_config_dir", "."))).expanduser() / stage_init_path
            self.stage_init_checkpoint = str(stage_init_path)
            self._load_stage_initialization_checkpoint(self.stage_init_checkpoint)
        self.component_init_checkpoints = self._normalize_component_init_checkpoints(component_init_config)
        if self.component_init_checkpoints:
            self._load_component_initialization_checkpoints(self.component_init_checkpoints)
        self._apply_stage_freezing()
        self._latest_grad_norm: float | None = None
        self._last_chart_log_step = -1
        self._last_avqi_validation_step = -1
        self._latest_avqi_gap_to_clean: float | None = None
        self._latest_avqi_metrics: dict[str, float] | None = None
        self._best_avqi_gap_abs_to_clean = float("inf")
        self._best_avqi_gap_pos_to_clean = float("inf")
        self._best_avqi_gap_neg_to_clean = float("-inf")
        self._best_avqi_gap_checkpoints: dict[str, str] = {}

    def _load_stage_initialization_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = load_hybrid_checkpoint(checkpoint_path, map_location="cpu")
        validate_hybrid_architecture_metadata(
            checkpoint,
            self.architecture_config,
            allow_refinement_mask_variant=self.stage_init_allow_refinement_mask_variant,
        )
        state_dict = _checkpoint_state_dict(checkpoint)
        load_result = self.load_state_dict(state_dict, strict=False)
        self.stage_init_source_stage = checkpoint.get("hybrid_stage")
        self.stage_init_missing_keys = list(load_result.missing_keys)
        self.stage_init_unexpected_keys = list(load_result.unexpected_keys)

    def _normalize_component_init_checkpoints(self, checkpoints: dict[str, str]) -> dict[str, str]:
        normalized = {}
        for component, checkpoint_path in checkpoints.items():
            component = str(component)
            if component not in COMPONENT_STATE_PREFIXES:
                raise ValueError(
                    f"Unknown component_init_checkpoints key '{component}'. "
                    f"Expected one of {sorted(COMPONENT_STATE_PREFIXES)}."
                )
            path = Path(str(checkpoint_path)).expanduser()
            if not path.is_absolute():
                path = Path(str(self.config.get("_config_dir", "."))).expanduser() / path
            normalized[component] = str(path)
        return normalized

    def _load_component_initialization_checkpoints(self, checkpoints: dict[str, str]) -> None:
        self.component_init_load_results = {}
        for component, checkpoint_path in checkpoints.items():
            checkpoint = load_hybrid_checkpoint(checkpoint_path, map_location="cpu")
            validate_hybrid_architecture_metadata(checkpoint, self.architecture_config)
            prefixes = COMPONENT_STATE_PREFIXES[component]
            state_dict = {
                key: value
                for key, value in _checkpoint_state_dict(checkpoint).items()
                if key.startswith(prefixes)
            }
            if not state_dict:
                raise ValueError(
                    f"No state_dict entries for component '{component}' were found in {checkpoint_path}."
                )
            load_result = self.load_state_dict(state_dict, strict=False)
            self.component_init_load_results[component] = {
                "checkpoint": checkpoint_path,
                "source_stage": checkpoint.get("hybrid_stage"),
                "loaded_keys": sorted(state_dict),
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "conditioner", None) is not None and self.conditioner.freeze:
            self.conditioner.encoder.eval()
        self.xcodec.eval()
        if mode:
            self._set_frozen_stage_modules_eval()
        return self

    def _apply_stage_freezing(self) -> None:
        for module in (self.discriminative, self.conditioner, self.lm, self.refinement, self.fusion):
            for parameter in module.parameters():
                parameter.requires_grad = True

        if self.stage == "disc":
            trainable = {self.discriminative}
        elif self.stage == "gen":
            trainable = {self.conditioner.adapter, self.lm}
            if self.gen_train_refinement:
                trainable.add(self.refinement)
        elif self.stage == "fusion":
            trainable = {self.fusion}
        elif self.stage == "joint":
            trainable = {self.discriminative, self.conditioner.adapter, self.lm, self.refinement, self.fusion}
        else:
            raise ValueError(f"Unknown hybrid stage: {self.stage}")

        for module in (self.discriminative, self.conditioner, self.lm, self.refinement, self.fusion):
            enabled = module in trainable
            for parameter in module.parameters():
                parameter.requires_grad = enabled
        for parameter in self.conditioner.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.conditioner.adapter.parameters():
            parameter.requires_grad = self.conditioner.adapter in trainable

    def _set_frozen_stage_modules_eval(self) -> None:
        if self.stage == "disc":
            for module in (self.conditioner, self.lm, self.refinement, self.fusion):
                module.eval()
        elif self.stage == "gen":
            frozen_modules = [self.discriminative, self.fusion]
            if not self.gen_train_refinement:
                frozen_modules.append(self.refinement)
            for module in frozen_modules:
                module.eval()
        elif self.stage == "fusion":
            for module in (self.discriminative, self.conditioner, self.lm, self.refinement):
                module.eval()

    @staticmethod
    def _normalize_batch(batch: Any, test: bool = False) -> dict[str, Any]:
        if isinstance(batch, dict):
            return {
                "mode": batch.get("mode", "se"),
                "degraded_wav": batch["degraded_wav"],
                "clean_wav": batch.get("clean_wav"),
                "sample_rate": batch["sample_rate"],
                "length": batch.get("length"),
                "utterance_id": batch.get("utterance_id"),
                "source_path": batch.get("source_path"),
                "clean_path": batch.get("clean_path"),
            }
        if test:
            mode, _enroll, src, tgt, fs, lengths, names = batch
            return {
                "mode": mode,
                "degraded_wav": src,
                "clean_wav": tgt,
                "sample_rate": fs,
                "length": lengths,
                "utterance_id": names,
            }
        mode, _enroll, mix, speech, _interf, fs, lengths, names = batch
        return {
            "mode": mode,
            "degraded_wav": mix,
            "clean_wav": speech,
            "sample_rate": fs,
            "length": lengths,
            "utterance_id": names,
        }

    def _sample_rate_int(self, sample_rate: torch.Tensor | int) -> int:
        if torch.is_tensor(sample_rate):
            unique = sample_rate.detach().cpu().flatten().unique()
            if unique.numel() != 1:
                raise ValueError("HybridUniSE requires batches bucketed by sample_rate.")
            return int(unique.item())
        return int(sample_rate)

    def _lengths_at_16k(
        self,
        lengths: torch.Tensor | None,
        batch_size: int,
        fallback_length: int,
        sample_rate: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.full((batch_size,), fallback_length, dtype=torch.long, device=device)
        sr = self._sample_rate_int(sample_rate)
        return torch.round(lengths.to(device=device, dtype=torch.float32) * 16000.0 / float(sr)).long().clamp_min(1)

    def _disc(self, wav: torch.Tensor, sample_rate: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec, _ = sfi_stft(wav, sample_rate, self.sfi)
        context = torch.no_grad() if self.stage in {"gen", "fusion"} else torch.enable_grad()
        with context:
            disc_spec = self.discriminative(spec)
        disc_wav = sfi_istft(disc_spec, sample_rate, self.sfi, length=wav.size(-1))
        return disc_wav, spec, disc_spec

    def _align_lm_hidden_to_stft(
        self,
        lm_hidden: torch.Tensor,
        lm_hidden_mask: torch.Tensor | None,
        stft_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_frames = int(lm_hidden.size(1))
        stft_frames = int(stft_frames)
        if self.lm_stft_alignment_mode == "raw_cross_attn":
            return lm_hidden, lm_hidden_mask
        tolerance = int(self.config.get("lm_stft_alignment_tolerance", 2))
        mismatch = abs(token_frames - stft_frames)
        if mismatch <= tolerance:
            if token_frames == stft_frames:
                return lm_hidden, lm_hidden_mask
            if self.lm_stft_alignment_mode == "strict":
                raise ValueError(
                    "Strict LM/STFT alignment requires equal lengths: "
                    f"tokens={token_frames}, stft_frames={stft_frames}."
                )
        elif self.lm_stft_alignment_mode == "strict":
            raise ValueError(
                "LM token hidden-state length is not aligned with the 16 kHz STFT grid: "
                f"tokens={token_frames}, stft_frames={stft_frames}, tolerance={tolerance}. "
                "Check X-Codec hop rate, WavLM conditioning, and padding masks."
            )

        if lm_hidden_mask is None:
            hidden = F.interpolate(
                lm_hidden.transpose(1, 2),
                size=stft_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            return hidden, None
        source_mask = lm_hidden_mask.to(
            device=lm_hidden.device,
            dtype=lm_hidden.dtype,
        ).unsqueeze(1)
        hidden = F.interpolate(
            lm_hidden.transpose(1, 2) * source_mask,
            size=stft_frames,
            mode="linear",
            align_corners=False,
        )
        interpolated_weight = F.interpolate(
            source_mask,
            size=stft_frames,
            mode="linear",
            align_corners=False,
        )
        hidden = (hidden / interpolated_weight.clamp_min(1.0e-8)).transpose(1, 2)
        mask = F.interpolate(
            lm_hidden_mask.float().unsqueeze(1),
            size=stft_frames,
            mode="nearest",
        ).squeeze(1).bool()
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        return hidden, mask

    def _validate_lm_stft_alignment(self, lm_hidden: torch.Tensor, stft_frames: int) -> None:
        previous_mode = self.lm_stft_alignment_mode
        self.lm_stft_alignment_mode = "strict"
        try:
            self._align_lm_hidden_to_stft(lm_hidden, None, stft_frames)
        finally:
            self.lm_stft_alignment_mode = previous_mode

    def _gen(
        self,
        degraded_wav: torch.Tensor,
        clean_wav: torch.Tensor | None,
        sample_rate: torch.Tensor | int,
        length: torch.Tensor | None = None,
        do_sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        context = torch.no_grad() if self.stage == "fusion" else torch.enable_grad()
        with context:
            wav_16k = linear_resample(degraded_wav, sample_rate, 16000)
            degraded_spec_16k, _ = sfi_stft(wav_16k, 16000, self.gen_sfi)
            lengths_16k = self._lengths_at_16k(
                length,
                wav_16k.size(0),
                wav_16k.size(-1),
                sample_rate,
                wav_16k.device,
            )
            prefix, prefix_mask = self.conditioner.encode_with_mask(
                wav_16k,
                lengths_16k,
            )

            if clean_wav is not None:
                clean_16k = linear_resample(clean_wav, sample_rate, 16000)
                token_batch = self.xcodec.encode_first_rvq_batch(clean_16k, lengths_16k=lengths_16k)
                optimize_lm_objective = (
                    self.stage == "gen"
                    and self.lm.training
                    and self.lm_objective_enabled
                )
                transition_ce_multiplier = (
                    self.lm_transition_ce_multiplier
                    if optimize_lm_objective
                    else 1.0
                )
                transition_predecessor_margin_weight = (
                    self.lm_transition_predecessor_margin_weight
                    if optimize_lm_objective
                    else 0.0
                )
                lm_out = self.lm(
                    prefix,
                    token_batch.tokens,
                    target_mask=token_batch.mask,
                    prefix_mask=prefix_mask,
                    history_embedding_dropout_prob=(
                        self.lm_history_embedding_dropout_prob
                        if optimize_lm_objective
                        else 0.0
                    ),
                    history_corruption_replacement_fraction=(
                        self.lm_history_corruption_replacement_fraction
                        if optimize_lm_objective
                        else 0.0
                    ),
                    transition_stall_history_corruption=(
                        self.lm_transition_stall_history_corruption
                        if optimize_lm_objective
                        else False
                    ),
                    transition_loss_weight=transition_ce_multiplier,
                    normalize_transition_weights_per_sample=(
                        self.lm_normalize_transition_weights_per_sample
                        if optimize_lm_objective
                        else False
                    ),
                    transition_predecessor_margin=(
                        self.lm_transition_predecessor_margin
                    ),
                    transition_predecessor_margin_weight=(
                        transition_predecessor_margin_weight
                    ),
                )
                lm_out["weighted_nll"] = lm_out["ce_objective_loss"]
                lm_out["training_objective_loss"] = None
                lm_out["prefix_only_nll"] = None
                lm_out["prefix_only_weighted_nll"] = None
                lm_out["prefix_only_transition_nll"] = None
                lm_out["prefix_only_transition_predecessor_margin_loss"] = None
                lm_out["prefix_only_transition_predecessor_rate"] = None
                if optimize_lm_objective:
                    training_objective_loss = lm_out["objective_loss"]
                    if self.lm_prefix_only_aux_weight > 0.0:
                        prefix_only_out = self.lm(
                            prefix,
                            token_batch.tokens,
                            target_mask=token_batch.mask,
                            prefix_mask=prefix_mask,
                            zero_history=True,
                            transition_loss_weight=transition_ce_multiplier,
                            normalize_transition_weights_per_sample=(
                                self.lm_normalize_transition_weights_per_sample
                            ),
                            transition_predecessor_margin=(
                                self.lm_transition_predecessor_margin
                            ),
                            transition_predecessor_margin_weight=(
                                self.lm_transition_predecessor_margin_weight
                            ),
                        )
                        auxiliary_weight = self.lm_prefix_only_aux_weight
                        training_objective_loss = (
                            training_objective_loss
                            + auxiliary_weight * prefix_only_out["objective_loss"]
                        ) / (1.0 + auxiliary_weight)
                        lm_out["prefix_only_nll"] = prefix_only_out["loss"]
                        lm_out["prefix_only_weighted_nll"] = prefix_only_out[
                            "ce_objective_loss"
                        ]
                        lm_out["prefix_only_transition_nll"] = prefix_only_out[
                            "transition_nll"
                        ]
                        lm_out[
                            "prefix_only_transition_predecessor_margin_loss"
                        ] = prefix_only_out[
                            "transition_predecessor_margin_loss"
                        ]
                        lm_out[
                            "prefix_only_transition_predecessor_rate"
                        ] = prefix_only_out["transition_predecessor_rate"]
                    lm_out["training_objective_loss"] = training_objective_loss
            else:
                token_lengths = self.xcodec.token_lengths_from_waveform_lengths(lengths_16k, device=wav_16k.device)
                max_tokens = int(token_lengths.max().item())
                generated_mask = (
                    torch.arange(max_tokens, device=wav_16k.device).unsqueeze(0)
                    < token_lengths.unsqueeze(1)
                )
                lm_out = self.lm.generate(
                    prefix,
                    max_tokens=max_tokens,
                    do_sample=do_sample,
                    prefix_mask=prefix_mask,
                )
                lm_out = {
                    "loss": None,
                    "accuracy": None,
                    "logits": None,
                    "targets": lm_out["tokens"].masked_fill(~generated_mask, self.lm.pad_token_id),
                    "hidden_states": lm_out["hidden_states"],
                    "hidden_mask": lm_out["hidden_mask"] & generated_mask,
                }

            aligned_hidden, aligned_mask = self._align_lm_hidden_to_stft(
                lm_out["hidden_states"],
                lm_out.get("hidden_mask"),
                degraded_spec_16k.size(-1),
            )
            gen_spec_16k = self.refinement(
                degraded_spec_16k,
                aligned_hidden,
                lm_hidden_mask=aligned_mask,
            )
            lm_out["aligned_hidden_states"] = aligned_hidden
            lm_out["aligned_hidden_mask"] = aligned_mask
        gen_wav_16k = sfi_istft(gen_spec_16k, 16000, self.gen_sfi, length=wav_16k.size(-1))
        gen_wav = align_length(linear_resample(gen_wav_16k, 16000, sample_rate), degraded_wav.size(-1))
        gen_spec, _ = sfi_stft(gen_wav, sample_rate, self.sfi)
        return gen_wav, gen_spec, gen_wav_16k, gen_spec_16k, lm_out

    def forward(
        self,
        degraded_wav: torch.Tensor,
        sample_rate: torch.Tensor | int,
        clean_wav: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
        return_intermediates: bool = False,
        do_sample: bool = False,
    ) -> HybridOutput:
        degraded_wav = degraded_wav.float()
        clean_wav = clean_wav.float() if clean_wav is not None else None
        needs_disc = self.stage in {"disc", "fusion", "joint"} or return_intermediates
        disc_wav = degraded_spec = disc_spec = None
        if needs_disc:
            disc_wav, degraded_spec, disc_spec = self._disc(degraded_wav, sample_rate)

        gen_wav = gen_wav_16k = gen_spec = gen_spec_16k = fusion_mask = None
        token_logits = token_targets = lm_hidden_states = None
        token_nll = token_objective_nll = token_weighted_nll = None
        token_initial_nll = token_repeat_nll = token_transition_nll = None
        token_accuracy = token_initial_accuracy = None
        token_repeat_accuracy = token_transition_accuracy = None
        token_transition_fraction = None
        token_transition_predecessor_margin_loss = None
        token_transition_predecessor_rate = None
        token_prefix_only_nll = token_prefix_only_weighted_nll = None
        token_prefix_only_transition_nll = None
        token_prefix_only_transition_predecessor_margin_loss = None
        token_prefix_only_transition_predecessor_rate = None
        lm_hidden_mask = None
        aligned_lm_hidden_states = None
        aligned_lm_hidden_mask = None
        if self.stage in {"gen", "fusion", "joint"} or return_intermediates:
            gen_clean_wav = clean_wav
            if self.stage == "fusion" and not self.fusion_use_teacher_forcing:
                gen_clean_wav = None
            gen_wav, gen_spec, gen_wav_16k, gen_spec_16k, lm_out = self._gen(
                degraded_wav,
                gen_clean_wav,
                sample_rate,
                length=length,
                do_sample=do_sample,
            )
            token_logits = lm_out["logits"]
            token_targets = lm_out["targets"]
            token_nll = lm_out["loss"]
            token_objective_nll = lm_out.get("training_objective_loss")
            token_weighted_nll = lm_out.get("weighted_nll")
            token_initial_nll = lm_out.get("initial_nll")
            token_repeat_nll = lm_out.get("repeat_nll")
            token_transition_nll = lm_out.get("transition_nll")
            token_accuracy = lm_out.get("accuracy")
            token_initial_accuracy = lm_out.get("initial_accuracy")
            token_repeat_accuracy = lm_out.get("repeat_accuracy")
            token_transition_accuracy = lm_out.get("transition_accuracy")
            token_transition_fraction = lm_out.get("transition_fraction")
            token_transition_predecessor_margin_loss = lm_out.get(
                "transition_predecessor_margin_loss"
            )
            token_transition_predecessor_rate = lm_out.get(
                "transition_predecessor_rate"
            )
            token_prefix_only_nll = lm_out.get("prefix_only_nll")
            token_prefix_only_weighted_nll = lm_out.get(
                "prefix_only_weighted_nll"
            )
            token_prefix_only_transition_nll = lm_out.get(
                "prefix_only_transition_nll"
            )
            token_prefix_only_transition_predecessor_margin_loss = lm_out.get(
                "prefix_only_transition_predecessor_margin_loss"
            )
            token_prefix_only_transition_predecessor_rate = lm_out.get(
                "prefix_only_transition_predecessor_rate"
            )
            lm_hidden_states = lm_out["hidden_states"]
            lm_hidden_mask = lm_out.get("hidden_mask")
            aligned_lm_hidden_states = lm_out.get("aligned_hidden_states")
            aligned_lm_hidden_mask = lm_out.get("aligned_hidden_mask")

        if self.stage in {"fusion", "joint"} and gen_spec is not None:
            if disc_spec is None:
                raise RuntimeError("Fusion stage requires discriminative output.")
            fusion_disc_spec = disc_spec.detach() if self.stage == "fusion" else disc_spec
            fusion_gen_spec = gen_spec.detach() if self.stage == "fusion" else gen_spec
            final_spec, fusion_mask = self.fusion(fusion_disc_spec, fusion_gen_spec)
            final_wav = sfi_istft(final_spec, sample_rate, self.sfi, length=degraded_wav.size(-1))
        else:
            final_spec = disc_spec if self.stage == "disc" else gen_spec
            final_wav = disc_wav if self.stage == "disc" else gen_wav

        return HybridOutput(
            final_wav=align_length(final_wav, degraded_wav.size(-1)),
            disc_wav=align_length(disc_wav, degraded_wav.size(-1)) if disc_wav is not None else None,
            gen_wav=align_length(gen_wav, degraded_wav.size(-1)) if gen_wav is not None else None,
            gen_wav_16k=gen_wav_16k,
            final_spec=final_spec,
            disc_spec=disc_spec,
            gen_spec=gen_spec,
            gen_spec_16k=gen_spec_16k,
            fusion_mask=fusion_mask,
            token_logits=token_logits,
            token_targets=token_targets,
            token_nll=token_nll,
            token_objective_nll=token_objective_nll,
            token_weighted_nll=token_weighted_nll,
            token_initial_nll=token_initial_nll,
            token_repeat_nll=token_repeat_nll,
            token_transition_nll=token_transition_nll,
            token_accuracy=token_accuracy,
            token_initial_accuracy=token_initial_accuracy,
            token_repeat_accuracy=token_repeat_accuracy,
            token_transition_accuracy=token_transition_accuracy,
            token_transition_fraction=token_transition_fraction,
            token_transition_predecessor_margin_loss=(
                token_transition_predecessor_margin_loss
            ),
            token_transition_predecessor_rate=token_transition_predecessor_rate,
            token_prefix_only_nll=token_prefix_only_nll,
            token_prefix_only_weighted_nll=token_prefix_only_weighted_nll,
            token_prefix_only_transition_nll=token_prefix_only_transition_nll,
            token_prefix_only_transition_predecessor_margin_loss=(
                token_prefix_only_transition_predecessor_margin_loss
            ),
            token_prefix_only_transition_predecessor_rate=(
                token_prefix_only_transition_predecessor_rate
            ),
            lm_hidden_states=lm_hidden_states,
            lm_hidden_mask=lm_hidden_mask,
            aligned_lm_hidden_states=aligned_lm_hidden_states,
            aligned_lm_hidden_mask=aligned_lm_hidden_mask,
            length=length,
        )

    def _validate_output_finite(self, output: HybridOutput) -> None:
        _require_finite("final_wav", output.final_wav)
        _require_finite("disc_wav", output.disc_wav)
        _require_finite("gen_wav", output.gen_wav)
        _require_finite("gen_wav_16k", output.gen_wav_16k)
        _require_finite("final_spec.real", output.final_spec.real if output.final_spec is not None else None)
        _require_finite("final_spec.imag", output.final_spec.imag if output.final_spec is not None else None)
        _require_finite("disc_spec.real", output.disc_spec.real if output.disc_spec is not None else None)
        _require_finite("disc_spec.imag", output.disc_spec.imag if output.disc_spec is not None else None)
        _require_finite("gen_spec.real", output.gen_spec.real if output.gen_spec is not None else None)
        _require_finite("gen_spec.imag", output.gen_spec.imag if output.gen_spec is not None else None)
        _require_finite("fusion_mask", output.fusion_mask)

    @torch.inference_mode()
    def enhance(
        self,
        wav: torch.Tensor,
        sample_rate: torch.Tensor | int,
        checkpoint: str | None = None,
        return_intermediates: bool = False,
    ) -> HybridOutput:
        if checkpoint is not None:
            checkpoint_data = load_hybrid_checkpoint(checkpoint, map_location=self.device)
            validate_hybrid_checkpoint_metadata(checkpoint_data, self.stage)
            validate_hybrid_architecture_metadata(checkpoint_data, self.architecture_config)
            self.load_state_dict(_checkpoint_state_dict(checkpoint_data), strict=False)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        output = self.forward(
            wav.to(self.device),
            sample_rate,
            clean_wav=None,
            length=torch.full(
                (wav.size(0),),
                wav.size(-1),
                dtype=torch.long,
                device=self.device,
            ),
            return_intermediates=return_intermediates,
            do_sample=False,
        )
        self._validate_output_finite(output)
        return output

    def _losses(
        self,
        output: HybridOutput,
        clean: torch.Tensor,
        sample_rate: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        if output.disc_wav is not None:
            losses["disc_mrstft"] = self.mrstft_loss(output.disc_wav, clean)
        if output.gen_spec_16k is not None and output.gen_wav_16k is not None:
            clean_16k = linear_resample(clean, sample_rate, 16000)
            clean_spec_16k, _ = sfi_stft(clean_16k, 16000, self.gen_sfi)
            losses["complex"] = complex_mse(output.gen_spec_16k, clean_spec_16k)
            losses["mag"] = magnitude_mse(output.gen_spec_16k, clean_spec_16k)
            if output.token_nll is not None:
                losses["nll"] = output.token_nll
            token_metrics = {
                "lm_objective": output.token_objective_nll,
                "weighted_nll": output.token_weighted_nll,
                "initial_nll": output.token_initial_nll,
                "repeat_nll": output.token_repeat_nll,
                "transition_nll": output.token_transition_nll,
                "accuracy": output.token_accuracy,
                "initial_accuracy": output.token_initial_accuracy,
                "repeat_accuracy": output.token_repeat_accuracy,
                "transition_accuracy": output.token_transition_accuracy,
                "transition_fraction": output.token_transition_fraction,
                "transition_predecessor_margin_loss": (
                    output.token_transition_predecessor_margin_loss
                ),
                "transition_predecessor_rate": (
                    output.token_transition_predecessor_rate
                ),
                "prefix_only_nll": output.token_prefix_only_nll,
                "prefix_only_weighted_nll": (
                    output.token_prefix_only_weighted_nll
                ),
                "prefix_only_transition_nll": (
                    output.token_prefix_only_transition_nll
                ),
                "prefix_only_transition_predecessor_margin_loss": (
                    output.token_prefix_only_transition_predecessor_margin_loss
                ),
                "prefix_only_transition_predecessor_rate": (
                    output.token_prefix_only_transition_predecessor_rate
                ),
            }
            losses.update(
                {
                    name: value
                    for name, value in token_metrics.items()
                    if value is not None
                }
            )
            if self.pmsqe_loss is not None:
                losses["pmsqe"] = self.pmsqe_loss(output.gen_wav_16k, clean_16k, sample_rate=16000)
        if self.stage in {"fusion", "joint"} and output.final_wav is not None:
            losses["fusion_mrstft"] = self.mrstft_loss(output.final_wav, clean)
            losses["fusion_l1"] = F.l1_loss(output.final_wav, clean)
            if self.sqa_loss is not None:
                losses["sqa"] = self.sqa_loss(output.final_wav, clean, sample_rate=sample_rate)
        return losses

    def _weighted_stage_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.stage == "disc":
            return losses["disc_mrstft"]
        if self.stage == "gen":
            weights = self.loss_weights.get("gen", {})
            semantic_loss = losses.get("lm_objective")
            if semantic_loss is None:
                semantic_loss = losses.get(
                    "nll",
                    torch.zeros((), device=self.device),
                )
            return (
                float(weights.get("nll", 1.0)) * semantic_loss
                + float(weights.get("complex", 0.1)) * losses["complex"]
                + float(weights.get("mag", 0.9)) * losses["mag"]
                + float(weights.get("pmsqe", 0.01)) * losses.get("pmsqe", torch.zeros((), device=self.device))
            )
        if self.stage in {"fusion", "joint"}:
            weights = self.loss_weights.get("fusion", {})
            return (
                float(weights.get("mrstft", 1.0)) * losses["fusion_mrstft"]
                + float(weights.get("l1", 0.5)) * losses["fusion_l1"]
                + float(weights.get("sqa", 0.0)) * losses.get("sqa", torch.zeros((), device=self.device))
            )
        raise ValueError(f"Unknown hybrid stage: {self.stage}")

    def training_step(self, batch, batch_idx):
        data = self._normalize_batch(batch)
        output = self.forward(
            data["degraded_wav"],
            data["sample_rate"],
            clean_wav=data["clean_wav"],
            length=data["length"],
            return_intermediates=self.stage in {"fusion", "joint"},
        )
        losses = self._losses(output, data["clean_wav"], data["sample_rate"])
        loss = self._weighted_stage_loss(losses)
        self._validate_output_finite(output)
        _require_finite("train/loss", loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def _log_metrics_direct(self, metrics: dict[str, float]) -> None:
        if not self.trainer.is_global_zero:
            return
        step = int(self.global_step)
        metrics = {**metrics, "charts/global_step": step}
        for logger in self.trainer.loggers:
            logger.log_metrics(metrics, step=step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = int(self.global_step)
        interval = int(self.config.get("wandb_log_interval_steps", self.config.get("log_every_n_steps", 50)))
        if interval > 0 and step > 0 and step % interval == 0 and step != self._last_chart_log_step:
            metrics = {
                "charts/epoch": float(self.current_epoch),
                "charts/lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
            }
            if self._latest_grad_norm is not None:
                metrics["charts/grad_norm"] = self._latest_grad_norm
            self._log_metrics_direct(metrics)
            self._last_chart_log_step = step

        avqi_interval = int(self.config.get("avqi_validation_interval_steps", 0))
        if (
            avqi_interval > 0
            and step > 0
            and step % avqi_interval == 0
            and step != self._last_avqi_validation_step
        ):
            avqi_metrics = self._validation_avqi_metrics() if self.trainer.is_global_zero else None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                payload = [avqi_metrics]
                torch.distributed.broadcast_object_list(payload, src=0)
                avqi_metrics = payload[0]
            if avqi_metrics is None:
                self._last_avqi_validation_step = step
                return
            if self.trainer.is_global_zero:
                avqi_wandb_metrics = self._format_avqi_wandb_metrics(avqi_metrics)
                self.print(f"AVQI W&B metrics: {', '.join(sorted(avqi_wandb_metrics))}")
                self._log_metrics_direct(avqi_wandb_metrics)
                self._latest_avqi_gap_to_clean = float(avqi_metrics["avqi_gap_to_clean"])
                self._latest_avqi_metrics = {key: float(value) for key, value in avqi_metrics.items()}
            if bool(self.config.get("save_avqi_best_checkpoints", True)):
                self._save_best_avqi_gap_checkpoints(avqi_metrics)
            self._last_avqi_validation_step = step

    def on_before_optimizer_step(self, optimizer):
        grad_norm_sq = torch.zeros((), device=self.device)
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grad_norm_sq += parameter.grad.detach().float().norm(2).pow(2)
        grad_norm = grad_norm_sq.sqrt()
        _require_finite("train/grad_norm", grad_norm)
        self._latest_grad_norm = float(grad_norm.detach().cpu())

    def validation_step(self, batch, batch_idx):
        data = self._normalize_batch(batch)
        output = self.forward(
            data["degraded_wav"],
            data["sample_rate"],
            clean_wav=data["clean_wav"],
            length=data["length"],
            return_intermediates=self.stage in {"fusion", "joint"},
        )
        losses = self._losses(output, data["clean_wav"], data["sample_rate"])
        loss = self._weighted_stage_loss(losses)
        self._validate_output_finite(output)
        _require_finite("val/loss", loss)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=False, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, on_step=False, on_epoch=True, sync_dist=True)

    def _load_avqi_runner(self):
        script_path = self.config.get("avqi_validation_script")
        if not script_path:
            raise ValueError("avqi_validation_script must be set when avqi_validation_interval_steps is enabled")
        script_path = Path(script_path)
        spec = importlib.util.spec_from_file_location("validation_avqi_gap", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load AVQI validation script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_validation_avqi_metrics, module.format_wandb_avqi_metrics

    def _format_avqi_wandb_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        _, format_wandb_avqi_metrics = self._load_avqi_runner()
        return format_wandb_avqi_metrics(metrics)

    def _validation_avqi_metrics(self) -> dict[str, float]:
        was_training = self.training
        self.eval()

        @torch.inference_mode()
        def enhance_one(path: str):
            audio, sample_rate = sf.read(path, always_2d=False)
            wav = torch.as_tensor(audio, dtype=torch.float32, device=self.device)
            if wav.ndim == 2:
                wav = wav.mean(dim=1)
            sample_rate = int(sample_rate)
            if sample_rate != 16000:
                wav = linear_resample(wav.unsqueeze(0), sample_rate, 16000).squeeze(0)
                sample_rate = 16000
            output = self.enhance(wav, sample_rate, return_intermediates=False)
            enhanced = output.final_wav.reshape(-1).detach().cpu().numpy()
            return enhanced, sample_rate

        run_validation_avqi_metrics, _ = self._load_avqi_runner()
        model_name = self.config.get("avqi_validation_model_name") or self.config.get("model_name") or "hybrid_unise"
        optional_kwargs = {
            "listening_root": self.config.get("avqi_validation_listening_root"),
            "workers": self.config.get("avqi_validation_workers"),
        }
        accepted_params = inspect.signature(run_validation_avqi_metrics).parameters
        runner_kwargs = {
            key: value
            for key, value in optional_kwargs.items()
            if value is not None and key in accepted_params
        }
        metrics = run_validation_avqi_metrics(
            str(model_name),
            int(self.global_step),
            enhance_one,
            pair_csv=self.config.get("avqi_validation_pair_csv"),
            output_root=self.config.get("avqi_validation_output_root"),
            clean_cache=self.config.get("avqi_validation_clean_cache"),
            **runner_kwargs,
        )
        if was_training:
            self.train()
        return metrics

    def _avqi_checkpoint_dir(self) -> Path:
        checkpoint_dir = self.config.get("ckpt_dir") or self.config.get("checkpoint_dir") or "checkpoints"
        return Path(checkpoint_dir)

    @staticmethod
    def _gap_token(gap: float) -> str:
        sign = "pos" if gap >= 0 else "neg"
        return f"{sign}{abs(gap):.6f}".replace(".", "p")

    def _save_semantic_checkpoint(self, prefix: str, direction: str, metrics: dict[str, float]) -> None:
        checkpoint_dir = self._avqi_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        gap = float(metrics["avqi_gap_to_clean"])
        step = int(self.global_step)
        epoch = int(self.current_epoch)
        checkpoint_path = checkpoint_dir / f"{prefix}_{self._gap_token(gap)}_epoch{epoch:02d}-step{step:06d}.ckpt"

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        self._best_avqi_gap_checkpoints[direction] = str(checkpoint_path)
        trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
        if not trainer.is_global_zero:
            return

        for old_checkpoint in checkpoint_dir.glob(f"{prefix}_*.ckpt"):
            if old_checkpoint != checkpoint_path:
                old_checkpoint.unlink(missing_ok=True)

        metrics_path = checkpoint_path.with_suffix(".json")
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint": str(checkpoint_path),
                    "direction": direction,
                    "epoch": epoch,
                    "global_step": step,
                    "metrics": {key: float(value) for key, value in metrics.items()},
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        for old_metrics in checkpoint_dir.glob(f"{prefix}_*.json"):
            if old_metrics != metrics_path:
                old_metrics.unlink(missing_ok=True)

    def _save_best_avqi_gap_checkpoints(self, metrics: dict[str, float]) -> None:
        gap = float(metrics["avqi_gap_to_clean"])
        if not math.isfinite(gap):
            return

        if abs(gap) < self._best_avqi_gap_abs_to_clean:
            self._best_avqi_gap_abs_to_clean = abs(gap)
            self._save_semantic_checkpoint("best_avqi_gap_abs", "abs", metrics)

        if gap >= 0 and gap < self._best_avqi_gap_pos_to_clean:
            self._best_avqi_gap_pos_to_clean = gap
            self._save_semantic_checkpoint("best_avqi_gap_pos", "pos", metrics)

        if gap < 0 and gap > self._best_avqi_gap_neg_to_clean:
            self._best_avqi_gap_neg_to_clean = gap
            self._save_semantic_checkpoint("best_avqi_gap_neg", "neg", metrics)

    def on_test_start(self):
        save_enhanced = self.config.get("save_enhanced")
        if not save_enhanced:
            return
        save_dir = Path(save_enhanced)
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        for scp_name in ("inf.scp", "ref.scp"):
            (save_dir / scp_name).unlink(missing_ok=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hybrid_stage"] = self.stage
        checkpoint["hybrid_architecture_config"] = self.architecture_config
        checkpoint["hybrid_lm_objective_json"] = self.lm_objective_json
        checkpoint["hybrid_lm_objective_sha256"] = self.lm_objective_sha256
        checkpoint["latest_avqi_gap_to_clean"] = self._latest_avqi_gap_to_clean
        checkpoint["latest_avqi_metrics"] = self._latest_avqi_metrics
        checkpoint["best_avqi_gap_abs_to_clean"] = self._best_avqi_gap_abs_to_clean
        checkpoint["best_avqi_gap_pos_to_clean"] = self._best_avqi_gap_pos_to_clean
        checkpoint["best_avqi_gap_neg_to_clean"] = self._best_avqi_gap_neg_to_clean
        checkpoint["best_avqi_gap_checkpoints"] = dict(self._best_avqi_gap_checkpoints)
        if getattr(self, "stage_init_checkpoint", None):
            checkpoint["hybrid_stage_init_checkpoint"] = str(self.stage_init_checkpoint)
            checkpoint["hybrid_stage_init_source_stage"] = getattr(self, "stage_init_source_stage", None)
        if getattr(self, "component_init_checkpoints", None):
            checkpoint["hybrid_component_init_checkpoints"] = dict(self.component_init_checkpoints)

    def on_load_checkpoint(self, checkpoint):
        validate_hybrid_checkpoint_metadata(checkpoint, self.stage)
        validate_hybrid_architecture_metadata(checkpoint, self.architecture_config)
        validate_checkpoint_lm_objective_identity(
            checkpoint,
            self.lm_objective_json,
            self.lm_objective_sha256,
        )
        self._latest_avqi_gap_to_clean = checkpoint.get("latest_avqi_gap_to_clean")
        self._latest_avqi_metrics = checkpoint.get("latest_avqi_metrics")
        self._best_avqi_gap_abs_to_clean = float(checkpoint.get("best_avqi_gap_abs_to_clean", float("inf")))
        self._best_avqi_gap_pos_to_clean = float(checkpoint.get("best_avqi_gap_pos_to_clean", float("inf")))
        self._best_avqi_gap_neg_to_clean = float(checkpoint.get("best_avqi_gap_neg_to_clean", float("-inf")))
        self._best_avqi_gap_checkpoints = dict(checkpoint.get("best_avqi_gap_checkpoints") or {})

    def test_step(self, batch, batch_idx):
        data = self._normalize_batch(batch, test=True)
        output = self.enhance(
            data["degraded_wav"],
            data["sample_rate"],
            return_intermediates=self.stage != "disc" or bool(self.config.get("save_intermediates", False)),
        )
        self._validate_output_finite(output)
        if "save_enhanced" not in self.config or self.config["save_enhanced"] is None:
            return
        save_dir = Path(self.config["save_enhanced"])
        final_dir = save_dir / "wav"
        final_dir.mkdir(parents=True, exist_ok=True)
        disc_dir = save_dir / "disc"
        gen_dir = save_dir / "gen"
        if bool(self.config.get("save_intermediates", False)):
            disc_dir.mkdir(parents=True, exist_ok=True)
            gen_dir.mkdir(parents=True, exist_ok=True)
        sr = self._sample_rate_int(data["sample_rate"])
        names = data["utterance_id"] or [f"sample_{batch_idx}"]
        final_path = final_dir / f"{names[0]}.wav"
        sf.write(final_path, output.final_wav[0].detach().cpu().numpy(), samplerate=sr)
        with (save_dir / "inf.scp").open("a") as handle:
            handle.write(f"{names[0]} {final_path}\n")
        ref_path = None
        clean_paths = data.get("clean_path")
        if clean_paths:
            ref_path = clean_paths[0]
        elif data["clean_wav"] is not None:
            ref_dir = save_dir / "ref"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_path = ref_dir / f"{names[0]}.wav"
            sf.write(ref_path, data["clean_wav"][0].detach().cpu().numpy(), samplerate=sr)
        if ref_path is not None:
            with (save_dir / "ref.scp").open("a") as handle:
                handle.write(f"{names[0]} {ref_path}\n")
        if bool(self.config.get("save_intermediates", False)):
            if output.disc_wav is not None:
                sf.write(disc_dir / f"{names[0]}.wav", output.disc_wav[0].detach().cpu().numpy(), samplerate=sr)
            if output.gen_wav is not None:
                sf.write(gen_dir / f"{names[0]}.wav", output.gen_wav[0].detach().cpu().numpy(), samplerate=sr)

    def configure_optimizers(self):
        opt_cfg = self.config.get("opt", {"lr": 2.0e-4})
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, **opt_cfg)
        sch_cfg = self.config.get("sch")
        if not sch_cfg:
            return optimizer

        def warmup_lambda(step: int) -> float:
            warmup_steps = int(sch_cfg.get("warmup_steps", 0))
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), 1e-6)
            return float(sch_cfg.get("decay", 1.0)) ** max(0, step - warmup_steps)

        return [optimizer], [{"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda), "interval": "step"}]
