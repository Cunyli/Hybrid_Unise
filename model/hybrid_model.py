from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F

from .audio import SFIConfig, align_length, linear_resample, sfi_istft, sfi_stft
from .hybrid_discriminative import DiscriminativeBranch
from .hybrid_fusion import FusionBranch
from .hybrid_lm import HybridSemanticLM, WavLMConditioner
from .hybrid_losses import MultiResolutionSTFTLoss, build_external_loss, complex_mse, magnitude_mse
from .hybrid_refinement import GenerativeRefinementBranch
from .hybrid_types import HybridOutput
from .hybrid_xcodec import XCodecFirstRVQTokenizer


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
) -> None:
    checkpoint_architecture = checkpoint.get("hybrid_architecture_config")
    if checkpoint_architecture is None:
        return
    comparable_checkpoint_architecture = {
        key: checkpoint_architecture.get(key)
        for key in expected_architecture
    }
    if comparable_checkpoint_architecture != expected_architecture:
        raise ValueError("Checkpoint hybrid architecture config does not match current config.")


def _checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    return checkpoint.get("state_dict", checkpoint)


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
        )
        self.xcodec = XCodecFirstRVQTokenizer(
            vocab_size=vocab_size,
            backend=str(xcodec_config.get("backend", "deterministic_stub")),
            model_path=xcodec_config.get("model_path"),
            rvq_axis=int(xcodec_config.get("rvq_axis", -1)),
            backend_kwargs=xcodec_config.get("backend_kwargs", {}),
        )
        self.lm = HybridSemanticLM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=int(lm_config.get("num_layers", 12)),
            num_attention_heads=int(lm_config.get("num_attention_heads", 8)),
            dropout=float(lm_config.get("dropout", 0.1)),
            max_position_embeddings=int(lm_config.get("max_position_embeddings", 4096)),
            label_smoothing=float(lm_config.get("label_smoothing", 0.0)),
        )

        refinement_config = config.get("refinement", {})
        self.refinement = GenerativeRefinementBranch(
            channels=int(refinement_config.get("channels", 64)),
            hidden=int(refinement_config.get("hidden", 128)),
            num_blocks=int(refinement_config.get("num_blocks", 4)),
            lm_hidden=hidden_size,
            num_heads=int(refinement_config.get("num_heads", 8)),
            dropout=float(refinement_config.get("dropout", 0.0)),
        )

        fusion_config = config.get("fusion", {})
        self.fusion = FusionBranch(channels=int(fusion_config.get("channels", 32)))
        self.mrstft_loss = MultiResolutionSTFTLoss(**config.get("mrstft_loss", {}))
        self.loss_weights = config.get("loss_weights", {})
        external_losses = config.get("external_losses", {})
        self.pmsqe_loss = build_external_loss("PMSQE", external_losses.get("pmsqe"))
        self.sqa_loss = build_external_loss("SQA", external_losses.get("sqa"))
        self.fusion_use_teacher_forcing = bool(config.get("fusion_use_teacher_forcing", False))
        self.resampling_backend = str(config.get("resampling", {}).get("backend", "linear"))
        if self.resampling_backend != "linear":
            raise ValueError(
                "Only resampling.backend='linear' is implemented in this repository. "
                "Wire a verified backend before selecting another value."
            )
        self.lm_stft_alignment_mode = str(config.get("lm_stft_alignment_mode", "interpolate"))
        if self.lm_stft_alignment_mode not in {"interpolate", "strict"}:
            raise ValueError("lm_stft_alignment_mode must be 'interpolate' or 'strict'")
        self.stage_init_checkpoint = config.get("stage_init_checkpoint")
        if self.stage_init_checkpoint:
            stage_init_path = Path(str(self.stage_init_checkpoint)).expanduser()
            if not stage_init_path.is_absolute():
                stage_init_path = Path(str(config.get("_config_dir", "."))).expanduser() / stage_init_path
            self.stage_init_checkpoint = str(stage_init_path)
            self._load_stage_initialization_checkpoint(self.stage_init_checkpoint)
        self._apply_stage_freezing()

    def _load_stage_initialization_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = load_hybrid_checkpoint(checkpoint_path, map_location="cpu")
        validate_hybrid_architecture_metadata(checkpoint, self.architecture_config)
        state_dict = _checkpoint_state_dict(checkpoint)
        load_result = self.load_state_dict(state_dict, strict=False)
        self.stage_init_source_stage = checkpoint.get("hybrid_stage")
        self.stage_init_missing_keys = list(load_result.missing_keys)
        self.stage_init_unexpected_keys = list(load_result.unexpected_keys)

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
            trainable = {self.conditioner.adapter, self.lm, self.refinement}
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
            for module in (self.discriminative, self.fusion):
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

        hidden = F.interpolate(
            lm_hidden.transpose(1, 2),
            size=stft_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        if lm_hidden_mask is None:
            return hidden, None
        mask = F.interpolate(
            lm_hidden_mask.float().unsqueeze(1),
            size=stft_frames,
            mode="nearest",
        ).squeeze(1).bool()
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
            prefix = self.conditioner(wav_16k)

            if clean_wav is not None:
                clean_16k = linear_resample(clean_wav, sample_rate, 16000)
                lengths_16k = self._lengths_at_16k(
                    length,
                    clean_16k.size(0),
                    clean_16k.size(-1),
                    sample_rate,
                    clean_16k.device,
                )
                token_batch = self.xcodec.encode_first_rvq_batch(clean_16k, lengths_16k=lengths_16k)
                lm_out = self.lm(prefix, token_batch.tokens, target_mask=token_batch.mask)
            else:
                max_tokens = degraded_spec_16k.size(-1)
                lm_out = self.lm.generate(prefix, max_tokens=max_tokens, do_sample=do_sample)
                lm_out = {
                    "loss": None,
                    "accuracy": None,
                    "logits": None,
                    "targets": lm_out["tokens"],
                    "hidden_states": lm_out["hidden_states"],
                    "hidden_mask": lm_out["hidden_mask"],
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
        disc_wav, degraded_spec, disc_spec = self._disc(degraded_wav, sample_rate)

        gen_wav = gen_wav_16k = gen_spec = gen_spec_16k = fusion_mask = None
        token_logits = token_targets = lm_hidden_states = None
        token_nll = None
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
            lm_hidden_states = lm_out["hidden_states"]
            lm_hidden_mask = lm_out.get("hidden_mask")
            aligned_lm_hidden_states = lm_out.get("aligned_hidden_states")
            aligned_lm_hidden_mask = lm_out.get("aligned_hidden_mask")

        if self.stage in {"fusion", "joint"} and gen_spec is not None:
            fusion_disc_spec = disc_spec.detach() if self.stage == "fusion" else disc_spec
            fusion_gen_spec = gen_spec.detach() if self.stage == "fusion" else gen_spec
            final_spec, fusion_mask = self.fusion(fusion_disc_spec, fusion_gen_spec)
            final_wav = sfi_istft(final_spec, sample_rate, self.sfi, length=degraded_wav.size(-1))
        else:
            final_spec = disc_spec if self.stage == "disc" else gen_spec
            final_wav = disc_wav if self.stage == "disc" else gen_wav

        return HybridOutput(
            final_wav=align_length(final_wav, degraded_wav.size(-1)),
            disc_wav=align_length(disc_wav, degraded_wav.size(-1)),
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
            length=torch.tensor([wav.size(-1)], device=self.device),
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

        losses["disc_mrstft"] = self.mrstft_loss(output.disc_wav, clean)
        if output.gen_spec_16k is not None and output.gen_wav_16k is not None:
            clean_16k = linear_resample(clean, sample_rate, 16000)
            clean_spec_16k, _ = sfi_stft(clean_16k, 16000, self.gen_sfi)
            losses["complex"] = complex_mse(output.gen_spec_16k, clean_spec_16k)
            losses["mag"] = magnitude_mse(output.gen_spec_16k, clean_spec_16k)
            if output.token_nll is not None:
                losses["nll"] = output.token_nll
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
            return (
                float(weights.get("nll", 1.0)) * losses.get("nll", torch.zeros((), device=self.device))
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
            return_intermediates=self.stage != "disc",
        )
        losses = self._losses(output, data["clean_wav"], data["sample_rate"])
        loss = self._weighted_stage_loss(losses)
        self._validate_output_finite(output)
        _require_finite("train/loss", loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = self._normalize_batch(batch)
        output = self.forward(
            data["degraded_wav"],
            data["sample_rate"],
            clean_wav=data["clean_wav"],
            length=data["length"],
            return_intermediates=self.stage != "disc",
        )
        losses = self._losses(output, data["clean_wav"], data["sample_rate"])
        loss = self._weighted_stage_loss(losses)
        self._validate_output_finite(output)
        _require_finite("val/loss", loss)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=False, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, on_step=False, on_epoch=True, sync_dist=True)

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
        if getattr(self, "stage_init_checkpoint", None):
            checkpoint["hybrid_stage_init_checkpoint"] = str(self.stage_init_checkpoint)
            checkpoint["hybrid_stage_init_source_stage"] = getattr(self, "stage_init_source_stage", None)

    def on_load_checkpoint(self, checkpoint):
        validate_hybrid_checkpoint_metadata(checkpoint, self.stage)
        validate_hybrid_architecture_metadata(checkpoint, self.architecture_config)

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
