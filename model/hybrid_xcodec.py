from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from .audio import SFIConfig, sfi_stft
from .wavlm_utils import (
    feature_mask_from_lengths,
    validate_waveform_lengths,
    waveform_attention_mask,
    wavlm_feature_lengths,
)


@dataclass(frozen=True)
class XCodecTokenBatch:
    tokens: torch.LongTensor
    mask: torch.BoolTensor


class XCodecFirstRVQTokenizer(nn.Module):
    """Frozen first-RVQ-layer token interface for X-Codec.

    The deterministic backend is an explicit implementation choice for tests
    and environments where verified X-Codec weights are not present. A real
    backend can be added behind the same encode_first_rvq interface without
    changing the hybrid training path.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        backend: str = "deterministic_stub",
        model_path: str | None = None,
        sample_rate: int = 16000,
        rvq_axis: int = -1,
        rvq_index: int = 0,
        codec_hop_length: int | None = None,
        backend_kwargs: dict | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.backend = backend
        self.model_path = model_path
        self.sample_rate = int(sample_rate)
        self.rvq_axis = int(rvq_axis)
        self.rvq_index = int(rvq_index)
        if self.rvq_index < 0:
            raise ValueError("xcodec.rvq_index must be non-negative")
        self.backend_kwargs = backend_kwargs or {}
        self.sfi_config = SFIConfig(window_ms=20.0, hop_ms=10.0, supported_sample_rates=(self.sample_rate,))
        default_hop = round(self.sample_rate * self.sfi_config.hop_ms / 1000.0) if backend == "deterministic_stub" else 320
        self.codec_hop_length = int(codec_hop_length if codec_hop_length is not None else default_hop)
        if self.codec_hop_length <= 0:
            raise ValueError("xcodec.codec_hop_length must be positive")
        self.backend_impl = None
        if backend != "deterministic_stub":
            self.backend_impl = self._load_backend(backend)

    def _load_backend(self, backend: str):
        module_name, sep, attr_name = backend.partition(":")
        if sep != ":" or not module_name or not attr_name:
            raise ValueError(
                "X-Codec backend must be 'deterministic_stub' or 'module:callable'. "
                f"Got {backend!r}."
            )
        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)
        if isinstance(target, type):
            instance = target(self.model_path, **self.backend_kwargs)
        elif self.backend_kwargs and hasattr(target, "__name__"):
            instance = target(self.model_path, **self.backend_kwargs)
        else:
            instance = target
        if not callable(instance):
            raise TypeError(f"X-Codec backend {backend!r} is not callable")
        return instance

    def _select_rvq(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim != 3:
            raise ValueError(f"X-Codec {label} must be shaped [B,T] or a 3D RVQ tensor, got {tuple(tensor.shape)}")
        axis = self.rvq_axis if self.rvq_axis >= 0 else tensor.ndim + self.rvq_axis
        if axis <= 0 or axis >= tensor.ndim:
            raise ValueError(f"xcodec.rvq_axis must select a non-batch RVQ dimension, got {self.rvq_axis}")
        if self.rvq_index >= tensor.size(axis):
            raise ValueError(
                f"xcodec.rvq_index={self.rvq_index} is out of range for "
                f"X-Codec {label} with {tensor.size(axis)} RVQ layers"
            )
        return tensor.select(dim=axis, index=self.rvq_index)

    def _select_first_rvq(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        return self._select_rvq(tensor, label)

    def token_lengths_from_waveform_lengths(
        self,
        lengths_16k: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        target_device = device if device is not None else lengths_16k.device
        lengths = lengths_16k.to(device=target_device, dtype=torch.long).flatten().clamp_min(1)
        return torch.div(
            lengths + self.codec_hop_length - 1,
            self.codec_hop_length,
            rounding_mode="floor",
        ).clamp_min(1)

    def max_token_count_from_waveform_lengths(self, lengths_16k: torch.Tensor) -> int:
        return int(self.token_lengths_from_waveform_lengths(lengths_16k).max().item())

    def _mask_from_lengths(
        self,
        token_count: int,
        lengths_16k: torch.Tensor | None,
        device: torch.device,
    ) -> torch.BoolTensor:
        if lengths_16k is None:
            return torch.ones(1, token_count, dtype=torch.bool, device=device)
        token_lengths = self.token_lengths_from_waveform_lengths(lengths_16k, device=device)
        token_lengths = token_lengths.clamp(min=1, max=token_count)
        steps = torch.arange(token_count, device=device).unsqueeze(0)
        return steps < token_lengths.unsqueeze(1)

    def _coerce_backend_output(
        self,
        output,
        device: torch.device,
        lengths_16k: torch.Tensor | None,
    ) -> XCodecTokenBatch:
        mask = None
        tokens = output
        if isinstance(output, XCodecTokenBatch):
            return output
        if isinstance(output, dict):
            tokens = output.get("tokens")
            mask = output.get("mask")
            lengths_16k = output.get("lengths", lengths_16k)
        elif isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError("X-Codec backend tuple output must be (tokens, mask)")
            tokens, mask = output
        if tokens is None:
            raise ValueError("X-Codec backend did not return tokens")
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens, device=device)
        tokens = self._select_first_rvq(tokens.to(device=device, dtype=torch.long), "tokens")
        if mask is None:
            mask = self._mask_from_lengths(tokens.size(1), lengths_16k, device).expand(tokens.size(0), -1)
        elif not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=device)
        mask = self._select_first_rvq(mask.to(device=device), "mask").bool()
        if mask.shape != tokens.shape:
            raise ValueError(f"X-Codec token mask shape {tuple(mask.shape)} does not match tokens {tuple(tokens.shape)}")
        if not mask.any(dim=1).all():
            raise ValueError("X-Codec token mask must keep at least one valid token per sample")
        invalid = mask & ((tokens < 0) | (tokens >= self.vocab_size))
        if invalid.any():
            raise ValueError("X-Codec backend returned token IDs outside [0, vocab_size) at valid positions")
        tokens = tokens.masked_fill(~mask, self.vocab_size + 1)
        return XCodecTokenBatch(tokens=tokens, mask=mask)

    @torch.no_grad()
    def encode_first_rvq_batch(
        self,
        clean_wav_16k: torch.Tensor,
        lengths_16k: torch.Tensor | None = None,
    ) -> XCodecTokenBatch:
        if self.backend_impl is not None:
            output = self.backend_impl(clean_wav_16k, sample_rate=self.sample_rate)
            return self._coerce_backend_output(output, clean_wav_16k.device, lengths_16k)

        spec, _ = sfi_stft(clean_wav_16k, self.sample_rate, self.sfi_config)
        frame_energy = torch.log1p(spec.abs().mean(dim=1))
        if frame_energy.size(1) == 0:
            tokens = torch.zeros(clean_wav_16k.size(0), 1, dtype=torch.long, device=clean_wav_16k.device)
            mask = torch.ones_like(tokens, dtype=torch.bool)
            return XCodecTokenBatch(tokens=tokens, mask=mask)
        min_v = frame_energy.amin(dim=1, keepdim=True)
        max_v = frame_energy.amax(dim=1, keepdim=True)
        norm = (frame_energy - min_v) / (max_v - min_v).clamp_min(1e-6)
        tokens = torch.clamp((norm * (self.vocab_size - 1)).round().long(), 0, self.vocab_size - 1)
        mask = self._mask_from_lengths(tokens.size(1), lengths_16k, clean_wav_16k.device).expand(tokens.size(0), -1)
        if not mask.any(dim=1).all():
            raise ValueError("X-Codec token mask must keep at least one valid token per sample")
        tokens = tokens.masked_fill(~mask, self.vocab_size + 1)
        return XCodecTokenBatch(tokens=tokens, mask=mask)

    @torch.no_grad()
    def encode_first_rvq(
        self,
        clean_wav_16k: torch.Tensor,
        lengths_16k: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        return self.encode_first_rvq_batch(clean_wav_16k, lengths_16k=lengths_16k).tokens


class WavLMKMeansVQTokenizer(nn.Module):
    """Experimental frozen WavLM + fixed KMeans semantic token target.

    This is a diagnostic target for checking the Hybrid-UniSE LM contract. It
    deliberately has no decoder/reconstruction path and should be selected
    explicitly with xcodec.target_type=wavlm_kmeans_vq.
    """

    def __init__(
        self,
        vocab_size: int,
        codebook_path: str,
        wavlm_model_path: str,
        sample_rate: int = 16000,
        layer_index: int = 11,
        feature_normalization: str = "standardize_l2",
        codec_hop_length: int = 320,
        centers_key: str = "centers",
        mean_key: str = "mean",
        std_key: str = "std",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.sample_rate = int(sample_rate)
        if self.sample_rate != 16000:
            raise ValueError("wavlm_kmeans_vq currently expects 16 kHz input")
        self.layer_index = int(layer_index)
        self.feature_normalization = str(feature_normalization)
        if self.feature_normalization not in {"none", "l2", "standardize_l2"}:
            raise ValueError("xcodec.feature_normalization must be none, l2, or standardize_l2")
        self.codec_hop_length = int(codec_hop_length)
        if self.codec_hop_length <= 0:
            raise ValueError("xcodec.codec_hop_length must be positive")

        with np.load(Path(codebook_path).expanduser()) as codebook:
            if centers_key not in codebook:
                raise ValueError(f"WavLM KMeans codebook is missing array {centers_key!r}")
            centers = torch.as_tensor(np.asarray(codebook[centers_key], dtype=np.float32))
            if centers.ndim != 2:
                raise ValueError("WavLM KMeans centers must be a 2D array [vocab, feature_dim]")
            if centers.size(0) != self.vocab_size:
                raise ValueError(
                    f"xcodec.vocab_size={self.vocab_size} must match KMeans centers "
                    f"count {centers.size(0)} for wavlm_kmeans_vq"
                )
            mean = self._optional_codebook_vector(codebook, mean_key, centers.size(1))
            std = self._optional_codebook_vector(codebook, std_key, centers.size(1))
        self.register_buffer("centers", centers)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

        self.encoder = AutoModel.from_pretrained(wavlm_model_path)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _optional_codebook_vector(codebook, key: str, feature_dim: int) -> torch.Tensor:
        if key not in codebook:
            return torch.empty(0, dtype=torch.float32)
        value = torch.as_tensor(np.asarray(codebook[key], dtype=np.float32)).reshape(-1)
        if value.numel() != feature_dim:
            raise ValueError(f"WavLM KMeans {key!r} must have {feature_dim} values")
        return value

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def token_lengths_from_waveform_lengths(
        self,
        lengths_16k: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        target_device = device if device is not None else lengths_16k.device
        lengths = lengths_16k.to(device=target_device, dtype=torch.long).flatten()
        if (lengths <= 0).any():
            raise ValueError("Waveform lengths must be positive")
        return wavlm_feature_lengths(self.encoder, lengths)

    def max_token_count_from_waveform_lengths(self, lengths_16k: torch.Tensor) -> int:
        return int(self.token_lengths_from_waveform_lengths(lengths_16k).max().item())

    def _mask_from_lengths(
        self,
        token_count: int,
        lengths_16k: torch.Tensor | None,
        device: torch.device,
    ) -> torch.BoolTensor:
        if lengths_16k is None:
            return torch.ones(1, token_count, dtype=torch.bool, device=device)
        token_lengths = self.token_lengths_from_waveform_lengths(lengths_16k, device=device)
        return feature_mask_from_lengths(token_count, token_lengths)

    def _select_hidden_layer(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not hidden_states:
            raise ValueError("Pretrained WavLM did not return hidden states")
        index = self.layer_index
        if index < 0:
            index += len(hidden_states)
        if index < 0 or index >= len(hidden_states):
            raise ValueError(
                f"xcodec.layer_index={self.layer_index} is out of range for "
                f"{len(hidden_states)} hidden-state tensors"
            )
        return hidden_states[index].float()

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        values = features
        if self.feature_normalization == "none":
            return values
        if self.feature_normalization == "standardize_l2":
            if self.feature_mean.numel() == 0 or self.feature_std.numel() == 0:
                raise ValueError("standardize_l2 WavLM KMeans codebook requires mean and std arrays")
            values = (values - self.feature_mean.view(1, 1, -1)) / self.feature_std.view(1, 1, -1).clamp_min(1.0e-5)
        if self.feature_normalization in {"l2", "standardize_l2"}:
            values = values / values.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return values

    @torch.no_grad()
    def encode_first_rvq_batch(
        self,
        clean_wav_16k: torch.Tensor,
        lengths_16k: torch.Tensor | None = None,
    ) -> XCodecTokenBatch:
        encoder_kwargs = {"output_hidden_states": True}
        if lengths_16k is not None:
            lengths = validate_waveform_lengths(clean_wav_16k, lengths_16k)
            attention_mask = waveform_attention_mask(clean_wav_16k, lengths)
            if not attention_mask.all():
                encoder_kwargs["attention_mask"] = attention_mask.long()
        outputs = self.encoder(clean_wav_16k, **encoder_kwargs)
        features = self._select_hidden_layer(tuple(outputs.hidden_states)).to(device=clean_wav_16k.device)
        if features.size(-1) != self.centers.size(1):
            raise ValueError(
                f"WavLM feature dim {features.size(-1)} does not match KMeans "
                f"centers dim {self.centers.size(1)}"
            )
        features = self._normalize_features(features)
        distances = torch.cdist(features.float(), self.centers.to(device=features.device).float())
        tokens = distances.argmin(dim=-1).long()
        mask = self._mask_from_lengths(tokens.size(1), lengths_16k, clean_wav_16k.device).expand(tokens.size(0), -1)
        if not mask.any(dim=1).all():
            raise ValueError("WavLM KMeans token mask must keep at least one valid token per sample")
        tokens = tokens.masked_fill(~mask, self.vocab_size + 1)
        return XCodecTokenBatch(tokens=tokens, mask=mask)

    @torch.no_grad()
    def encode_first_rvq(
        self,
        clean_wav_16k: torch.Tensor,
        lengths_16k: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        return self.encode_first_rvq_batch(clean_wav_16k, lengths_16k=lengths_16k).tokens
