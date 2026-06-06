from dataclasses import dataclass
import torch
from torch import nn
import importlib

from .audio import SFIConfig, sfi_stft


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
        backend_kwargs: dict | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.backend = backend
        self.model_path = model_path
        self.sample_rate = int(sample_rate)
        self.rvq_axis = int(rvq_axis)
        self.backend_kwargs = backend_kwargs or {}
        self.sfi_config = SFIConfig(window_ms=20.0, hop_ms=10.0, supported_sample_rates=(self.sample_rate,))
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

    def _select_first_rvq(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim != 3:
            raise ValueError(f"X-Codec {label} must be shaped [B,T] or a 3D RVQ tensor, got {tuple(tensor.shape)}")
        axis = self.rvq_axis if self.rvq_axis >= 0 else tensor.ndim + self.rvq_axis
        if axis <= 0 or axis >= tensor.ndim:
            raise ValueError(f"xcodec.rvq_axis must select a non-batch RVQ dimension, got {self.rvq_axis}")
        return tensor.select(dim=axis, index=0)

    def _mask_from_lengths(
        self,
        token_count: int,
        lengths_16k: torch.Tensor | None,
        device: torch.device,
    ) -> torch.BoolTensor:
        if lengths_16k is None:
            return torch.ones(1, token_count, dtype=torch.bool, device=device)
        lengths = lengths_16k.to(device=device, dtype=torch.long).flatten()
        token_lengths = []
        for length in lengths.tolist():
            probe = torch.zeros(1, int(length), device=device)
            spec, _ = sfi_stft(probe, self.sample_rate, self.sfi_config)
            token_lengths.append(spec.size(-1))
        token_lengths = torch.tensor(token_lengths, device=device, dtype=torch.long)
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
