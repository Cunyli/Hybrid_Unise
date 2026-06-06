from dataclasses import dataclass
from typing import Iterable, Union

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SFIConfig:
    """Sample-frequency-independent STFT settings.

    The physical 20 ms / 10 ms defaults come from the paper. Choosing
    n_fft=win_length is an implementation choice that keeps bin spacing at
    roughly 1/window_seconds across sample rates without silent 16 kHz
    resampling.
    """

    window_ms: float = 20.0
    hop_ms: float = 10.0
    supported_sample_rates: tuple[int, ...] = (8000, 16000, 24000, 32000, 48000)


@dataclass(frozen=True)
class SFIParams:
    sample_rate: int
    win_length: int
    hop_length: int
    n_fft: int
    n_bins: int


def _as_int_sample_rate(sample_rate: Union[int, torch.Tensor]) -> int:
    if torch.is_tensor(sample_rate):
        values = sample_rate.detach().cpu().flatten().unique()
        if values.numel() != 1:
            raise ValueError("SFI STFT expects one sample rate per batch; bucket mixed-rate data first.")
        return int(values.item())
    return int(sample_rate)


def stft_params(sample_rate: Union[int, torch.Tensor], config: SFIConfig) -> SFIParams:
    sr = _as_int_sample_rate(sample_rate)
    if config.supported_sample_rates and sr not in set(config.supported_sample_rates):
        raise ValueError(f"Unsupported sample_rate={sr}; supported={config.supported_sample_rates}")
    win_length = max(2, int(round(sr * config.window_ms / 1000.0)))
    hop_length = max(1, int(round(sr * config.hop_ms / 1000.0)))
    n_fft = win_length
    return SFIParams(
        sample_rate=sr,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft,
        n_bins=n_fft // 2 + 1,
    )


def _hann_window(params: SFIParams, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.hann_window(params.win_length, periodic=True, device=device, dtype=dtype)


def sfi_stft(
    wav: torch.Tensor,
    sample_rate: Union[int, torch.Tensor],
    config: SFIConfig,
) -> tuple[torch.Tensor, SFIParams]:
    if wav.ndim != 2:
        raise ValueError(f"wav must be shaped (B, T), got {tuple(wav.shape)}")
    params = stft_params(sample_rate, config)
    wav = _pad_min_length(wav, params.n_fft)
    spec = torch.stft(
        wav,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        win_length=params.win_length,
        window=_hann_window(params, wav.device, wav.dtype),
        center=True,
        return_complex=True,
    )
    return spec, params


def _pad_min_length(wav: torch.Tensor, min_length: int) -> torch.Tensor:
    if wav.size(-1) >= min_length:
        return wav
    return F.pad(wav, (0, min_length - wav.size(-1)))


def sfi_istft(
    spec: torch.Tensor,
    sample_rate: Union[int, torch.Tensor],
    config: SFIConfig,
    length: int,
) -> torch.Tensor:
    params = stft_params(sample_rate, config)
    return torch.istft(
        spec,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        win_length=params.win_length,
        window=_hann_window(params, spec.device, spec.real.dtype),
        center=True,
        length=int(length),
    )


def align_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.size(-1) > length:
        return wav[..., :length]
    if wav.size(-1) < length:
        return F.pad(wav, (0, length - wav.size(-1)))
    return wav


def linear_resample(
    wav: torch.Tensor,
    orig_sr: Union[int, torch.Tensor],
    target_sr: Union[int, torch.Tensor],
) -> torch.Tensor:
    """Dependency-light waveform resampling.

    This is an implementation choice for reproducible CPU tests and bootstrap
    training. Production runs can replace this utility with a higher quality
    backend at the wrapper level.
    """

    sr = _as_int_sample_rate(orig_sr)
    target_sr = _as_int_sample_rate(target_sr)
    if sr == target_sr:
        return wav
    target_len = max(1, int(round(wav.size(-1) * target_sr / sr)))
    return F.interpolate(wav.unsqueeze(1), size=target_len, mode="linear", align_corners=False).squeeze(1)


def supported_param_table(sample_rates: Iterable[int], config: SFIConfig) -> dict[int, SFIParams]:
    return {int(sr): stft_params(int(sr), config) for sr in sample_rates}
