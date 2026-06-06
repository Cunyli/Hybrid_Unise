from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AsteroidPMSQELoss(nn.Module):
    """Adapter for Asteroid's PMSQE implementation.

    PMSQE is a verified public perceptual speech quality loss. This wrapper is
    intentionally optional and lazy-imported so the main Hybrid-UniSE path does
    not depend on Asteroid unless the config enables PMSQE.
    """

    def __init__(self, sample_rate: int = 16000, **kwargs):
        super().__init__()
        if int(sample_rate) not in {8000, 16000}:
            raise ValueError("Asteroid PMSQE adapter supports 8 kHz or 16 kHz audio")
        self.sample_rate = int(sample_rate)
        self.n_fft = 512 if self.sample_rate == 16000 else 256
        self.hop_length = self.n_fft // 2
        self.win_length = self.n_fft
        try:
            from asteroid.losses import SingleSrcPMSQE
        except Exception as exc:
            raise ImportError(
                "Asteroid PMSQE is not importable. Install/repair asteroid or "
                "disable external_losses.pmsqe.enabled."
            ) from exc
        self.loss = SingleSrcPMSQE(sample_rate=self.sample_rate, **kwargs)

    def _power_spectrum(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim != 2:
            raise ValueError(f"AsteroidPMSQELoss expects waveform [B,T], got {tuple(wav.shape)}")
        if wav.size(-1) < self.n_fft:
            wav = torch.nn.functional.pad(wav, (0, self.n_fft - wav.size(-1)))
        window = torch.hann_window(self.win_length, device=wav.device, dtype=wav.dtype).sqrt()
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        return spec.abs().square().transpose(1, 2)

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        sample_rate: torch.Tensor | int = 16000,
    ) -> torch.Tensor:
        sr = int(sample_rate.item()) if torch.is_tensor(sample_rate) else int(sample_rate)
        if sr != self.sample_rate:
            raise ValueError(f"AsteroidPMSQELoss was configured for {self.sample_rate} Hz, got {sr}")
        value = self.loss(self._power_spectrum(estimate), self._power_spectrum(target))
        return value.mean() if value.ndim > 0 else value


class TorchaudioSquimObjectiveLoss(nn.Module):
    """SQA baseline using TorchAudio's frozen SQUIM objective model.

    This is an implementation choice, not the paper's full MOS/DNSMOS/ScoreQ/
    UTMOS/NISQA SQA ensemble. It encourages the enhanced waveform to match the
    clean waveform's predicted objective-quality vector.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mode: str = "l1",
        weights: list[float] | tuple[float, float, float] = (1.0, 1.0, 0.05),
    ):
        super().__init__()
        if int(sample_rate) != 16000:
            raise ValueError("TorchAudio SQUIM objective bundle expects 16 kHz audio")
        try:
            import torchaudio
        except Exception as exc:
            raise ImportError("torchaudio is required for TorchaudioSquimObjectiveLoss") from exc
        self.sample_rate = int(sample_rate)
        self.mode = mode
        self.model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        weights_tensor = torch.tensor(list(weights), dtype=torch.float32)
        if weights_tensor.numel() != 3:
            raise ValueError("TorchaudioSquimObjectiveLoss weights must contain 3 values")
        self.register_buffer("weights", weights_tensor)

    def _scores(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim != 2:
            raise ValueError(f"TorchaudioSquimObjectiveLoss expects waveform [B,T], got {tuple(wav.shape)}")
        outputs = self.model(wav)
        if len(outputs) != 3:
            raise RuntimeError(f"SQUIM objective returned {len(outputs)} outputs, expected 3")
        return torch.stack([output.float() for output in outputs], dim=-1)

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        sample_rate: torch.Tensor | int = 16000,
    ) -> torch.Tensor:
        sr = int(sample_rate.item()) if torch.is_tensor(sample_rate) else int(sample_rate)
        if sr != self.sample_rate:
            raise ValueError(f"TorchaudioSquimObjectiveLoss expects {self.sample_rate} Hz audio, got {sr}")
        estimate_scores = self._scores(estimate)
        with torch.no_grad():
            target_scores = self._scores(target)
        weights = self.weights.to(device=estimate_scores.device, dtype=estimate_scores.dtype)
        if self.mode == "l1":
            return (torch.abs(estimate_scores - target_scores) * weights).mean()
        if self.mode == "mse":
            return (F.mse_loss(estimate_scores, target_scores, reduction="none") * weights).mean()
        raise ValueError("TorchaudioSquimObjectiveLoss mode must be 'l1' or 'mse'")
