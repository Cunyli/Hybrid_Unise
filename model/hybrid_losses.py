from dataclasses import dataclass
import importlib
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes: Iterable[int] = (256, 512, 1024), hop_ratio: float = 0.25):
        super().__init__()
        self.fft_sizes = tuple(int(x) for x in fft_sizes)
        self.hop_ratio = float(hop_ratio)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for n_fft in self.fft_sizes:
            if estimate.size(-1) < n_fft:
                estimate_i = F.pad(estimate, (0, n_fft - estimate.size(-1)))
            else:
                estimate_i = estimate
            if target.size(-1) < n_fft:
                target_i = F.pad(target, (0, n_fft - target.size(-1)))
            else:
                target_i = target
            win_length = n_fft
            hop_length = max(1, int(round(n_fft * self.hop_ratio)))
            window = torch.hann_window(win_length, device=estimate_i.device, dtype=estimate_i.dtype)
            est = torch.stft(
                estimate_i,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            tgt = torch.stft(
                target_i,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            losses.append(F.l1_loss(est.abs(), tgt.abs()))
        return torch.stack(losses).mean()


def complex_mse(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.view_as_real(estimate), torch.view_as_real(target))


def magnitude_mse(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(estimate.abs(), target.abs())


@dataclass(frozen=True)
class DisabledExternalLoss:
    name: str

    def __call__(self, *_args, **_kwargs) -> torch.Tensor:
        raise RuntimeError(
            f"{self.name} is disabled: no verified implementation/weights are configured. "
            "Do not replace it with a surrogate loss without marking that as an implementation choice."
        )


class ExternalLossAdapter(nn.Module):
    """Adapter for verified external loss implementations.

    Config must provide an import path in `module:callable` form. The callable
    may be either a loss instance or a zero-argument factory returning one. It
    is called as `(estimate, target, sample_rate=sample_rate)`.
    """

    def __init__(self, name: str, import_path: str):
        super().__init__()
        self.name = name
        if isinstance(import_path, dict):
            config = import_path
            import_path = config.get("import_path")
            kwargs = config.get("kwargs", {})
        else:
            kwargs = {}
        self.import_path = import_path
        module_name, sep, attr_name = import_path.partition(":")
        if sep != ":" or not module_name or not attr_name:
            raise ValueError(f"{name} import_path must use 'module:callable', got {import_path!r}")
        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)
        self.loss = target(**kwargs) if isinstance(target, type) else target
        if not callable(self.loss):
            raise TypeError(f"{name} external loss {import_path!r} is not callable")

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        sample_rate: torch.Tensor | int,
    ) -> torch.Tensor:
        value = self.loss(estimate, target, sample_rate=sample_rate)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=estimate.device, dtype=estimate.dtype)
        return value.to(device=estimate.device, dtype=estimate.dtype)


def build_external_loss(name: str, config: dict | None) -> nn.Module | None:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return None
    import_path = config.get("import_path")
    if not import_path:
        raise ValueError(f"{name} loss is enabled but no import_path was provided")
    return ExternalLossAdapter(name=name, import_path={"import_path": str(import_path), "kwargs": config.get("kwargs", {})})
