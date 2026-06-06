import torch
from torch import nn


def blend_spectra(disc_spec: torch.Tensor, gen_spec: torch.Tensor, fusion_mask: torch.Tensor) -> torch.Tensor:
    return fusion_mask * disc_spec + (1.0 - fusion_mask) * gen_spec


class FusionBranch(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, disc_spec: torch.Tensor, gen_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.stack(
            [
                disc_spec.real,
                disc_spec.imag,
                disc_spec.abs(),
                gen_spec.real,
                gen_spec.imag,
                gen_spec.abs(),
            ],
            dim=1,
        )
        mask = torch.sigmoid(self.net(features)).squeeze(1)
        return blend_spectra(disc_spec, gen_spec, mask), mask

