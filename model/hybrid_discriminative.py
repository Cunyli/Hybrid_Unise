import torch
from torch import nn


class TFGridNetBlock(nn.Module):
    """Lightweight TF-GridNet-style block.

    This preserves the paper-facing interface and time/frequency recurrent
    structure. The exact TF-GridNet implementation is an implementation choice
    because this repository does not contain the original module.
    """

    def __init__(self, channels: int, lstm_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.time_rnn = nn.LSTM(channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.time_proj = nn.Linear(lstm_hidden * 2, channels)
        self.freq_rnn = nn.LSTM(channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.freq_proj = nn.Linear(lstm_hidden * 2, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        bsz, channels, freqs, frames = x.shape

        time_in = x.permute(0, 2, 3, 1).reshape(bsz * freqs, frames, channels)
        time_out, _ = self.time_rnn(time_in)
        time_out = self.time_proj(time_out).reshape(bsz, freqs, frames, channels).permute(0, 3, 1, 2)
        x = residual + self.dropout(time_out)

        residual = x
        freq_in = x.permute(0, 3, 2, 1).reshape(bsz * frames, freqs, channels)
        freq_out, _ = self.freq_rnn(freq_in)
        freq_out = self.freq_proj(freq_out).reshape(bsz, frames, freqs, channels).permute(0, 3, 2, 1)
        return residual + self.dropout(freq_out)


class DiscriminativeBranch(nn.Module):
    def __init__(
        self,
        embedding: int = 64,
        lstm_hidden: int = 256,
        num_blocks: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embedding, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.blocks = nn.ModuleList(
            [TFGridNetBlock(embedding, lstm_hidden, dropout=dropout) for _ in range(num_blocks)]
        )
        self.mask_head = nn.Conv2d(embedding, 2, kernel_size=1)

    def forward(self, degraded_spec: torch.Tensor) -> torch.Tensor:
        features = torch.stack(
            [degraded_spec.real, degraded_spec.imag, degraded_spec.abs()],
            dim=1,
        )
        x = self.encoder(features)
        for block in self.blocks:
            x = block(x)
        mask = torch.tanh(self.mask_head(x))
        complex_mask = torch.complex(mask[:, 0], mask[:, 1])
        return degraded_spec * complex_mask

