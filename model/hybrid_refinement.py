import torch
from torch import nn


class DPRNNCrossAttentionBlock(nn.Module):
    """DPRNN-inspired dual-path block with LM cross-attention.

    Block count, channels and normalization are implementation choices because
    the short paper does not publish the full DPRNN specification.
    """

    def __init__(self, channels: int, hidden: int, lm_hidden: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.time_rnn = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.time_proj = nn.Linear(hidden * 2, channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.lm_proj = nn.Linear(lm_hidden, channels)
        self.freq_rnn = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.freq_proj = nn.Linear(hidden * 2, channels)
        self.norm = nn.GroupNorm(1, channels)

    def forward(
        self,
        x: torch.Tensor,
        lm_hidden: torch.Tensor,
        lm_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, channels, freqs, frames = x.shape
        residual = x
        x = self.norm(x)
        time_in = x.permute(0, 2, 3, 1).reshape(bsz * freqs, frames, channels)
        time_out, _ = self.time_rnn(time_in)
        time_out = self.time_proj(time_out).reshape(bsz, freqs, frames, channels).permute(0, 3, 1, 2)
        x = residual + time_out

        query = x.mean(dim=2).transpose(1, 2)
        key_value = self.lm_proj(lm_hidden)
        attended, _ = self.cross_attn(
            query,
            key_value,
            key_value,
            key_padding_mask=lm_padding_mask,
            need_weights=False,
        )
        x = x + attended.transpose(1, 2).unsqueeze(2)

        residual = x
        freq_in = x.permute(0, 3, 2, 1).reshape(bsz * frames, freqs, channels)
        freq_out, _ = self.freq_rnn(freq_in)
        freq_out = self.freq_proj(freq_out).reshape(bsz, frames, freqs, channels).permute(0, 3, 2, 1)
        return residual + freq_out


class GenerativeRefinementBranch(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        hidden: int = 128,
        num_blocks: int = 4,
        lm_hidden: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=(2, 1), padding=1),
            nn.PReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                DPRNNCrossAttentionBlock(
                    channels=channels,
                    hidden=hidden,
                    lm_hidden=lm_hidden,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.PReLU(),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

    def forward(
        self,
        degraded_spec_16k: torch.Tensor,
        lm_hidden: torch.Tensor,
        lm_hidden_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = torch.stack(
            [degraded_spec_16k.abs(), degraded_spec_16k.real, degraded_spec_16k.imag],
            dim=1,
        )
        x = self.encoder(features)
        lm_padding_mask = None if lm_hidden_mask is None else ~lm_hidden_mask.bool()
        for block in self.blocks:
            x = block(x, lm_hidden, lm_padding_mask=lm_padding_mask)
        mask = torch.tanh(self.decoder(x))
        mask = mask[:, :, : degraded_spec_16k.size(1), : degraded_spec_16k.size(2)]
        complex_mask = torch.complex(mask[:, 0], mask[:, 1])
        return degraded_spec_16k * complex_mask
