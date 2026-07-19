import torch
from torch import nn


class TFGridNetBlock(nn.Module):
    """TF-GridNet block with full-band, sub-band, and frame-attention paths.

    The block follows the standard TF-GridNet decomposition used in speech
    separation/enhancement systems: intra-frame full-band modeling over
    frequency bins, sub-band temporal modeling over frames, and a cross-frame
    self-attention path. It keeps this repository's complex-mask interface.
    """

    def __init__(self, channels: int, lstm_hidden: int, attention_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if channels % attention_heads != 0:
            raise ValueError(f"attention_heads={attention_heads} must divide channels={channels}")
        self.intra_norm = nn.GroupNorm(1, channels)
        self.intra_rnn = nn.LSTM(channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.intra_proj = nn.Linear(lstm_hidden * 2, channels)

        self.sub_norm = nn.GroupNorm(1, channels)
        self.sub_rnn = nn.LSTM(channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.sub_proj = nn.Linear(lstm_hidden * 2, channels)

        self.attn_norm = nn.GroupNorm(1, channels)
        self.frame_attn = nn.MultiheadAttention(
            channels,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, freqs, frames = x.shape

        residual = x
        intra = self.intra_norm(x)
        intra = intra.permute(0, 3, 2, 1).reshape(bsz * frames, freqs, channels)
        intra, _ = self.intra_rnn(intra)
        intra = self.intra_proj(intra).reshape(bsz, frames, freqs, channels).permute(0, 3, 2, 1)
        x = residual + self.dropout(intra)

        residual = x
        sub = self.sub_norm(x)
        sub = sub.permute(0, 2, 3, 1).reshape(bsz * freqs, frames, channels)
        sub, _ = self.sub_rnn(sub)
        sub = self.sub_proj(sub).reshape(bsz, freqs, frames, channels).permute(0, 3, 1, 2)
        x = residual + self.dropout(sub)

        residual = x
        frame_tokens = self.attn_norm(x).mean(dim=2).transpose(1, 2)
        attended, _ = self.frame_attn(frame_tokens, frame_tokens, frame_tokens, need_weights=False)
        attended = attended + self.dropout(self.attn_ffn(attended))
        attended = attended.transpose(1, 2).unsqueeze(2).expand(-1, -1, freqs, -1)
        return residual + self.dropout(attended)


class DiscriminativeBranch(nn.Module):
    def __init__(
        self,
        embedding: int = 64,
        lstm_hidden: int = 256,
        num_blocks: int = 8,
        attention_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embedding, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                TFGridNetBlock(
                    embedding,
                    lstm_hidden,
                    attention_heads=attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
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
        complex_mask = torch.complex(mask[:, 0].float(), mask[:, 1].float()).to(dtype=degraded_spec.dtype)
        return degraded_spec * complex_mask
