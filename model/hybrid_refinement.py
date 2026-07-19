import torch
from torch import nn
import torch.nn.functional as F


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
        complex_mask = torch.complex(mask[:, 0].float(), mask[:, 1].float()).to(dtype=degraded_spec_16k.dtype)
        return degraded_spec_16k * complex_mask


class PaperDPRNNCrossAttentionBlock(nn.Module):
    """Chunked DPRNN block with LM cross-attention.

    This follows the paper's stated refinement order more closely than the
    legacy full-sequence block: inter/intra dual-path recurrence is performed
    on overlapping STFT-frame chunks, then the time path attends to the last LM
    hidden states as K/V.
    """

    def __init__(
        self,
        channels: int,
        hidden: int,
        lm_hidden: int,
        num_heads: int,
        chunk_size: int,
        hop_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.hop_size = int(hop_size)
        self.norm = nn.GroupNorm(1, channels)
        self.intra_rnn = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.intra_proj = nn.Linear(hidden * 2, channels)
        self.inter_rnn = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.inter_proj = nn.Linear(hidden * 2, channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.lm_proj = nn.Linear(lm_hidden, channels)
        self.out_norm = nn.GroupNorm(1, channels)

    def _segment(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        frames = x.size(-1)
        if frames <= self.chunk_size:
            total_frames = self.chunk_size
        else:
            num_chunks = (frames - self.chunk_size + self.hop_size - 1) // self.hop_size + 1
            total_frames = self.chunk_size + (num_chunks - 1) * self.hop_size
        pad_frames = total_frames - frames
        padded = F.pad(x, (0, pad_frames))
        chunks = padded.unfold(dimension=-1, size=self.chunk_size, step=self.hop_size)
        return chunks.contiguous(), pad_frames

    def _overlap_add(self, chunks: torch.Tensor, pad_frames: int) -> torch.Tensor:
        bsz, channels, freqs, num_chunks, chunk_size = chunks.shape
        total_frames = chunk_size + (num_chunks - 1) * self.hop_size
        output = chunks.new_zeros(bsz, channels, freqs, total_frames)
        counts = chunks.new_zeros(1, 1, 1, total_frames)
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.hop_size
            stop = start + chunk_size
            output[..., start:stop] = output[..., start:stop] + chunks[..., chunk_idx, :]
            counts[..., start:stop] = counts[..., start:stop] + 1.0
        output = output / counts.clamp_min(1.0)
        if pad_frames > 0:
            output = output[..., :-pad_frames]
        return output

    def _dual_path(self, x: torch.Tensor) -> torch.Tensor:
        chunks, pad_frames = self._segment(x)
        bsz, channels, freqs, num_chunks, chunk_size = chunks.shape

        intra_in = chunks.permute(0, 2, 3, 4, 1).reshape(bsz * freqs * num_chunks, chunk_size, channels)
        intra_out, _ = self.intra_rnn(intra_in)
        intra_out = self.intra_proj(intra_out)
        chunks = chunks + intra_out.reshape(bsz, freqs, num_chunks, chunk_size, channels).permute(0, 4, 1, 2, 3)

        inter_in = chunks.permute(0, 2, 4, 3, 1).reshape(bsz * freqs * chunk_size, num_chunks, channels)
        inter_out, _ = self.inter_rnn(inter_in)
        inter_out = self.inter_proj(inter_out)
        chunks = chunks + inter_out.reshape(bsz, freqs, chunk_size, num_chunks, channels).permute(0, 4, 1, 3, 2)
        return self._overlap_add(chunks, pad_frames)

    def forward(
        self,
        x: torch.Tensor,
        lm_hidden: torch.Tensor,
        lm_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = residual + self._dual_path(x)

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
        return self.out_norm(x)


class PaperDPRNNRefinementBranch(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        hidden: int = 128,
        num_blocks: int = 4,
        lm_hidden: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        dprnn_window_length: int = 640,
        dprnn_hop_length: int = 320,
        stft_hop_length: int = 160,
        dprnn_window_frames: int | None = None,
        dprnn_hop_frames: int | None = None,
    ):
        super().__init__()
        chunk_size = int(dprnn_window_frames or round(dprnn_window_length / stft_hop_length))
        hop_size = int(dprnn_hop_frames or round(dprnn_hop_length / stft_hop_length))
        self.chunk_size = max(1, chunk_size)
        self.hop_size = max(1, min(hop_size, self.chunk_size))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=(2, 1), padding=1),
            nn.PReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                PaperDPRNNCrossAttentionBlock(
                    channels=channels,
                    hidden=hidden,
                    lm_hidden=lm_hidden,
                    num_heads=num_heads,
                    chunk_size=self.chunk_size,
                    hop_size=self.hop_size,
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
        complex_mask = torch.complex(mask[:, 0].float(), mask[:, 1].float()).to(dtype=degraded_spec_16k.dtype)
        return degraded_spec_16k * complex_mask


class IdentityCenteredPaperDPRNNRefinementBranch(PaperDPRNNRefinementBranch):
    """Paper-DPRNN refinement with an identity-centered complex residual mask.

    The original paper-DPRNN branch predicts a full complex mask and multiplies
    it into the degraded spectrum. That makes the no-op solution depend on
    saturating the real mask near one. This variant instead predicts a residual
    complex mask around identity, so a zero decoder output preserves the input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_head = self.decoder[-1]
        if not isinstance(output_head, nn.Conv2d):
            raise TypeError("Identity-centered refinement requires a Conv2d head")
        nn.init.zeros_(output_head.weight)
        if output_head.bias is not None:
            nn.init.zeros_(output_head.bias)

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
        residual_mask = torch.tanh(self.decoder(x))
        residual_mask = residual_mask[:, :, : degraded_spec_16k.size(1), : degraded_spec_16k.size(2)]
        complex_residual = torch.complex(
            residual_mask[:, 0].float(),
            residual_mask[:, 1].float(),
        ).to(dtype=degraded_spec_16k.dtype)
        return degraded_spec_16k * (1.0 + complex_residual)
