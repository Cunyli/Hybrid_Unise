import torch
from torch import nn
import torch.nn.functional as F


class WavLMConditioner(nn.Module):
    def __init__(
        self,
        output_dim: int = 512,
        freeze: bool = True,
        pretrained_name_or_path: str = "microsoft/wavlm-base-plus",
        use_pretrained: bool = False,
        feature_dim: int = 768,
    ):
        super().__init__()
        self.use_pretrained = bool(use_pretrained)
        self.freeze = bool(freeze)
        if self.use_pretrained:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(pretrained_name_or_path)
            encoder_dim = int(getattr(self.encoder.config, "hidden_size", feature_dim))
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=400, stride=320, padding=40),
                nn.GELU(),
                nn.Conv1d(128, feature_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            encoder_dim = feature_dim
        self.adapter = nn.Linear(encoder_dim, output_dim)
        if self.freeze:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained:
            context = torch.no_grad() if self.freeze else torch.enable_grad()
            with context:
                outputs = self.encoder(wav_16k, output_hidden_states=True)
                hidden = torch.stack(outputs.hidden_states, dim=0).mean(dim=0)
        else:
            context = torch.no_grad() if self.freeze else torch.enable_grad()
            with context:
                hidden = self.encoder(wav_16k.unsqueeze(1)).transpose(1, 2)
        return self.adapter(hidden)


class HybridSemanticLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 512,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: int = 4096,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.sos_token_id = self.vocab_size
        self.pad_token_id = self.vocab_size + 1
        self.embedding = nn.Embedding(self.vocab_size + 2, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, self.vocab_size)
        self.label_smoothing = float(label_smoothing)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def _run(self, inputs_embeds: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = inputs_embeds.size(1)
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(f"LM sequence length {seq_len} exceeds max_position_embeddings")
        positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        hidden = inputs_embeds + self.position_embedding(positions)
        key_padding_mask = None if input_mask is None else ~input_mask.to(device=hidden.device, dtype=torch.bool)
        hidden = self.layers(
            hidden,
            mask=self._causal_mask(seq_len, hidden.device),
            src_key_padding_mask=key_padding_mask,
        )
        return self.norm(hidden)

    def forward(
        self,
        prefix: torch.Tensor,
        target_tokens: torch.LongTensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_tokens = target_tokens.long()
        if target_mask is None:
            target_mask = target_tokens != self.pad_token_id
        else:
            target_mask = target_mask.to(device=target_tokens.device, dtype=torch.bool)
        target_tokens = target_tokens.masked_fill(~target_mask, self.pad_token_id)
        sos = torch.full(
            (target_tokens.size(0), 1),
            self.sos_token_id,
            dtype=torch.long,
            device=target_tokens.device,
        )
        shifted_input = torch.cat([sos, target_tokens[:, :-1]], dim=1)
        token_input = torch.cat([sos, target_tokens], dim=1)
        token_input_mask = torch.cat([torch.ones_like(sos, dtype=torch.bool), target_mask], dim=1)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
        input_mask = torch.cat([prefix_mask, token_input_mask], dim=1)
        inputs = torch.cat([prefix, self.embedding(token_input)], dim=1)
        hidden = self._run(inputs, input_mask=input_mask)
        prefix_len = prefix.size(1)
        shifted_hidden = hidden[:, prefix_len : prefix_len + shifted_input.size(1), :]
        token_hidden = hidden[:, prefix_len + 1 : prefix_len + 1 + target_tokens.size(1), :]
        logits = self.output_head(shifted_hidden)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            label_smoothing=self.label_smoothing,
            ignore_index=self.pad_token_id,
            reduction="none",
        ).reshape_as(target_tokens)
        valid_count = target_mask.float().sum().clamp_min(1.0)
        nll = (token_losses * target_mask.float()).sum() / valid_count
        accuracy = (
            ((logits.argmax(dim=-1) == target_tokens) & target_mask).float().sum()
            / valid_count
        )
        return {
            "loss": nll,
            "accuracy": accuracy,
            "logits": logits,
            "targets": target_tokens,
            "hidden_states": token_hidden,
            "hidden_mask": target_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        generated = torch.empty(prefix.size(0), 0, dtype=torch.long, device=prefix.device)
        token_hidden = []
        current = torch.full((prefix.size(0), 1), self.sos_token_id, dtype=torch.long, device=prefix.device)
        for _ in range(int(max_tokens)):
            inputs = torch.cat([prefix, self.embedding(torch.cat([current, generated], dim=1))], dim=1)
            hidden = self._run(inputs)[:, -1:, :]
            logits = self.output_head(hidden).squeeze(1)
            if do_sample:
                probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            token_inputs = torch.cat([current, generated], dim=1)
            token_hidden.append(self._run(torch.cat([prefix, self.embedding(token_inputs)], dim=1))[:, -1:, :])
        return {
            "tokens": generated,
            "hidden_states": torch.cat(token_hidden, dim=1) if token_hidden else prefix[:, :0, :],
            "hidden_mask": torch.ones(generated.shape, dtype=torch.bool, device=generated.device),
        }
