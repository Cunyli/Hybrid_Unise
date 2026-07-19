import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel

from .wavlm_utils import (
    conv_stack_feature_lengths,
    feature_mask_from_lengths,
    validate_waveform_lengths,
    waveform_attention_mask,
    wavlm_feature_lengths,
)


def _prefix_mask(
    prefix: torch.Tensor,
    prefix_mask: torch.Tensor | None,
) -> torch.BoolTensor:
    if prefix_mask is None:
        return torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
    mask = prefix_mask.to(device=prefix.device, dtype=torch.bool)
    if mask.shape != prefix.shape[:2]:
        raise ValueError(
            f"Prefix mask shape {tuple(mask.shape)} does not match "
            f"prefix shape {tuple(prefix.shape[:2])}"
        )
    if not mask.any(dim=1).all():
        raise ValueError("Prefix mask must keep at least one frame per sample")
    return mask


def _compact_position_ids(input_mask: torch.Tensor) -> torch.LongTensor:
    mask = input_mask.to(dtype=torch.bool)
    positions = mask.long().cumsum(dim=-1) - 1
    return torch.where(mask, positions, torch.zeros_like(positions))


def _semantic_token_embeddings(
    embedding: nn.Embedding,
    token_input: torch.LongTensor,
    *,
    token_input_mask: torch.BoolTensor,
    zero_history: bool,
    history_embedding_dropout_prob: float,
    training: bool,
    history_corruption_replacement_fraction: float = 0.0,
    transition_stall_history_corruption: bool = False,
    semantic_vocab_size: int | None = None,
) -> torch.Tensor:
    history_embedding_dropout_prob = float(history_embedding_dropout_prob)
    history_corruption_replacement_fraction = float(
        history_corruption_replacement_fraction
    )
    if not math.isfinite(history_embedding_dropout_prob):
        raise ValueError("history_embedding_dropout_prob must be finite")
    if not 0.0 <= history_embedding_dropout_prob < 1.0:
        raise ValueError(
            "history_embedding_dropout_prob must be in the interval [0, 1)"
        )
    if not math.isfinite(history_corruption_replacement_fraction):
        raise ValueError("history_corruption_replacement_fraction must be finite")
    if not 0.0 <= history_corruption_replacement_fraction <= 1.0:
        raise ValueError(
            "history_corruption_replacement_fraction must be in the interval [0, 1]"
        )
    if not isinstance(transition_stall_history_corruption, bool):
        raise ValueError("transition_stall_history_corruption must be a bool")
    if (
        history_corruption_replacement_fraction > 0.0
        and history_embedding_dropout_prob == 0.0
    ):
        raise ValueError(
            "history_corruption_replacement_fraction requires a positive "
            "history_embedding_dropout_prob"
        )
    if (
        transition_stall_history_corruption
        and history_corruption_replacement_fraction == 0.0
    ):
        raise ValueError(
            "transition_stall_history_corruption requires a positive "
            "history_corruption_replacement_fraction"
        )
    if token_input_mask.shape != token_input.shape:
        raise ValueError(
            f"Token input mask shape {tuple(token_input_mask.shape)} does not match "
            f"token input shape {tuple(token_input.shape)}"
        )
    if zero_history:
        token_embeddings = embedding(token_input)
        token_embeddings = token_embeddings.clone()
        token_embeddings[:, 1:, :] = 0.0
    elif training and history_embedding_dropout_prob > 0.0:
        valid_history = token_input_mask.to(
            device=token_input.device,
            dtype=torch.bool,
        ).clone()
        valid_history[:, 0] = False
        valid_indices = valid_history.nonzero(as_tuple=True)
        uniform_values = torch.rand(
            valid_indices[0].numel(),
            device=token_input.device,
        )
        corruption_uniform = torch.zeros(
            valid_history.shape,
            dtype=uniform_values.dtype,
            device=token_input.device,
        )
        corruption_uniform[valid_indices] = uniform_values
        corrupt_history = torch.zeros_like(valid_history)
        corrupt_history[valid_indices] = (
            corruption_uniform[valid_indices] < history_embedding_dropout_prob
        )
        if history_corruption_replacement_fraction > 0.0:
            if semantic_vocab_size is None:
                raise ValueError(
                    "semantic_vocab_size is required for semantic history replacement"
                )
            semantic_vocab_size = int(semantic_vocab_size)
            if semantic_vocab_size < 2:
                raise ValueError(
                    "Semantic history replacement requires at least two "
                    "non-special vocabulary entries"
                )
            if embedding.num_embeddings < semantic_vocab_size + 2:
                raise ValueError(
                    "Embedding vocabulary must include the semantic vocabulary, "
                    "SOS, and padding entries"
                )
            replacement_eligible = torch.zeros_like(valid_history)
            if token_input.size(1) > 2:
                replacement_eligible[:, 1:-1] = token_input_mask[:, 2:].to(
                    device=token_input.device,
                    dtype=torch.bool,
                )
            replacement_threshold = (
                history_embedding_dropout_prob
                * history_corruption_replacement_fraction
            )
            replace_history = (
                corrupt_history
                & replacement_eligible
                & (corruption_uniform < replacement_threshold)
            )
            replacement_indices = replace_history.nonzero(as_tuple=True)
            replacement_count = replacement_indices[0].numel()
            corrupted_token_input = token_input.clone()
            if replacement_count > 0:
                original_tokens = token_input[replacement_indices]
                if (
                    (original_tokens < 0).any()
                    or (original_tokens >= semantic_vocab_size).any()
                ):
                    raise ValueError(
                        "Valid history tokens must be non-special semantic token IDs"
                    )
                replacement_unit = (
                    corruption_uniform[replacement_indices]
                    / replacement_threshold
                )
                replacement_offsets = (
                    torch.floor(replacement_unit * (semantic_vocab_size - 1))
                    .long()
                    .clamp_(max=semantic_vocab_size - 2)
                    + 1
                )
                replacement_tokens = (
                    original_tokens + replacement_offsets
                ) % semantic_vocab_size
                corrupted_token_input[replacement_indices] = replacement_tokens
                if transition_stall_history_corruption and token_input.size(1) > 3:
                    transition_stall = torch.zeros_like(replace_history)
                    transition_stall[:, 2:-1] = (
                        replace_history[:, 2:-1]
                        & token_input_mask[:, 1:-2].to(
                            device=token_input.device,
                            dtype=torch.bool,
                        )
                        & token_input_mask[:, 2:-1].to(
                            device=token_input.device,
                            dtype=torch.bool,
                        )
                        & token_input_mask[:, 3:].to(
                            device=token_input.device,
                            dtype=torch.bool,
                        )
                        & (token_input[:, 2:-1] != token_input[:, 1:-2])
                    )
                    transition_indices = transition_stall.nonzero(as_tuple=True)
                    if transition_indices[0].numel() > 0:
                        predecessor_tokens = token_input[
                            transition_indices[0],
                            transition_indices[1] - 1,
                        ]
                        if (
                            (predecessor_tokens < 0).any()
                            or (predecessor_tokens >= semantic_vocab_size).any()
                        ):
                            raise ValueError(
                                "Transition predecessor history tokens must be "
                                "non-special semantic token IDs"
                            )
                        corrupted_token_input[transition_indices] = predecessor_tokens
            token_embeddings = embedding(corrupted_token_input)
            drop_history = corrupt_history & ~replace_history
        else:
            token_embeddings = embedding(token_input)
            drop_history = corrupt_history
        token_embeddings = token_embeddings.masked_fill(
            drop_history.unsqueeze(-1),
            0.0,
        )
    else:
        token_embeddings = embedding(token_input)
    return token_embeddings


def _semantic_loss_metrics(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    target_mask: torch.BoolTensor,
    *,
    label_smoothing: float,
    ignore_index: int,
    transition_loss_weight: float,
    normalize_transition_weights_per_sample: bool = False,
    transition_predecessor_margin: float = 1.0,
    transition_predecessor_margin_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    if isinstance(label_smoothing, bool):
        raise ValueError("label_smoothing must be finite and in [0, 1]")
    label_smoothing = float(label_smoothing)
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be finite and in [0, 1]")
    if isinstance(transition_predecessor_margin, bool):
        raise ValueError(
            "transition_predecessor_margin must be a finite non-negative number"
        )
    if isinstance(transition_predecessor_margin_weight, bool):
        raise ValueError(
            "transition_predecessor_margin_weight must be a finite non-negative number"
        )
    transition_loss_weight = float(transition_loss_weight)
    transition_predecessor_margin = float(transition_predecessor_margin)
    transition_predecessor_margin_weight = float(
        transition_predecessor_margin_weight
    )
    if not math.isfinite(transition_loss_weight):
        raise ValueError("transition_loss_weight must be finite")
    if transition_loss_weight < 1.0:
        raise ValueError("transition_loss_weight must be at least 1.0")
    if not isinstance(normalize_transition_weights_per_sample, bool):
        raise ValueError("normalize_transition_weights_per_sample must be a bool")
    if (
        not math.isfinite(transition_predecessor_margin)
        or transition_predecessor_margin < 0.0
    ):
        raise ValueError(
            "transition_predecessor_margin must be a finite non-negative number"
        )
    if (
        not math.isfinite(transition_predecessor_margin_weight)
        or transition_predecessor_margin_weight < 0.0
    ):
        raise ValueError(
            "transition_predecessor_margin_weight must be a finite non-negative number"
        )
    if logits.ndim != 3:
        raise ValueError(f"Logits must be rank 3, got shape {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"Targets must be rank 2, got shape {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Logit sequence shape {tuple(logits.shape[:2])} does not match "
            f"target shape {tuple(targets.shape)}"
        )
    if target_mask.shape != targets.shape:
        raise ValueError(
            f"Target mask shape {tuple(target_mask.shape)} does not match "
            f"target shape {tuple(targets.shape)}"
        )
    if not target_mask.any(dim=1).all():
        raise ValueError("Target mask must keep at least one token per sample")
    if targets.size(1) > 1 and ((~target_mask[:, :-1]) & target_mask[:, 1:]).any():
        raise ValueError("Target mask must use contiguous right padding")
    valid_targets = targets[target_mask]
    if (valid_targets < 0).any() or (valid_targets >= logits.size(-1)).any():
        raise ValueError("Valid semantic targets must index the logit vocabulary")
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction="none",
    ).reshape_as(targets)
    transition_mask = torch.zeros_like(target_mask)
    repeat_mask = torch.zeros_like(target_mask)
    if targets.size(1) > 1:
        valid_pairs = target_mask[:, 1:] & target_mask[:, :-1]
        changed = targets[:, 1:] != targets[:, :-1]
        transition_mask[:, 1:] = valid_pairs & changed
        repeat_mask[:, 1:] = valid_pairs & ~changed
    initial_mask = torch.zeros_like(target_mask)
    if targets.size(1) > 0:
        initial_mask[:, 0] = target_mask[:, 0]

    valid_weights = target_mask.float()
    objective_weights = valid_weights + (
        transition_loss_weight - 1.0
    ) * transition_mask.to(dtype=valid_weights.dtype)
    if normalize_transition_weights_per_sample and transition_loss_weight != 1.0:
        valid_counts_per_sample = valid_weights.sum(dim=1)
        objective_weight_sums_per_sample = objective_weights.sum(dim=1)
        per_sample_scale = (
            valid_counts_per_sample
            / objective_weight_sums_per_sample.clamp_min(1.0)
        ).unsqueeze(1)
        normalized_objective_weights = objective_weights * per_sample_scale
        ce_objective_loss = (
            (token_losses * normalized_objective_weights).sum()
            / valid_weights.sum().clamp_min(1.0)
        )
    else:
        ce_objective_loss = (
            (token_losses * objective_weights).sum()
            / objective_weights.sum().clamp_min(1.0)
        )

    safe_targets = targets.masked_fill(~target_mask, 0)
    predecessor_targets = torch.zeros_like(safe_targets)
    if targets.size(1) > 1:
        predecessor_targets[:, 1:] = safe_targets[:, :-1]
    target_logits = (
        logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1).float()
    )
    predecessor_logits = logits.gather(
        -1,
        predecessor_targets.unsqueeze(-1),
    ).squeeze(-1).float()
    predecessor_margin_terms = F.relu(
        transition_predecessor_margin + predecessor_logits - target_logits
    )
    transition_count = transition_mask.float().sum()
    transition_predecessor_margin_loss = (
        predecessor_margin_terms * transition_mask.float()
    ).sum() / transition_count.clamp_min(1.0)
    objective_loss = ce_objective_loss
    if transition_predecessor_margin_weight > 0.0:
        objective_loss = (
            ce_objective_loss
            + transition_predecessor_margin_weight
            * transition_predecessor_margin_loss
        )

    def masked_mean(mask: torch.BoolTensor) -> torch.Tensor:
        weights = mask.to(dtype=valid_weights.dtype)
        return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)

    predictions = logits.argmax(dim=-1)

    def masked_accuracy(mask: torch.BoolTensor) -> torch.Tensor:
        correct = (predictions == targets) & mask
        return correct.float().sum() / mask.float().sum().clamp_min(1.0)

    transition_predecessor_rate = (
        ((predictions == predecessor_targets) & transition_mask).float().sum()
        / transition_count.clamp_min(1.0)
    )

    return {
        "loss": (token_losses * valid_weights).sum() / valid_weights.sum().clamp_min(1.0),
        "ce_objective_loss": ce_objective_loss,
        "objective_loss": objective_loss,
        "initial_nll": masked_mean(initial_mask),
        "repeat_nll": masked_mean(repeat_mask),
        "transition_nll": masked_mean(transition_mask),
        "initial_accuracy": masked_accuracy(initial_mask),
        "repeat_accuracy": masked_accuracy(repeat_mask),
        "transition_accuracy": masked_accuracy(transition_mask),
        "transition_predecessor_margin_loss": transition_predecessor_margin_loss,
        "transition_predecessor_rate": transition_predecessor_rate,
        "transition_fraction": (
            transition_mask.float().sum() / target_mask.float().sum().clamp_min(1.0)
        ),
    }


class WavLMConditioner(nn.Module):
    def __init__(
        self,
        output_dim: int = 512,
        freeze: bool = True,
        pretrained_name_or_path: str = "microsoft/wavlm-base-plus",
        use_pretrained: bool = False,
        feature_dim: int = 768,
        layer_mode: str = "mean",
        layer_index: int = -1,
    ):
        super().__init__()
        self.use_pretrained = bool(use_pretrained)
        self.freeze = bool(freeze)
        self.layer_mode = str(layer_mode)
        self.layer_index = int(layer_index)
        if self.layer_mode not in {"mean", "last", "layer"}:
            raise ValueError("wavlm.layer_mode must be 'mean', 'last', or 'layer'")
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

    def _select_pretrained_hidden(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not hidden_states:
            raise ValueError("Pretrained WavLM did not return hidden states")
        if self.layer_mode == "mean":
            return torch.stack(hidden_states, dim=0).mean(dim=0)
        if self.layer_mode == "last":
            return hidden_states[-1]
        layer_index = self.layer_index
        if layer_index < 0:
            layer_index += len(hidden_states)
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise ValueError(
                f"wavlm.layer_index={self.layer_index} is out of range for "
                f"{len(hidden_states)} hidden-state tensors"
            )
        return hidden_states[layer_index]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def encode_with_mask(
        self,
        wav_16k: torch.Tensor,
        lengths_16k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        lengths = None
        encoder_kwargs = {"output_hidden_states": True}
        if lengths_16k is not None:
            lengths = validate_waveform_lengths(wav_16k, lengths_16k)
            attention_mask = waveform_attention_mask(wav_16k, lengths)
            if not attention_mask.all():
                encoder_kwargs["attention_mask"] = attention_mask.long()
        if self.use_pretrained:
            context = torch.no_grad() if self.freeze else torch.enable_grad()
            with context:
                outputs = self.encoder(wav_16k, **encoder_kwargs)
                hidden = self._select_pretrained_hidden(tuple(outputs.hidden_states))
            feature_lengths = (
                None
                if lengths is None
                else wavlm_feature_lengths(self.encoder, lengths)
            )
        else:
            if self.layer_mode != "mean":
                raise ValueError("wavlm.layer_mode other than 'mean' requires use_pretrained=true")
            context = torch.no_grad() if self.freeze else torch.enable_grad()
            with context:
                hidden = self.encoder(wav_16k.unsqueeze(1)).transpose(1, 2)
            feature_lengths = (
                None
                if lengths is None
                else conv_stack_feature_lengths(self.encoder, lengths)
            )
        prefix = self.adapter(hidden)
        prefix_mask = (
            torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
            if feature_lengths is None
            else feature_mask_from_lengths(prefix.size(1), feature_lengths)
        )
        return prefix, prefix_mask

    def forward(
        self,
        wav_16k: torch.Tensor,
        lengths_16k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_with_mask(wav_16k, lengths_16k)[0]


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
        if isinstance(label_smoothing, bool):
            raise ValueError("label_smoothing must be finite and in [0, 1]")
        self.label_smoothing = float(label_smoothing)
        if (
            not math.isfinite(self.label_smoothing)
            or not 0.0 <= self.label_smoothing <= 1.0
        ):
            raise ValueError("label_smoothing must be finite and in [0, 1]")

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def _run(self, inputs_embeds: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = inputs_embeds.size(1)
        if input_mask is None:
            positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        else:
            positions = _compact_position_ids(
                input_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            )
        if int(positions.max().item()) >= self.position_embedding.num_embeddings:
            raise ValueError(
                f"LM position {int(positions.max().item())} exceeds max_position_embeddings"
            )
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
        prefix_mask: torch.Tensor | None = None,
        *,
        zero_history: bool = False,
        history_embedding_dropout_prob: float = 0.0,
        history_corruption_replacement_fraction: float = 0.0,
        transition_stall_history_corruption: bool = False,
        transition_loss_weight: float = 1.0,
        normalize_transition_weights_per_sample: bool = False,
        transition_predecessor_margin: float = 1.0,
        transition_predecessor_margin_weight: float = 0.0,
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
        valid_prefix = _prefix_mask(prefix, prefix_mask)
        input_mask = torch.cat([valid_prefix, token_input_mask], dim=1)
        token_embeddings = _semantic_token_embeddings(
            self.embedding,
            token_input,
            token_input_mask=token_input_mask,
            zero_history=zero_history,
            history_embedding_dropout_prob=history_embedding_dropout_prob,
            training=self.training,
            history_corruption_replacement_fraction=(
                history_corruption_replacement_fraction
            ),
            transition_stall_history_corruption=(
                transition_stall_history_corruption
            ),
            semantic_vocab_size=self.vocab_size,
        )
        inputs = torch.cat([prefix, token_embeddings], dim=1)
        hidden = self._run(inputs, input_mask=input_mask)
        prefix_len = prefix.size(1)
        shifted_hidden = hidden[:, prefix_len : prefix_len + shifted_input.size(1), :]
        token_hidden = hidden[:, prefix_len + 1 : prefix_len + 1 + target_tokens.size(1), :]
        logits = self.output_head(shifted_hidden)
        loss_metrics = _semantic_loss_metrics(
            logits,
            target_tokens,
            target_mask,
            label_smoothing=self.label_smoothing,
            ignore_index=self.pad_token_id,
            transition_loss_weight=transition_loss_weight,
            normalize_transition_weights_per_sample=(
                normalize_transition_weights_per_sample
            ),
            transition_predecessor_margin=transition_predecessor_margin,
            transition_predecessor_margin_weight=(
                transition_predecessor_margin_weight
            ),
        )
        valid_count = target_mask.float().sum().clamp_min(1.0)
        accuracy = (
            ((logits.argmax(dim=-1) == target_tokens) & target_mask).float().sum()
            / valid_count
        )
        return {
            **loss_metrics,
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
        prefix_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid_prefix = _prefix_mask(prefix, prefix_mask)
        generated = torch.empty(prefix.size(0), 0, dtype=torch.long, device=prefix.device)
        token_hidden = []
        current = torch.full((prefix.size(0), 1), self.sos_token_id, dtype=torch.long, device=prefix.device)
        for _ in range(int(max_tokens)):
            token_inputs = torch.cat([current, generated], dim=1)
            inputs = torch.cat([prefix, self.embedding(token_inputs)], dim=1)
            token_mask = torch.ones(token_inputs.shape, dtype=torch.bool, device=prefix.device)
            input_mask = torch.cat([valid_prefix, token_mask], dim=1)
            hidden = self._run(inputs, input_mask=input_mask)[:, -1:, :]
            logits = self.output_head(hidden).squeeze(1)
            if do_sample:
                probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            token_inputs = torch.cat([current, generated], dim=1)
            token_mask = torch.ones(token_inputs.shape, dtype=torch.bool, device=prefix.device)
            input_mask = torch.cat([valid_prefix, token_mask], dim=1)
            token_hidden.append(
                self._run(
                    torch.cat([prefix, self.embedding(token_inputs)], dim=1),
                    input_mask=input_mask,
                )[:, -1:, :]
            )
        return {
            "tokens": generated,
            "hidden_states": torch.cat(token_hidden, dim=1) if token_hidden else prefix[:, :0, :],
            "hidden_mask": torch.ones(generated.shape, dtype=torch.bool, device=generated.device),
        }


class HybridLlamaSemanticLM(nn.Module):
    """LLaMA-style decoder-only LM for Hybrid-UniSE clean codec tokens.

    The public Hybrid-UniSE paper describes a decoder-only LM with 12 LLaMA
    layers. This class keeps the repository's existing prefix + first-RVQ token
    contract while swapping the TransformerEncoder stack for LLaMA decoder
    blocks and RoPE.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 512,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: int = 4096,
        label_smoothing: float = 0.0,
        intermediate_size: int | None = None,
        rms_norm_eps: float = 1.0e-6,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.sos_token_id = self.vocab_size
        self.pad_token_id = self.vocab_size + 1
        self.embedding = nn.Embedding(self.vocab_size + 2, hidden_size)
        self.config = LlamaConfig(
            vocab_size=self.vocab_size + 2,
            hidden_size=hidden_size,
            intermediate_size=int(intermediate_size or hidden_size * 4),
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.sos_token_id,
            eos_token_id=self.sos_token_id,
            rms_norm_eps=rms_norm_eps,
        )
        self.decoder = LlamaModel(self.config)
        # We always provide inputs_embeds, so the internal token table would be
        # unused trainable state. Freeze it instead of relying on it silently.
        for parameter in self.decoder.embed_tokens.parameters():
            parameter.requires_grad = False
        self.output_head = nn.Linear(hidden_size, self.vocab_size, bias=False)
        if isinstance(label_smoothing, bool):
            raise ValueError("label_smoothing must be finite and in [0, 1]")
        self.label_smoothing = float(label_smoothing)
        if (
            not math.isfinite(self.label_smoothing)
            or not 0.0 <= self.label_smoothing <= 1.0
        ):
            raise ValueError("label_smoothing must be finite and in [0, 1]")

    def _run(self, inputs_embeds: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = inputs_embeds.size(1)
        attention_mask = None
        if input_mask is None:
            position_ids = torch.arange(
                seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)
        else:
            attention_mask = input_mask.to(device=inputs_embeds.device, dtype=torch.long)
            position_ids = _compact_position_ids(attention_mask)
        if int(position_ids.max().item()) >= self.config.max_position_embeddings:
            raise ValueError(
                f"LM position {int(position_ids.max().item())} exceeds max_position_embeddings"
            )
        output = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return output.last_hidden_state

    def forward(
        self,
        prefix: torch.Tensor,
        target_tokens: torch.LongTensor,
        target_mask: torch.Tensor | None = None,
        prefix_mask: torch.Tensor | None = None,
        *,
        zero_history: bool = False,
        history_embedding_dropout_prob: float = 0.0,
        history_corruption_replacement_fraction: float = 0.0,
        transition_stall_history_corruption: bool = False,
        transition_loss_weight: float = 1.0,
        normalize_transition_weights_per_sample: bool = False,
        transition_predecessor_margin: float = 1.0,
        transition_predecessor_margin_weight: float = 0.0,
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
        valid_prefix = _prefix_mask(prefix, prefix_mask)
        input_mask = torch.cat([valid_prefix, token_input_mask], dim=1)
        token_embeddings = _semantic_token_embeddings(
            self.embedding,
            token_input,
            token_input_mask=token_input_mask,
            zero_history=zero_history,
            history_embedding_dropout_prob=history_embedding_dropout_prob,
            training=self.training,
            history_corruption_replacement_fraction=(
                history_corruption_replacement_fraction
            ),
            transition_stall_history_corruption=(
                transition_stall_history_corruption
            ),
            semantic_vocab_size=self.vocab_size,
        )
        inputs = torch.cat([prefix, token_embeddings], dim=1)
        hidden = self._run(inputs, input_mask=input_mask)
        prefix_len = prefix.size(1)
        shifted_hidden = hidden[:, prefix_len : prefix_len + shifted_input.size(1), :]
        token_hidden = hidden[:, prefix_len + 1 : prefix_len + 1 + target_tokens.size(1), :]
        logits = self.output_head(shifted_hidden)
        loss_metrics = _semantic_loss_metrics(
            logits,
            target_tokens,
            target_mask,
            label_smoothing=self.label_smoothing,
            ignore_index=self.pad_token_id,
            transition_loss_weight=transition_loss_weight,
            normalize_transition_weights_per_sample=(
                normalize_transition_weights_per_sample
            ),
            transition_predecessor_margin=transition_predecessor_margin,
            transition_predecessor_margin_weight=(
                transition_predecessor_margin_weight
            ),
        )
        valid_count = target_mask.float().sum().clamp_min(1.0)
        accuracy = (
            ((logits.argmax(dim=-1) == target_tokens) & target_mask).float().sum()
            / valid_count
        )
        return {
            **loss_metrics,
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
        prefix_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid_prefix = _prefix_mask(prefix, prefix_mask)
        generated = torch.empty(prefix.size(0), 0, dtype=torch.long, device=prefix.device)
        token_hidden = []
        current = torch.full((prefix.size(0), 1), self.sos_token_id, dtype=torch.long, device=prefix.device)
        for _ in range(int(max_tokens)):
            token_inputs = torch.cat([current, generated], dim=1)
            inputs = torch.cat([prefix, self.embedding(token_inputs)], dim=1)
            token_mask = torch.ones(token_inputs.shape, dtype=torch.bool, device=prefix.device)
            input_mask = torch.cat([valid_prefix, token_mask], dim=1)
            hidden = self._run(inputs, input_mask=input_mask)[:, -1:, :]
            logits = self.output_head(hidden).squeeze(1)
            if do_sample:
                probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            token_inputs = torch.cat([current, generated], dim=1)
            token_mask = torch.ones(token_inputs.shape, dtype=torch.bool, device=prefix.device)
            input_mask = torch.cat([valid_prefix, token_mask], dim=1)
            token_hidden.append(
                self._run(
                    torch.cat([prefix, self.embedding(token_inputs)], dim=1),
                    input_mask=input_mask,
                )[:, -1:, :]
            )
        return {
            "tokens": generated,
            "hidden_states": torch.cat(token_hidden, dim=1) if token_hidden else prefix[:, :0, :],
            "hidden_mask": torch.ones(generated.shape, dtype=torch.bool, device=generated.device),
        }
