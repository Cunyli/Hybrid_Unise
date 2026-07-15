from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HybridOutput:
    final_wav: torch.Tensor
    disc_wav: Optional[torch.Tensor]
    gen_wav: Optional[torch.Tensor]
    gen_wav_16k: Optional[torch.Tensor]
    final_spec: torch.Tensor
    disc_spec: Optional[torch.Tensor]
    gen_spec: Optional[torch.Tensor]
    gen_spec_16k: Optional[torch.Tensor]
    fusion_mask: Optional[torch.Tensor]
    token_logits: Optional[torch.Tensor]
    token_targets: Optional[torch.Tensor]
    token_nll: Optional[torch.Tensor]
    lm_hidden_states: Optional[torch.Tensor]
    lm_hidden_mask: Optional[torch.Tensor]
    aligned_lm_hidden_states: Optional[torch.Tensor]
    aligned_lm_hidden_mask: Optional[torch.Tensor]
    length: Optional[torch.Tensor]
    token_objective_nll: Optional[torch.Tensor] = None
    token_weighted_nll: Optional[torch.Tensor] = None
    token_initial_nll: Optional[torch.Tensor] = None
    token_repeat_nll: Optional[torch.Tensor] = None
    token_transition_nll: Optional[torch.Tensor] = None
    token_accuracy: Optional[torch.Tensor] = None
    token_initial_accuracy: Optional[torch.Tensor] = None
    token_repeat_accuracy: Optional[torch.Tensor] = None
    token_transition_accuracy: Optional[torch.Tensor] = None
    token_transition_fraction: Optional[torch.Tensor] = None
    token_transition_predecessor_margin_loss: Optional[torch.Tensor] = None
    token_transition_predecessor_rate: Optional[torch.Tensor] = None
    token_prefix_only_nll: Optional[torch.Tensor] = None
    token_prefix_only_weighted_nll: Optional[torch.Tensor] = None
    token_prefix_only_transition_nll: Optional[torch.Tensor] = None
    token_prefix_only_transition_predecessor_margin_loss: Optional[torch.Tensor] = None
    token_prefix_only_transition_predecessor_rate: Optional[torch.Tensor] = None
