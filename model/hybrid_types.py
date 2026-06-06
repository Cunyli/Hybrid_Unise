from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HybridOutput:
    final_wav: torch.Tensor
    disc_wav: torch.Tensor
    gen_wav: Optional[torch.Tensor]
    gen_wav_16k: Optional[torch.Tensor]
    final_spec: torch.Tensor
    disc_spec: torch.Tensor
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
