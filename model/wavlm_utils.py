import torch
from torch import nn


def validate_waveform_lengths(
    wav: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.LongTensor:
    values = lengths.to(device=wav.device, dtype=torch.long).flatten()
    if values.numel() != wav.size(0):
        raise ValueError(
            f"Expected {wav.size(0)} waveform lengths, got {values.numel()}"
        )
    if (values <= 0).any():
        raise ValueError("Waveform lengths must be positive")
    if (values > wav.size(-1)).any():
        raise ValueError(
            f"Waveform lengths must not exceed padded length {wav.size(-1)}"
        )
    return values


def waveform_attention_mask(
    wav: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.BoolTensor:
    values = validate_waveform_lengths(wav, lengths)
    steps = torch.arange(wav.size(-1), device=wav.device).unsqueeze(0)
    return steps < values.unsqueeze(1)


def wavlm_feature_lengths(
    encoder: nn.Module,
    lengths: torch.Tensor,
) -> torch.LongTensor:
    output_length_fn = getattr(encoder, "_get_feat_extract_output_lengths", None)
    if not callable(output_length_fn):
        raise TypeError(
            "WavLM encoder must expose _get_feat_extract_output_lengths"
        )
    feature_lengths = torch.as_tensor(
        output_length_fn(lengths, add_adapter=False),
        device=lengths.device,
        dtype=torch.long,
    ).flatten()
    if feature_lengths.shape != lengths.shape:
        raise ValueError(
            "WavLM feature-length helper returned an unexpected batch shape"
        )
    if (feature_lengths < 1).any():
        shortest = int(lengths[feature_lengths.argmin()].item())
        raise ValueError(
            "Waveform is too short for one WavLM feature frame: "
            f"length={shortest} samples"
        )
    return feature_lengths


def conv_stack_feature_lengths(
    encoder: nn.Module,
    lengths: torch.Tensor,
) -> torch.LongTensor:
    output_lengths = lengths.to(dtype=torch.long)
    conv_count = 0
    for layer in encoder.modules():
        if not isinstance(layer, nn.Conv1d):
            continue
        conv_count += 1
        kernel_size = int(layer.kernel_size[0])
        stride = int(layer.stride[0])
        padding = int(layer.padding[0])
        dilation = int(layer.dilation[0])
        output_lengths = torch.div(
            output_lengths
            + 2 * padding
            - dilation * (kernel_size - 1)
            - 1,
            stride,
            rounding_mode="floor",
        ) + 1
    if conv_count == 0:
        raise TypeError("Conditioner encoder contains no Conv1d feature extractor")
    if (output_lengths < 1).any():
        raise ValueError("Waveform is too short for one conditioner feature frame")
    return output_lengths


def feature_mask_from_lengths(
    feature_count: int,
    feature_lengths: torch.Tensor,
) -> torch.BoolTensor:
    if (feature_lengths > feature_count).any():
        raise ValueError(
            "Computed feature length exceeds the encoder output frame count"
        )
    steps = torch.arange(feature_count, device=feature_lengths.device).unsqueeze(0)
    return steps < feature_lengths.unsqueeze(1)
