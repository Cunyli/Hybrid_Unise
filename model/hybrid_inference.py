from pathlib import Path
from typing import Union

import soundfile as sf
import torch
import yaml

from .hybrid_model import (
    HybridUniSELightning,
    load_hybrid_checkpoint,
    validate_hybrid_architecture_metadata,
    validate_hybrid_checkpoint_metadata,
)
from .hybrid_types import HybridOutput


def load_hybrid_model(
    config_path: Union[str, Path],
    checkpoint: Union[str, Path, None] = None,
    stage: str | None = None,
    device: Union[str, torch.device, None] = None,
):
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    if config.get("model_type") != "hybrid_unise":
        raise ValueError("Hybrid inference requires a config with model_type: hybrid_unise")
    if stage is not None:
        config["stage"] = stage
    if checkpoint is not None:
        config["ckpt_path"] = str(checkpoint)
    config["stage_init_checkpoint"] = None
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HybridUniSELightning(config).to(device)
    ckpt_path = checkpoint or config.get("ckpt_path")
    if ckpt_path:
        checkpoint_data = load_hybrid_checkpoint(ckpt_path, map_location=device)
        validate_hybrid_checkpoint_metadata(checkpoint_data, model.stage)
        validate_hybrid_architecture_metadata(checkpoint_data, model.architecture_config)
        model.load_state_dict(checkpoint_data.get("state_dict", checkpoint_data), strict=False)
    model.eval()
    return model


@torch.inference_mode()
def enhance(
    wav: torch.Tensor,
    sample_rate: int,
    checkpoint: Union[str, Path, None] = None,
    config_path: Union[str, Path] = "conf/hybrid_unise_urgent2026.yaml",
    stage: str | None = None,
    return_intermediates: bool = False,
    device: Union[str, torch.device, None] = None,
) -> HybridOutput:
    model = load_hybrid_model(config_path=config_path, checkpoint=checkpoint, stage=stage, device=device)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return model.enhance(
        wav.to(model.device),
        torch.tensor([int(sample_rate)], device=model.device),
        return_intermediates=return_intermediates,
    )


@torch.inference_mode()
def enhance_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    checkpoint: Union[str, Path, None] = None,
    config_path: Union[str, Path] = "conf/hybrid_unise_urgent2026.yaml",
    stage: str | None = None,
    return_intermediates: bool = False,
    device: Union[str, torch.device, None] = None,
) -> HybridOutput:
    wav, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    wav_tensor = torch.from_numpy(wav[:, :1].T)
    output = enhance(
        wav_tensor,
        sample_rate,
        checkpoint=checkpoint,
        config_path=config_path,
        stage=stage,
        return_intermediates=return_intermediates,
        device=device,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output.final_wav[0].detach().cpu().numpy(), sample_rate)
    return output
