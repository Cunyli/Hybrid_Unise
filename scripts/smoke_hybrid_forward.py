import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from model import Model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny Hybrid-UniSE forward/loss smoke check")
    parser.add_argument("--config", default="conf/hybrid_unise_smoke.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=3200)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    if config.get("model_type") != "hybrid_unise":
        raise ValueError("smoke_hybrid_forward.py requires model_type: hybrid_unise")

    device = torch.device(args.device)
    model = Model(config).to(device)
    model.eval()

    degraded = torch.randn(1, args.samples, device=device)
    clean = torch.randn(1, args.samples, device=device)
    sample_rate = torch.tensor([16000], device=device)

    with torch.inference_mode():
        output = model(degraded, sample_rate, clean_wav=clean)
        losses = model._losses(output, clean, sample_rate)
        stage_loss = model._weighted_stage_loss(losses)

    if output.final_wav.shape != degraded.shape:
        raise RuntimeError(f"final_wav shape mismatch: {output.final_wav.shape} vs {degraded.shape}")
    if output.fusion_mask is None:
        raise RuntimeError("fusion_mask is missing")
    if not torch.isfinite(output.final_wav).all():
        raise RuntimeError("final_wav contains NaN/Inf")
    if output.token_targets is None or output.lm_hidden_states is None:
        raise RuntimeError("LM token targets or hidden states are missing")

    print("final_wav", tuple(output.final_wav.shape))
    print("disc_wav", tuple(output.disc_wav.shape))
    print("gen_wav", tuple(output.gen_wav.shape))
    print("fusion_mask", tuple(output.fusion_mask.shape), float(output.fusion_mask.min()), float(output.fusion_mask.max()))
    print("token_targets", tuple(output.token_targets.shape))
    print("lm_hidden_states", tuple(output.lm_hidden_states.shape))
    print("losses", {key: float(value.detach().cpu()) for key, value in losses.items()})
    print("stage_loss", float(stage_loss.detach().cpu()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
