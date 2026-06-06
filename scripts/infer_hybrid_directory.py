import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
import yaml

from model import Model
from model.hybrid_model import (
    load_hybrid_checkpoint,
    validate_hybrid_architecture_metadata,
    validate_hybrid_checkpoint_metadata,
)


def iter_audio_files(input_root: Path):
    suffixes = {".wav", ".flac"}
    return sorted(path for path in input_root.rglob("*") if path.suffix.lower() in suffixes)


def find_reference_path(reference_root: Path, rel_path: Path) -> Path | None:
    exact_path = reference_root / rel_path
    if exact_path.is_file():
        return exact_path
    for suffix in (".wav", ".flac"):
        candidate = reference_root / rel_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def load_checkpoint(model, checkpoint_path: str | None, device: torch.device) -> None:
    if not checkpoint_path:
        return
    checkpoint = load_hybrid_checkpoint(checkpoint_path, map_location=device)
    validate_hybrid_checkpoint_metadata(checkpoint, model.stage)
    validate_hybrid_architecture_metadata(checkpoint, model.architecture_config)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)


def write_scp(path: Path, entries: list[tuple[str, Path]]) -> None:
    with path.open("w") as handle:
        for utt_id, wav_path in entries:
            handle.write(f"{utt_id} {wav_path}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid-UniSE directory inference")
    parser.add_argument("--config", default="conf/hybrid_unise_urgent2026.yaml")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--output-root", default="outputs/hybrid_unise")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--stage",
        choices=("disc", "gen", "fusion", "joint"),
        default=None,
        help="Override config stage for loading a stage-specific checkpoint",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-intermediates", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    reference_root = Path(args.reference_root) if args.reference_root else None
    output_root = Path(args.output_root)
    wav_dir = output_root / "wav"
    disc_dir = output_root / "disc"
    gen_dir = output_root / "gen"
    wav_dir.mkdir(parents=True, exist_ok=True)
    if args.save_intermediates:
        disc_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    if config.get("model_type") != "hybrid_unise":
        raise ValueError("scripts/infer_hybrid_directory.py requires model_type: hybrid_unise")
    if args.stage is not None:
        config["stage"] = args.stage
    config["stage_init_checkpoint"] = None
    if args.checkpoint:
        config["ckpt_path"] = args.checkpoint
    if args.save_intermediates:
        config["save_intermediates"] = True

    device = torch.device(args.device)
    model = Model(config).to(device)
    load_checkpoint(model, args.checkpoint or config.get("ckpt_path"), device)
    model.eval()

    inf_entries: list[tuple[str, Path]] = []
    ref_entries: list[tuple[str, Path]] = []

    for src_path in iter_audio_files(input_root):
        rel = src_path.relative_to(input_root)
        utt_id = str(rel.with_suffix("")).replace("/", "_")
        wav, sample_rate = sf.read(src_path, dtype="float32", always_2d=True)
        wav_tensor = torch.from_numpy(wav[:, :1].T).to(device)
        with torch.inference_mode():
            output = model.enhance(
                wav_tensor,
                torch.tensor([sample_rate], device=device),
                return_intermediates=args.save_intermediates,
            )

        out_path = wav_dir / rel.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, output.final_wav[0].detach().cpu().numpy(), sample_rate)
        inf_entries.append((utt_id, out_path))
        if reference_root is not None:
            ref_path = find_reference_path(reference_root, rel)
            if ref_path is not None:
                ref_entries.append((utt_id, ref_path))

        if args.save_intermediates:
            disc_path = disc_dir / rel.with_suffix(".wav")
            disc_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(disc_path, output.disc_wav[0].detach().cpu().numpy(), sample_rate)
            if output.gen_wav is not None:
                gen_path = gen_dir / rel.with_suffix(".wav")
                gen_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(gen_path, output.gen_wav[0].detach().cpu().numpy(), sample_rate)

    write_scp(output_root / "inf.scp", inf_entries)
    write_scp(output_root / "ref.scp", ref_entries)
    print(f"Wrote {len(inf_entries)} enhanced files to {wav_dir}")
    print(f"Wrote {output_root / 'inf.scp'} and {output_root / 'ref.scp'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
