import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml


DEFAULT_OUTPUT_ROOT = Path("/scratch/work/lil14/data/TAU/simulated/phone_room_enhanced")
DEFAULT_USE_SIMULATION_ROOT = Path("/scratch/work/lil14/USE_simulation")
DEFAULT_TARGET_SAMPLE_RATE = 16000


def load_json_list(path):
    with Path(path).expanduser().open() as f:
        values = json.load(f)
    if not isinstance(values, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [str(Path(value).expanduser()) for value in values]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def stable_seed(base_seed, split, repeat_idx, clean_path):
    text = f"{base_seed}|{split}|{repeat_idx}|{clean_path}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def read_mono(path, target_sr=None):
    audio, sr = sf.read(path, always_2d=True)
    audio = audio[:, :1].T.astype(np.float32, copy=False)
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        sr = target_sr
    return audio, sr


def read_noise_like(path, target_sr, target_len, rng):
    info = sf.info(path)
    duration = target_len / target_sr
    source_len = int(np.ceil(duration * info.samplerate)) + 1
    if info.frames > source_len:
        start = rng.randrange(0, info.frames - source_len + 1)
        audio, sr = sf.read(path, start=start, stop=start + source_len, always_2d=True)
        audio = audio[:, :1].T.astype(np.float32, copy=False)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        return audio
    audio, _ = read_mono(path, target_sr=target_sr)
    return audio


def read_with_retries(paths, rng, label, read_fn, max_attempts):
    attempts = []
    for attempt_idx in range(int(max_attempts)):
        path = paths[rng.randrange(len(paths))]
        try:
            audio = read_fn(path)
            return path, audio, attempts
        except Exception as exc:
            attempts.append(
                {
                    "attempt": int(attempt_idx),
                    "path": str(Path(path).expanduser()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    raise RuntimeError(
        f"Failed to read a usable {label} after {max_attempts} attempts. "
        f"Last errors: {attempts[-3:]}"
    )


def match_length(audio, length):
    if audio.shape[1] > length:
        return audio[:, :length]
    if audio.shape[1] < length:
        return np.pad(audio, ((0, 0), (0, length - audio.shape[1])), constant_values=0)
    return audio


def clean_output_path(clean_dir, split, index, clean_path):
    return clean_dir / f"tau_{split}_{index:05d}_{clean_stem(clean_path)}_clean.wav"


def clean_stem(clean_path):
    return Path(clean_path).stem


def make_uid(split, repeat_idx, index, clean_path):
    return f"tau_{split}_r{repeat_idx:02d}_{index:05d}_{clean_stem(clean_path)}"


def write_pair_csv(path, pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "noisy_filepath", "clean_filepath", "sample_rate"])
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "uid": pair["id"],
                    "noisy_filepath": pair["noisy_path"],
                    "clean_filepath": pair["clean_path"],
                    "sample_rate": pair["sample_rate"],
                }
            )


def import_use_simulation(use_simulation_root):
    root = Path(use_simulation_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"USE_simulation root not found: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from simulate_degradation import apply_degradation_with_wind, random_select_and_order

    return apply_degradation_with_wind, random_select_and_order


def export_augmented_split(args):
    apply_degradation_with_wind, random_select_and_order = import_use_simulation(args.use_simulation_root)
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    clean_paths = load_json_list(args.clean_json)
    noise_paths = load_json_list(args.noise_json)
    rir_paths = load_json_list(args.rir_json)
    wind_noise_paths = load_json_list(args.wind_noise_json) if args.wind_noise_json else []

    if args.limit is not None:
        clean_paths = clean_paths[: args.limit]
    if not clean_paths:
        raise ValueError(f"No clean paths found in {args.clean_json}")
    if not noise_paths:
        raise ValueError(f"No noise paths found in {args.noise_json}")
    if not rir_paths:
        raise ValueError(f"No RIR paths found in {args.rir_json}")

    output_root = Path(args.output_root).expanduser() / args.split
    noisy_dir = output_root / "noisy"
    clean_dir = output_root / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    manifest = []
    total = len(clean_paths) * int(args.num_repeats)
    generated = 0
    for repeat_idx in range(int(args.num_repeats)):
        for index, clean_path in enumerate(clean_paths):
            item_seed = stable_seed(args.seed, args.split, repeat_idx, clean_path)
            py_rng = random.Random(item_seed)

            clean, sample_rate = read_mono(clean_path, target_sr=args.target_sample_rate)
            item_cfg = copy.deepcopy(cfg)
            item_cfg.setdefault("stft_cfg", {})["sampling_rate"] = int(sample_rate)

            noise_path, noise, noise_retries = read_with_retries(
                noise_paths,
                py_rng,
                "noise",
                lambda path: read_noise_like(path, sample_rate, clean.shape[1], py_rng),
                args.max_audio_load_attempts,
            )
            rir_path, rir_result, rir_retries = read_with_retries(
                rir_paths,
                py_rng,
                "RIR",
                lambda path: read_mono(path, target_sr=sample_rate),
                args.max_audio_load_attempts,
            )
            rir, _ = rir_result

            degrad_cfgs, selected_degrads = random_select_and_order(item_cfg, seed=item_seed)
            wind_noise_path = None
            wind_noise = None
            if "wind_noise" in selected_degrads:
                if not wind_noise_paths:
                    raise ValueError("wind_noise was selected, but --wind-noise-json was not provided")
                wind_noise_path, wind_noise, wind_noise_retries = read_with_retries(
                    wind_noise_paths,
                    py_rng,
                    "wind noise",
                    lambda path: read_noise_like(path, sample_rate, clean.shape[1], py_rng),
                    args.max_audio_load_attempts,
                )
            else:
                wind_noise_retries = []

            clean_out, noisy = apply_degradation_with_wind(
                item_cfg,
                clean,
                noise,
                rir,
                wind_noise,
                degrad_cfgs,
                selected_degrads,
                seed=item_seed,
            )
            noisy = match_length(noisy, clean.shape[1])
            clean_out = match_length(clean_out, clean.shape[1])

            uid = make_uid(args.split, repeat_idx, index, clean_path)
            noisy_path = noisy_dir / f"{uid}_noisy.wav"
            if noisy_path.exists() and not args.force:
                raise FileExistsError(f"{noisy_path} exists. Use --force to overwrite.")
            sf.write(noisy_path, noisy.squeeze(), sample_rate, subtype=args.subtype)

            resampled_clean_path = clean_output_path(clean_dir, args.split, index, clean_path)
            if not resampled_clean_path.exists() or args.force:
                sf.write(resampled_clean_path, clean_out.squeeze(), sample_rate, subtype=args.subtype)

            source_clean_path = str(Path(clean_path).resolve())
            clean_pair_path = str(resampled_clean_path.resolve())
            pair = {
                "id": uid,
                "split": args.split,
                "repeat_idx": int(repeat_idx),
                "clean_path": clean_pair_path,
                "noisy_path": str(noisy_path.resolve()),
                "sample_rate": int(sample_rate),
            }
            pairs.append(pair)
            manifest.append(
                {
                    **pair,
                    "seed": int(item_seed),
                    "source_clean_path": source_clean_path,
                    "resampled_clean_path": clean_pair_path,
                    "clean_num_samples": int(clean.shape[1]),
                    "noisy_num_samples": int(noisy.shape[1]),
                    "noise_path": str(Path(noise_path).resolve()),
                    "rir_path": str(Path(rir_path).resolve()),
                    "wind_noise_path": str(Path(wind_noise_path).resolve()) if wind_noise_path else None,
                    "noise_load_retries": noise_retries,
                    "rir_load_retries": rir_retries,
                    "wind_noise_load_retries": wind_noise_retries,
                    "noise_resampled_to": int(sample_rate),
                    "rir_resampled_to": int(sample_rate),
                    "simulation_config": str(Path(args.config).resolve()),
                    "degradation_config": degrad_cfgs,
                    "selected_degradations": selected_degrads,
                }
            )

            generated += 1
            if generated % args.log_interval == 0 or generated == total:
                print(f"{args.split}: generated {generated}/{total}", flush=True)

    write_json(output_root / "paired.json", pairs)
    write_jsonl(output_root / "metadata.jsonl", manifest)
    write_json(output_root / "metadata.json", manifest)
    write_pair_csv(output_root / "paired.csv", pairs)
    write_json(
        output_root / "summary.json",
        {
            "split": args.split,
            "num_clean": len(clean_paths),
            "num_repeats": int(args.num_repeats),
            "num_examples": len(pairs),
            "clean_json": str(Path(args.clean_json).resolve()),
            "noise_json": str(Path(args.noise_json).resolve()),
            "rir_json": str(Path(args.rir_json).resolve()),
            "wind_noise_json": str(Path(args.wind_noise_json).resolve()) if args.wind_noise_json else None,
            "output_root": str(output_root.resolve()),
            "seed": int(args.seed),
            "target_sample_rate": int(args.target_sample_rate),
            "audio_policy": "Resample clean/noise/RIR in memory to target_sample_rate; write fixed clean/noisy pairs at target_sample_rate.",
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export repeated fixed TAU degradations.")
    parser.add_argument("--split", choices=["train", "valid", "test", "same_clean_valid"])
    parser.add_argument("--clean-json", type=Path)
    parser.add_argument("--noise-json", type=Path)
    parser.add_argument("--rir-json", type=Path)
    parser.add_argument("--wind-noise-json", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--use-simulation-root", type=Path, default=DEFAULT_USE_SIMULATION_ROOT)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--target-sample-rate", type=int, default=DEFAULT_TARGET_SAMPLE_RATE)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subtype", default="FLOAT")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--max-audio-load-attempts", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    required = {
        "--split": args.split,
        "--clean-json": args.clean_json,
        "--noise-json": args.noise_json,
        "--rir-json": args.rir_json,
        "--config": args.config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")
    if args.num_repeats <= 0:
        parser.error("--num-repeats must be positive")
    if args.target_sample_rate <= 0:
        parser.error("--target-sample-rate must be positive")
    if args.max_audio_load_attempts <= 0:
        parser.error("--max-audio-load-attempts must be positive")
    return args


def main():
    args = parse_args()
    export_augmented_split(args)


if __name__ == "__main__":
    main()
