import argparse
import csv
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import yaml
from pesq import PesqError, pesq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import DataModule
from model import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--save_examples", type=int, default=5)
    return parser.parse_args()


def unpack_batch(batch):
    if len(batch) == 8:
        mode, enroll, mix, speech, interf, fs, lengths, names = batch
    elif len(batch) == 7:
        mode, enroll, mix, speech, fs, lengths, names = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return mode, enroll, mix, speech, fs, lengths, names


def pesq_score(sample_rate, reference, degraded):
    if sample_rate not in (8000, 16000):
        raise ValueError(f"PESQ supports 8000 or 16000 Hz, got {sample_rate}")
    mode = "wb" if sample_rate == 16000 else "nb"
    return pesq(sample_rate, reference, degraded, mode)


def mean(values):
    return float(sum(values) / len(values)) if values else None


def build_dataloader(config, split):
    kwargs = dict(config["dataset_config"][f"{split}_kwargs"])
    kwargs["batch_size"] = 1
    kwargs["num_workers"] = 1
    kwargs["prefetch"] = 0
    data_module = DataModule(
        train_kwargs=config["dataset_config"]["train_kwargs"],
        val_kwargs=kwargs if split == "val" else config["dataset_config"]["val_kwargs"],
        test_kwargs=kwargs if split == "test" else config["dataset_config"]["test_kwargs"],
    )
    data_module.setup("fit" if split == "val" else "test")
    return data_module.val_dataloader() if split == "val" else data_module.test_dataloader()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(config=config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    dataloader = build_dataloader(config, args.split)
    examples_dir = args.output_dir / "examples"
    if args.save_examples > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    enhanced_scores = []
    noisy_scores = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            mode, enroll, mix, speech, fs, lengths, names = unpack_batch(batch)
            mix = mix.to(device)
            speech = speech.to(device)
            enroll_mel = None
            enroll_feats = None
            if enroll is not None:
                enroll = enroll.to(device)
                enroll_mel = model.stft_logmel(enroll)
                enroll_feats = model.extract_semantic_features(enroll)
            mix_mel = model.stft_logmel(mix)
            mix_feats = model.extract_semantic_features(mix)
            global_ids, semantic_ids = model.dnn.generate(
                task_name=mode,
                enroll_mel=enroll_mel,
                enroll_feats=enroll_feats,
                mix_mel=mix_mel,
                mix_feats=mix_feats,
                do_sample=False,
            )
            enhanced = model.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)
            if not torch.is_tensor(enhanced):
                enhanced = torch.from_numpy(enhanced).to(device)
            enhanced = enhanced[:, : speech.size(-1)]

            for sample_idx, name in enumerate(names):
                sample_rate = int(fs[sample_idx])
                sample_length = min(int(lengths[sample_idx]), speech.size(-1), mix.size(-1), enhanced.size(-1))
                clean_np = speech[sample_idx, :sample_length].detach().cpu().numpy()
                noisy_np = mix[sample_idx, :sample_length].detach().cpu().numpy()
                enhanced_np = enhanced[sample_idx, :sample_length].detach().cpu().numpy()

                enhanced_pesq = None
                noisy_pesq = None
                try:
                    enhanced_pesq = pesq_score(sample_rate, clean_np, enhanced_np)
                    enhanced_scores.append(enhanced_pesq)
                except PesqError:
                    pass
                try:
                    noisy_pesq = pesq_score(sample_rate, clean_np, noisy_np)
                    noisy_scores.append(noisy_pesq)
                except PesqError:
                    pass

                rows.append(
                    {
                        "name": name,
                        "sample_rate": sample_rate,
                        "length": sample_length,
                        "enhanced_pesq": enhanced_pesq,
                        "noisy_pesq": noisy_pesq,
                    }
                )
                if len(rows) <= args.save_examples:
                    sf.write(examples_dir / f"{name}_clean.wav", clean_np, sample_rate)
                    sf.write(examples_dir / f"{name}_noisy.wav", noisy_np, sample_rate)
                    sf.write(examples_dir / f"{name}_enhanced.wav", enhanced_np, sample_rate)

            print(
                f"{args.split} batch {batch_idx + 1}/{len(dataloader)} "
                f"enhanced_pesq={mean(enhanced_scores)} noisy_pesq={mean(noisy_scores)}",
                flush=True,
            )

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_examples": len(rows),
        "enhanced_pesq_mean": mean(enhanced_scores),
        "enhanced_pesq_count": len(enhanced_scores),
        "noisy_pesq_mean": mean(noisy_scores),
        "noisy_pesq_count": len(noisy_scores),
    }
    with (args.output_dir / f"{args.split}_generation_pesq.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sample_rate", "length", "enhanced_pesq", "noisy_pesq"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
