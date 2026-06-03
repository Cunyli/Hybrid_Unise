import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from pesq import PesqError, pesq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import DataModule
from model.bicodec import BiCodecTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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
    return mode, mix, speech, fs, lengths, names


def pesq_score(sample_rate, reference, degraded):
    if sample_rate not in (8000, 16000):
        raise ValueError(f"PESQ supports 8000 or 16000 Hz, got {sample_rate}")
    mode = "wb" if sample_rate == 16000 else "nb"
    return pesq(sample_rate, reference, degraded, mode)


def mean(values):
    return float(sum(values) / len(values)) if values else None


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BiCodecTokenizer(model_dir=config["codec_ckpt_dir"]).to(device).eval()

    kwargs = dict(config["dataset_config"][f"{args.split}_kwargs"])
    kwargs["batch_size"] = 1
    kwargs["num_workers"] = 1
    kwargs["prefetch"] = 0

    data_module = DataModule(
        train_kwargs=config["dataset_config"]["train_kwargs"],
        val_kwargs=kwargs if args.split == "val" else config["dataset_config"]["val_kwargs"],
        test_kwargs=kwargs if args.split == "test" else config["dataset_config"]["test_kwargs"],
    )
    data_module.setup("fit" if args.split == "val" else "test")
    dataloader = data_module.val_dataloader() if args.split == "val" else data_module.test_dataloader()

    rows = []
    oracle_scores = []
    noisy_scores = []
    examples_dir = args.output_dir / "examples"
    if args.save_examples > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            mode, mix, speech, fs, lengths, names = unpack_batch(batch)
            mix = mix.to(device)
            speech = speech.to(device)

            global_tokens, semantic_tokens = tokenizer.tokenize(speech)
            recon = tokenizer.detokenize(global_tokens, semantic_tokens).squeeze(1)
            if not torch.is_tensor(recon):
                recon = torch.from_numpy(recon).to(device)
            recon = recon[:, :speech.size(-1)]

            for sample_idx, name in enumerate(names):
                sample_rate = int(fs[sample_idx])
                sample_length = min(int(lengths[sample_idx]), speech.size(-1), recon.size(-1), mix.size(-1))
                clean_np = speech[sample_idx, :sample_length].detach().cpu().numpy()
                recon_np = recon[sample_idx, :sample_length].detach().cpu().numpy()
                noisy_np = mix[sample_idx, :sample_length].detach().cpu().numpy()

                oracle_pesq = None
                noisy_pesq = None
                try:
                    oracle_pesq = pesq_score(sample_rate, clean_np, recon_np)
                    oracle_scores.append(oracle_pesq)
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
                        "oracle_pesq": oracle_pesq,
                        "noisy_pesq": noisy_pesq,
                    }
                )

                if len(rows) <= args.save_examples:
                    sf.write(examples_dir / f"{name}_clean.wav", clean_np, sample_rate)
                    sf.write(examples_dir / f"{name}_oracle.wav", recon_np, sample_rate)
                    sf.write(examples_dir / f"{name}_noisy.wav", noisy_np, sample_rate)

            print(
                f"{args.split} batch {batch_idx + 1}/{len(dataloader)} "
                f"oracle_pesq={mean(oracle_scores)} noisy_pesq={mean(noisy_scores)}",
                flush=True,
            )

    summary = {
        "config": str(args.config),
        "split": args.split,
        "num_examples": len(rows),
        "oracle_pesq_mean": mean(oracle_scores),
        "oracle_pesq_count": len(oracle_scores),
        "noisy_pesq_mean": mean(noisy_scores),
        "noisy_pesq_count": len(noisy_scores),
    }

    with (args.output_dir / f"{args.split}_oracle_pesq.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sample_rate", "length", "oracle_pesq", "noisy_pesq"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
