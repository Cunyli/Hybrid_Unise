import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import DataModule
from model.bicodec import BiCodecTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def unpack_batch(batch):
    if len(batch) == 8:
        mode, enroll, mix, speech, interf, fs, lengths, names = batch
    elif len(batch) == 7:
        mode, enroll, mix, speech, fs, lengths, names = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return mode, mix, speech, fs, lengths, names


def build_dataloader(config, split):
    kwargs = dict(config["dataset_config"][f"{split}_kwargs"])
    kwargs["batch_size"] = 1
    kwargs["num_workers"] = 1
    kwargs["prefetch"] = 0

    data_module = DataModule(
        train_kwargs=kwargs if split == "train" else config["dataset_config"]["train_kwargs"],
        val_kwargs=kwargs if split == "val" else config["dataset_config"]["val_kwargs"],
        test_kwargs=kwargs if split == "test" else config["dataset_config"]["test_kwargs"],
    )
    data_module.setup("test" if split == "test" else "fit")
    if split == "train":
        return data_module.train_dataloader()
    if split == "val":
        return data_module.val_dataloader()
    return data_module.test_dataloader()


def token_stats(clean_tokens, noisy_tokens):
    clean = clean_tokens.reshape(-1)
    noisy = noisy_tokens.reshape(-1)
    length = min(clean.numel(), noisy.numel())
    clean = clean[:length]
    noisy = noisy[:length]
    matches = clean == noisy
    unique_clean = torch.unique(clean).numel()
    unique_noisy = torch.unique(noisy).numel()
    return {
        "num_tokens": int(length),
        "exact_match": float(matches.float().mean().item()) if length else None,
        "unique_clean": int(unique_clean),
        "unique_noisy": int(unique_noisy),
    }


def weighted_mean(rows, key, weight_key):
    total_weight = sum(row[weight_key] for row in rows if row[key] is not None)
    if total_weight == 0:
        return None
    return float(sum(row[key] * row[weight_key] for row in rows if row[key] is not None) / total_weight)


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BiCodecTokenizer(model_dir=config["codec_ckpt_dir"]).to(device).eval()
    tokenizer.requires_grad_(False)
    dataloader = build_dataloader(config, args.split)

    rows = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            mode, mix, speech, fs, lengths, names = unpack_batch(batch)
            mix = mix.to(device)
            speech = speech.to(device)

            clean_global, clean_semantic = tokenizer.tokenize(speech)
            noisy_global, noisy_semantic = tokenizer.tokenize(mix)

            for sample_idx, name in enumerate(names):
                global_stats = token_stats(clean_global[sample_idx], noisy_global[sample_idx])
                semantic_stats = token_stats(clean_semantic[sample_idx], noisy_semantic[sample_idx])
                row = {
                    "name": name,
                    "sample_rate": int(fs[sample_idx]),
                    "length": int(lengths[sample_idx]),
                    "global_tokens": global_stats["num_tokens"],
                    "global_exact_match": global_stats["exact_match"],
                    "global_unique_clean": global_stats["unique_clean"],
                    "global_unique_noisy": global_stats["unique_noisy"],
                    "semantic_tokens": semantic_stats["num_tokens"],
                    "semantic_exact_match": semantic_stats["exact_match"],
                    "semantic_unique_clean": semantic_stats["unique_clean"],
                    "semantic_unique_noisy": semantic_stats["unique_noisy"],
                }
                rows.append(row)

            print(
                f"{args.split} batch {batch_idx + 1}/{len(dataloader)} "
                f"semantic_match={weighted_mean(rows, 'semantic_exact_match', 'semantic_tokens')} "
                f"global_match={weighted_mean(rows, 'global_exact_match', 'global_tokens')}",
                flush=True,
            )

    summary = {
        "config": str(args.config),
        "split": args.split,
        "num_examples": len(rows),
        "semantic_exact_match": weighted_mean(rows, "semantic_exact_match", "semantic_tokens"),
        "global_exact_match": weighted_mean(rows, "global_exact_match", "global_tokens"),
        "semantic_tokens": sum(row["semantic_tokens"] for row in rows),
        "global_tokens": sum(row["global_tokens"] for row in rows),
    }

    csv_path = args.output_dir / f"{args.split}_token_similarity.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "name",
            "sample_rate",
            "length",
            "global_tokens",
            "global_exact_match",
            "global_unique_clean",
            "global_unique_noisy",
            "semantic_tokens",
            "semantic_exact_match",
            "semantic_unique_clean",
            "semantic_unique_noisy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
