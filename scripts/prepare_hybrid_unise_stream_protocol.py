#!/usr/bin/env python
import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.hybrid_webdataset_protocol import (  # noqa: E402
    DEFAULT_HYBRID_PROTOCOL_ROOT,
    _check_not_teamwork,
    manifest_item,
    shard_key,
    write_jsonl,
)


DEFAULT_PROTOCOL_NAME = "hybrid_unise_v1_stream_80_10_10"


def iter_manifest_items(path):
    path = _check_not_teamwork(path)
    tar_exists_cache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status", "done") != "done":
                continue
            item = manifest_item(row)
            key = shard_key(item)
            tar_exists = tar_exists_cache.get(key)
            if tar_exists is None:
                tar_exists = (Path(item["_shard_dir"]) / item["shard"]).is_file()
                tar_exists_cache[key] = tar_exists
            if tar_exists:
                yield item


def split_keys(keys, seed, train_ratio=0.8, valid_ratio=0.1):
    keys = list(keys)
    random.Random(int(seed)).shuffle(keys)
    n_total = len(keys)
    if n_total >= 3:
        n_train = max(1, int(round(n_total * float(train_ratio))))
        n_train = min(n_train, n_total - 2)
        n_valid = max(1, int(round(n_total * float(valid_ratio))))
        n_valid = min(n_valid, n_total - n_train - 1)
        return {
            "train": keys[:n_train],
            "valid": keys[n_train : n_train + n_valid],
            "test": keys[n_train + n_valid :],
        }
    n_train = int(round(n_total * float(train_ratio)))
    n_valid = int(round(n_total * float(valid_ratio)))
    n_train = min(max(n_train, 1 if n_total >= 3 else n_train), n_total)
    n_valid = min(max(n_valid, 1 if n_total - n_train >= 2 else n_valid), n_total - n_train)
    train = keys[:n_train]
    valid = keys[n_train : n_train + n_valid]
    test = keys[n_train + n_valid :]
    if n_total >= 3 and not test:
        test = [valid.pop()] if valid else [train.pop()]
    return {"train": train, "valid": valid, "test": test}


def scan_shards(manifest_path, max_shards=None):
    stats = {}
    for item in iter_manifest_items(manifest_path):
        key = shard_key(item)
        record = stats.get(key)
        if record is None:
            record = {
                "_shard_dir": item["_shard_dir"],
                "shard": item["shard"],
                "role": item.get("role"),
                "sample_count": 0,
                "dataset_counts": Counter(),
            }
            stats[key] = record
            if max_shards is not None and len(stats) > int(max_shards):
                stats.pop(key, None)
                break
        record["sample_count"] += 1
        record["dataset_counts"].update([item.get("dataset", "")])
    if not stats:
        raise ValueError(f"No usable shards found from {manifest_path}")
    return stats


def shard_records(stats, keys):
    records = []
    for key in keys:
        record = stats[key]
        records.append(
            {
                "_shard_dir": record["_shard_dir"],
                "shard": record["shard"],
                "role": record.get("role"),
                "dataset_counts": dict(sorted(record["dataset_counts"].items())),
                "sample_count": int(record["sample_count"]),
            }
        )
    return records


def reservoir_add(reservoir, item, count, rng, seen):
    seen += 1
    if len(reservoir) < int(count):
        reservoir.append(item)
    else:
        j = rng.randrange(seen)
        if j < int(count):
            reservoir[j] = item
    return seen


def sample_rows_for_split(manifest_path, split_keys_by_name, counts, seed, stop_when_full=False):
    wanted = {
        split: set(keys)
        for split, keys in split_keys_by_name.items()
        if split in counts and int(counts[split]) > 0
    }
    rngs = {split: random.Random(int(seed) + idx * 7919) for idx, split in enumerate(sorted(wanted))}
    reservoirs = {split: [] for split in wanted}
    seen = {split: 0 for split in wanted}
    for item in iter_manifest_items(manifest_path):
        key = shard_key(item)
        for split, split_keys in wanted.items():
            if key in split_keys:
                seen[split] = reservoir_add(reservoirs[split], item, counts[split], rngs[split], seen[split])
                break
        if stop_when_full and all(len(reservoirs[split]) >= int(counts[split]) for split in wanted):
            break
    return reservoirs


def make_recipe(split, clean, noise, rir, index, seed, target_sample_rate, cut_duration, degradation_config):
    recipe_seed = int(seed) + index
    return {
        "uid": f"{split}_{index:08d}_{clean['key']}",
        "split": split,
        "seed": recipe_seed,
        "target_sample_rate": int(target_sample_rate),
        "cut_duration": float(cut_duration),
        "clean": {k: clean.get(k) for k in ("_shard_dir", "shard", "audio_member", "json_member", "key", "dataset", "role")},
        "noise": {k: noise.get(k) for k in ("_shard_dir", "shard", "audio_member", "json_member", "key", "dataset", "role")},
        "rir": {k: rir.get(k) for k in ("_shard_dir", "shard", "audio_member", "json_member", "key", "dataset", "role")},
        "degradation_config": degradation_config,
    }


def fixed_recipes(split, clean_rows, noise_rows, rir_rows, count, seed, target_sample_rate, cut_duration, degradation_config):
    if not clean_rows or not noise_rows or not rir_rows:
        raise ValueError(f"{split} fixed recipes require non-empty clean/noise/rir pools")
    rng = random.Random(int(seed))
    clean_order = list(clean_rows)
    rng.shuffle(clean_order)
    count = min(int(count), len(clean_order))
    recipes = []
    for idx, clean in enumerate(clean_order[:count]):
        noise = noise_rows[rng.randrange(len(noise_rows))]
        rir = rir_rows[rng.randrange(len(rir_rows))]
        recipes.append(
            make_recipe(
                split,
                clean,
                noise,
                rir,
                idx,
                int(seed) + idx * 1000003,
                target_sample_rate,
                cut_duration,
                degradation_config,
            )
        )
    return recipes


def role_summary(records):
    sample_count = sum(record["sample_count"] for record in records)
    datasets = Counter()
    for record in records:
        datasets.update(record["dataset_counts"])
    return {
        "shard_count": len(records),
        "sample_count": sample_count,
        "dataset_counts": dict(sorted(datasets.items())),
    }


def write_role_splits(output_root, role, stats, split):
    for name in ("train", "valid", "test"):
        write_jsonl(output_root / name / f"{role}_shards.jsonl", shard_records(stats, split[name]))


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed Hybrid-UniSE WebDataset stream protocol artifacts")
    parser.add_argument("--active-root", type=Path, default=DEFAULT_HYBRID_PROTOCOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--protocol-name", default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--valid-recipes", type=int, default=1000)
    parser.add_argument("--test-recipes", type=int, default=1000)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--cut-duration", type=float, default=5.0)
    parser.add_argument("--degradation-config", default="./conf/use_simulation_phone_room_16k.yaml")
    parser.add_argument("--degradation-version", default="hybrid_unise_v1")
    parser.add_argument("--max-clean-shards", type=int)
    parser.add_argument("--max-noise-shards", type=int)
    parser.add_argument("--max-rir-shards", type=int)
    parser.add_argument(
        "--fast-recipe-sampling",
        action="store_true",
        help="Stop recipe sampling once each reservoir is full; use only for small smoke protocols.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    active_root = _check_not_teamwork(args.active_root)
    output_root = args.output_root or (active_root / "splits" / args.protocol_name)
    output_root = _check_not_teamwork(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists; pass --overwrite only for an intentional regeneration")

    manifest_paths = {
        "clean": active_root / "v1_verified" / "clean" / "manifest.jsonl",
        "noise": active_root / "v1_verified" / "noise" / "manifest.jsonl",
        "rir": active_root / "v1_verified" / "rir" / "manifest.jsonl",
    }
    max_shards = {
        "clean": args.max_clean_shards,
        "noise": args.max_noise_shards,
        "rir": args.max_rir_shards,
    }
    stats = {role: scan_shards(path, max_shards=max_shards[role]) for role, path in manifest_paths.items()}

    splits = {
        "clean": split_keys(sorted(stats["clean"]), args.seed),
        "noise": split_keys(sorted(stats["noise"]), args.seed + 11),
        "rir": split_keys(sorted(stats["rir"]), args.seed + 23),
    }
    for role in ("clean", "noise", "rir"):
        write_role_splits(output_root, role, stats[role], splits[role])

    degradation_config = {"config_path": args.degradation_config, "version": args.degradation_version}
    clean_rows = sample_rows_for_split(
        manifest_paths["clean"],
        {"valid": splits["clean"]["valid"], "test": splits["clean"]["test"]},
        {"valid": args.valid_recipes, "test": args.test_recipes},
        args.seed + 1000,
        stop_when_full=args.fast_recipe_sampling,
    )
    noise_counts = {"valid": max(args.valid_recipes, 1), "test": max(args.test_recipes, 1)}
    rir_counts = {"valid": max(args.valid_recipes, 1), "test": max(args.test_recipes, 1)}
    noise_rows = sample_rows_for_split(
        manifest_paths["noise"],
        {
            "valid": splits["noise"]["valid"] or splits["noise"]["train"],
            "test": splits["noise"]["test"] or splits["noise"]["train"],
        },
        noise_counts,
        args.seed + 2000,
        stop_when_full=args.fast_recipe_sampling,
    )
    rir_rows = sample_rows_for_split(
        manifest_paths["rir"],
        {
            "valid": splits["rir"]["valid"] or splits["rir"]["train"],
            "test": splits["rir"]["test"] or splits["rir"]["train"],
        },
        rir_counts,
        args.seed + 3000,
        stop_when_full=args.fast_recipe_sampling,
    )
    valid_recipes = fixed_recipes(
        "valid",
        clean_rows["valid"],
        noise_rows["valid"],
        rir_rows["valid"],
        args.valid_recipes,
        args.seed + 101,
        args.target_sample_rate,
        args.cut_duration,
        degradation_config,
    )
    test_recipes = fixed_recipes(
        "test",
        clean_rows["test"],
        noise_rows["test"],
        rir_rows["test"],
        args.test_recipes,
        args.seed + 202,
        args.target_sample_rate,
        args.cut_duration,
        degradation_config,
    )
    write_jsonl(output_root / "valid" / "fixed_recipes.jsonl", valid_recipes)
    write_jsonl(output_root / "test" / "fixed_recipes.jsonl", test_recipes)

    clean_sets = {split: set(splits["clean"][split]) for split in ("train", "valid", "test")}
    if clean_sets["train"] & clean_sets["valid"] or clean_sets["train"] & clean_sets["test"] or clean_sets["valid"] & clean_sets["test"]:
        raise RuntimeError("clean train/valid/test shard sets are not disjoint")

    summary = {
        "protocol_name": args.protocol_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_root": str(active_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "source_manifest_paths": {role: str(path) for role, path in manifest_paths.items()},
        "target_sample_rate": args.target_sample_rate,
        "cut_duration": args.cut_duration,
        "degradation_config": degradation_config,
        "train_streaming": {
            "unit": "shard",
            "sample_rows_embedded": False,
            "reader": "sequential tar audio-member scan",
        },
        "splits": {
            split: {
                role: role_summary(shard_records(stats[role], splits[role][split]))
                for role in ("clean", "noise", "rir")
            }
            for split in ("train", "valid", "test")
        },
        "fixed_recipe_counts": {"valid": len(valid_recipes), "test": len(test_recipes)},
        "clean_shard_disjoint": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                f"# {args.protocol_name}",
                "",
                "Canonical Hybrid-UniSE WebDataset protocol.",
                "",
                "- `train/*_shards.jsonl` contains shard-level training pools only; sample rows are not embedded.",
                "- Training streams audio members sequentially from tar shards with deterministic epoch/worker shuffling.",
                "- `valid/fixed_recipes.jsonl` and `test/fixed_recipes.jsonl` are deterministic reusable degradation recipes.",
                "- Component WebDataset directories and `v1_verified` manifests are not modified.",
                "- Reuse this directory for later experiments instead of regenerating it.",
                "",
                f"Created at: {summary['created_at']}",
                f"Seed: {args.seed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_root)


if __name__ == "__main__":
    main()
