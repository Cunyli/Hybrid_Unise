import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


AVQI_ROOT = Path("/scratch/work/lil14/avqi")
DEFAULT_PAIR_CSV = Path("/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv")
DEFAULT_RESULTS_DIR = AVQI_ROOT / "avqi_output" / "unise_tau_ablation_6models"

STEP_VERSIONS = {
    "highpass": "praat",
    "read_and_resample": "praat",
    "sv_length_norm": "praat",
    "cs_voiced_segments": "praat",
    "concatenate": "praat",
    "cpps": "praat",
    "slope": "praat",
    "tilt": "praat",
    "shimmer": "praat",
    "hnr": "praat",
    "pitch": "praat",
}


def import_avqi(avqi_root):
    root = Path(avqi_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from avqi_code import run_avqi

    return run_avqi


def parse_clean_stem(clean_path):
    stem = Path(clean_path).stem
    for suffix in ("_clean", "_noisy"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] == "tau" and parts[2].startswith("r") and parts[3].isdigit():
        stem = "_".join(parts[4:])
    elif len(parts) >= 4 and parts[0] == "tau" and parts[2].isdigit():
        stem = "_".join(parts[3:])
    if "_" not in stem:
        raise ValueError(f"Cannot parse speaker/task from clean filename: {clean_path}")
    speaker, task = stem.rsplit("_", 1)
    if task not in {"cs", "sv"}:
        raise ValueError(f"Unexpected task '{task}' from clean filename: {clean_path}")
    return speaker, task


def repeat_id(uid):
    parts = str(uid).split("_")
    for part in parts:
        if len(part) >= 2 and part[0] == "r" and part[1:].isdigit():
            return part
    return "r00"


def health_group(speaker):
    return "H" if speaker.startswith("V") else "P"


def load_pairs(pair_csv):
    speakers = {}
    rows_by_uid = {}
    with Path(pair_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            speaker, task = parse_clean_stem(row["clean_filepath"])
            row = dict(row)
            row["speaker"] = speaker
            row["task"] = task
            row["repeat_id"] = repeat_id(row["uid"])
            row["pair_id"] = f"{speaker}__{row['repeat_id']}"
            speakers.setdefault(row["pair_id"], {})[task] = row
            rows_by_uid[row["uid"]] = row
    return speakers, rows_by_uid


def enhanced_candidates(enhanced_dir, row):
    enhanced_dir = Path(enhanced_dir)
    noisy = Path(row["noisy_filepath"])
    clean = Path(row["clean_filepath"])
    return [
        enhanced_dir / f"{row['uid']}.wav",
        enhanced_dir / f"{clean.stem}.wav",
        enhanced_dir / noisy.name,
        enhanced_dir / f"{row['uid']}_noisy.wav",
        enhanced_dir / f"{noisy.stem}.wav",
        enhanced_dir / "wav" / f"{row['uid']}.wav",
        enhanced_dir / "wav" / f"{clean.stem}.wav",
        enhanced_dir / "wav" / noisy.name,
    ]


def resolve_enhanced(enhanced_dir, row):
    for candidate in enhanced_candidates(enhanced_dir, row):
        if candidate.is_file():
            return candidate
    return None


def path_for_condition(condition, row, condition_dirs):
    if condition == "clean":
        return Path(row["clean_filepath"])
    if condition == "degraded":
        return Path(row["noisy_filepath"])
    enhanced_dir = condition_dirs[condition]
    if isinstance(enhanced_dir, dict):
        task_dir = enhanced_dir[row["task"]]
        return resolve_enhanced(task_dir, row)
    return resolve_enhanced(enhanced_dir, row)


def score_condition(run_avqi, condition, speakers, condition_dirs, target_sr, speaking_type):
    results = {"H": {}, "P": {}}
    skipped = []
    complete = {pair_id: tasks for pair_id, tasks in speakers.items() if "cs" in tasks and "sv" in tasks}
    for idx, (pair_id, tasks) in enumerate(sorted(complete.items()), start=1):
        cs_path = path_for_condition(condition, tasks["cs"], condition_dirs)
        sv_path = path_for_condition(condition, tasks["sv"], condition_dirs)
        if cs_path is None or sv_path is None or not cs_path.is_file() or not sv_path.is_file():
            skipped.append(pair_id)
            continue
        speaker = tasks["cs"]["speaker"]
        print(f"[{condition}] {idx}/{len(complete)} {pair_id}")
        result = run_avqi(
            str(sv_path),
            str(cs_path),
            target_sr=target_sr,
            speaking_type=speaking_type,
            step_versions=STEP_VERSIONS,
            remove_sv_silence_with_sox=False,
        )
        results[health_group(speaker)][pair_id] = result
    return results, skipped


def flatten_avqi(results):
    values = []
    for speakers in results.values():
        for metrics in speakers.values():
            value = metrics.get("avqi")
            if value is not None:
                values.append(float(value))
    return values


def write_summary(results_dir, manifest):
    rows = []
    for condition, info in manifest["conditions"].items():
        json_path = Path(info["json"])
        if not json_path.is_file():
            continue
        values = flatten_avqi(json.loads(json_path.read_text()))
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        else:
            variance = 0.0
        rows.append(
            {
                "condition": condition,
                "n": len(values),
                "avqi_mean": mean,
                "avqi_std": variance ** 0.5,
                "skipped": len(info.get("skipped_speakers", [])),
            }
        )
    out_csv = Path(results_dir) / "tau_fixed_avqi_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "n", "avqi_mean", "avqi_std", "skipped"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {out_csv}")


def copy_audio(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is not None and Path(src).is_file():
        shutil.copy2(src, dst)


def write_samples(speakers, condition_dirs, conditions, samples_dir):
    samples_dir = Path(samples_dir)
    complete = {pair_id: tasks for pair_id, tasks in speakers.items() if "cs" in tasks and "sv" in tasks}
    for pair_id, tasks in sorted(complete.items()):
        for task, row in sorted(tasks.items()):
            item_dir = samples_dir / row["speaker"] / row["repeat_id"] / task
            copy_audio(Path(row["clean_filepath"]), item_dir / "clean.wav")
            copy_audio(Path(row["noisy_filepath"]), item_dir / "degraded.wav")
            for condition in conditions:
                if condition in {"clean", "degraded"}:
                    continue
                enhanced = path_for_condition(condition, row, condition_dirs)
                copy_audio(enhanced, item_dir / f"{condition}.wav")
    print(f"[SAVE] samples under {samples_dir}")


def parse_condition(values):
    condition_dirs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--condition must be NAME=DIR, got: {value}")
        name, path = value.split("=", 1)
        condition_dirs[name] = Path(path)
    return condition_dirs


def parse_task_ensemble(values):
    condition_dirs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--task-ensemble must be NAME=CS_DIR,SV_DIR, got: {value}")
        name, paths = value.split("=", 1)
        parts = paths.split(",")
        if len(parts) != 2:
            raise ValueError(f"--task-ensemble must be NAME=CS_DIR,SV_DIR, got: {value}")
        condition_dirs[name] = {"cs": Path(parts[0]), "sv": Path(parts[1])}
    return condition_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Score UniSE TAU enhanced outputs with AVQI-compatible JSON/plots inputs.")
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--avqi-root", type=Path, default=AVQI_ROOT)
    parser.add_argument("--db-name", default="TAU_fixed")
    parser.add_argument("--speaking-type", choices=["both", "cs", "sv"], default="both")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--condition", action="append", default=[], help="Enhanced condition as NAME=DIR")
    parser.add_argument("--task-ensemble", action="append", default=[], help="Task ensemble as NAME=CS_DIR,SV_DIR")
    parser.add_argument("--skip-clean-degraded", action="store_true")
    parser.add_argument("--write-samples", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_avqi = import_avqi(args.avqi_root)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    speakers, _ = load_pairs(args.pair_csv)

    condition_dirs = {}
    condition_dirs.update(parse_condition(args.condition))
    condition_dirs.update(parse_task_ensemble(args.task_ensemble))

    conditions = []
    if not args.skip_clean_degraded:
        conditions.extend(["clean", "degraded"])
    conditions.extend(condition_dirs)

    manifest = {
        "pair_csv": str(args.pair_csv),
        "target_sr": args.target_sr,
        "speaking_type": args.speaking_type,
        "conditions": {},
    }
    for condition in conditions:
        results, skipped = score_condition(
            run_avqi,
            condition,
            speakers,
            condition_dirs,
            args.target_sr,
            args.speaking_type,
        )
        out_json = args.results_dir / f"avqi_results_{args.db_name}_{condition}_{args.speaking_type}.json"
        out_json.write_text(json.dumps(results, indent=4, ensure_ascii=False) + "\n")
        n_scored = sum(len(group) for group in results.values())
        manifest["conditions"][condition] = {
            "enhanced_dir": str(condition_dirs.get(condition)) if condition in condition_dirs else None,
            "json": str(out_json),
            "n_scored": n_scored,
            "skipped_speakers": skipped,
        }
        print(f"[SAVE] {out_json} ({n_scored} speakers, skipped={len(skipped)})")

    manifest_path = args.results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n")
    write_summary(args.results_dir, manifest)
    if args.write_samples:
        write_samples(speakers, condition_dirs, conditions, args.results_dir / "samples")


if __name__ == "__main__":
    main()
