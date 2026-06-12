import argparse
import csv
import json
from pathlib import Path


TASKS = ("cs", "sv")


def infer_task(row):
    text = " ".join(
        str(row.get(key, ""))
        for key in ("uid", "clean_filepath", "noisy_filepath", "source_clean_path")
    ).lower()
    for task in TASKS:
        if f"_{task}" in text or text.endswith(task):
            return task
    return "other"


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split TAU paired.csv by task suffix, e.g. cs vs sv.")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    args = parser.parse_args()

    rows = read_csv(args.pair_manifest)
    if not rows:
        raise ValueError(f"No rows found in {args.pair_manifest}")

    grouped = {task: [] for task in TASKS}
    grouped["other"] = []
    for row in rows:
        grouped[infer_task(row)].append(row)

    fieldnames = list(rows[0].keys())
    summary = {
        "source_pair_manifest": str(args.pair_manifest.resolve()),
        "split_name": args.split_name,
        "total_rows": len(rows),
        "tasks": {},
    }
    for task, task_rows in grouped.items():
        if not task_rows:
            continue
        out_dir = args.output_root / f"{args.split_name}_{task}"
        write_csv(out_dir / "paired.csv", task_rows, fieldnames)
        write_json(
            out_dir / "summary.json",
            {
                "source_pair_manifest": str(args.pair_manifest.resolve()),
                "split_name": f"{args.split_name}_{task}",
                "task": task,
                "num_examples": len(task_rows),
            },
        )
        summary["tasks"][task] = {
            "num_examples": len(task_rows),
            "pair_manifest": str((out_dir / "paired.csv").resolve()),
        }

    write_json(args.output_root / f"{args.split_name}_task_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
