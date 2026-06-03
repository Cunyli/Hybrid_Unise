import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import PesqError, pesq
from pystoi import stoi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_scp", type=Path, required=True)
    parser.add_argument("--inf_scp", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pesq_invalid_manifest", type=Path)
    return parser.parse_args()


def read_scp(path):
    items = {}
    for line in path.read_text().splitlines():
        if line.strip():
            uid, wav_path = line.split(maxsplit=1)
            items[uid] = Path(wav_path)
    return items


def read_pesq_invalid_ids(path):
    if path is None:
        return set()
    return {
        record.get("uid") or record.get("id")
        for record in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def read_mono(path):
    wav, sample_rate = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sample_rate


def finite_values(rows, metric):
    return [row[metric] for row in rows if np.isfinite(row[metric])]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    refs = read_scp(args.ref_scp)
    infs = read_scp(args.inf_scp)
    pesq_invalid_ids = read_pesq_invalid_ids(args.pesq_invalid_manifest)
    rows = []
    for uid, inf_path in infs.items():
        ref, ref_sr = read_mono(refs[uid])
        inf, inf_sr = read_mono(inf_path)
        if ref_sr != inf_sr:
            raise ValueError(f"{uid}: sample-rate mismatch ref={ref_sr}, inf={inf_sr}")
        length = min(len(ref), len(inf))
        ref = ref[:length]
        inf = inf[:length]
        mode = "wb" if ref_sr == 16000 else "nb"

        row = {"uid": uid, "PESQ": np.nan, "ESTOI": np.nan, "error": ""}
        if uid in pesq_invalid_ids:
            row["error"] += "PESQ_MASKED;"
        else:
            try:
                pesq_score = float(pesq(ref_sr, ref, inf, mode, on_error=PesqError.RETURN_VALUES))
                if pesq_score >= 0.0:
                    row["PESQ"] = pesq_score
                else:
                    row["error"] += f"PESQ_RETURN:{pesq_score};"
            except Exception as exc:
                row["error"] += f"PESQ:{exc};"
        try:
            row["ESTOI"] = float(stoi(ref, inf, fs_sig=ref_sr, extended=True))
        except Exception as exc:
            row["error"] += f"ESTOI:{exc};"
        rows.append(row)
        print(f"{uid} PESQ={row['PESQ']} ESTOI={row['ESTOI']}", flush=True)

    summary = {}
    for metric in ("PESQ", "ESTOI"):
        values = finite_values(rows, metric)
        summary[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "count": len(values),
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }

    with (args.output_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "PESQ", "ESTOI", "error"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "RESULTS.txt").open("w") as f:
        for metric, values in summary.items():
            if values["mean"] is not None:
                f.write(f"{metric}: {values['mean']:.4f}\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
