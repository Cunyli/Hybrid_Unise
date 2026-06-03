#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/unified-audio/QuarkAudio-UniSE}"
INPUT_ROOT="${INPUT_ROOT:-/scratch/work/lil14/data/TAU_SD_degraded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/work/lil14/data/TAU_SD_enhanced/unise}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/tau_sd_work}"
FLAT_INPUT="${FLAT_INPUT:-$WORK_DIR/input_flat}"
FLAT_OUTPUT="${FLAT_OUTPUT:-$WORK_DIR/output_flat}"
CONFIG_OUT="${CONFIG_OUT:-$WORK_DIR/config.yaml}"
CKPT_PATH="${CKPT_PATH:-$ROOT_DIR/checkpoints/epoch=20-step=109367.ckpt}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/pretrained/Spark-TTS-0.5B}"

cd "$ROOT_DIR"
test -f "$CKPT_PATH"
test -d "$MODEL_DIR"

mkdir -p "$FLAT_INPUT" "$FLAT_OUTPUT" "$OUTPUT_ROOT" "$(dirname "$CONFIG_OUT")"

find "$INPUT_ROOT" -type f -name "*.wav" | sort | while read -r f; do
  ln -sf "$f" "$FLAT_INPUT/$(basename "$f")"
done

python - "$CONFIG_OUT" "$CKPT_PATH" "$MODEL_DIR" "$FLAT_INPUT" <<'PY'
import sys
import yaml
from pathlib import Path

config_out, ckpt_path, model_dir, flat_input = sys.argv[1:]
cfg = yaml.safe_load(Path("conf/config.yaml").read_text())
cfg["ckpt_path"] = ckpt_path
cfg["codec_ckpt_dir"] = model_dir
cfg["dataset_config"]["test_kwargs"] = {
    "batch_size": 1,
    "num_workers": 1,
    "prefetch": 1,
    "mode": "se",
    "data_enroll_dir": None,
    "enroll_duration": 5.0,
    "data_src_dir": flat_input,
    "data_tgt_dir": flat_input,
}
Path(config_out).write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Wrote", config_out)
PY

python test.py --config "$CONFIG_OUT" --save_enhanced "$FLAT_OUTPUT"

find "$INPUT_ROOT" -type f -name "*.wav" | sort | while read -r src; do
  rel="${src#$INPUT_ROOT/}"
  name="$(basename "$src")"
  enhanced="$FLAT_OUTPUT/$name"

  if [[ ! -f "$enhanced" ]]; then
    echo "Missing enhanced file: $enhanced"
    exit 1
  fi

  out="$OUTPUT_ROOT/$rel"
  mkdir -p "$(dirname "$out")"
  cp "$enhanced" "$out"
done

echo "Input wav:"
find "$INPUT_ROOT" -type f -name "*.wav" | wc -l

echo "Flat output wav:"
find "$FLAT_OUTPUT" -type f -name "*.wav" | wc -l

echo "Structured output wav:"
find "$OUTPUT_ROOT" -type f -name "*.wav" | wc -l
