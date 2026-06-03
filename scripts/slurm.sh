#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/unified-audio/QuarkAudio-UniSE}"
TASK="${TASK:-${1:-train}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$ROOT_DIR/logs/slurm}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-unise}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  PARTITION="${PARTITION:-gpu-a100-80g}"
  GPU_TYPE="${GPU_TYPE:-a100}"
  GPUS="${GPUS:-1}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
  MEMORY="${MEMORY:-64G}"
  TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
  JOB_NAME="${JOB_NAME:-unise-${TASK}}"

  mkdir -p "$RUN_LOG_DIR"
  echo "Submitting $TASK to $PARTITION"
  sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --gres="gpu:${GPU_TYPE}:${GPUS}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$RUN_LOG_DIR/slurm_%j.out" \
    --error="$RUN_LOG_DIR/slurm_%j.err" \
    --export=ALL,ROOT_DIR="$ROOT_DIR",TASK="$TASK",RUN_LOG_DIR="$RUN_LOG_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME" \
    "$0"
  exit 0
fi

mkdir -p "$RUN_LOG_DIR"
cd "$ROOT_DIR"

LIVE_LOG="$RUN_LOG_DIR/${TASK}_${SLURM_JOB_ID}.log"
echo "Task: $TASK" | tee -a "$LIVE_LOG"
echo "Job: $SLURM_JOB_ID" | tee -a "$LIVE_LOG"
echo "Started: $(date)" | tee -a "$LIVE_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LIVE_LOG"

module load triton/2025.1-gcc 2>/dev/null || true
module load gcc/13.3.0 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

best_checkpoint() {
  python - "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

for pattern in ("**/best_guarded.ckpt", "**/best_avqi_gap.ckpt"):
    candidates = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if candidates:
        print(candidates[-1])
        raise SystemExit(0)

ckpts = sorted(root.glob("**/latest_*.ckpt"), key=lambda path: path.stat().st_mtime)
if not ckpts:
    ckpts = sorted(root.glob("**/*.ckpt"), key=lambda path: path.stat().st_mtime)
if ckpts:
    print(ckpts[-1])
PY
}

case "$TASK" in
  train)
    CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/conf/tau_fixed_unise.yaml}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
    python train.py --config "$CONFIG_PATH" 2>&1 | tee -a "$LIVE_LOG"
    ;;

  infer)
    CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/conf/tau_fixed_unise.yaml}"
    OUTPUT_DIR="${OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/unise/phone_room/test}"
    CKPT_ROOT="${CKPT_ROOT:-$ROOT_DIR/checkpoints/tau_fixed}"
    CKPT_PATH="${CKPT_PATH:-$(best_checkpoint "$CKPT_ROOT")}"
    PAIR_CSV="${PAIR_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}"
    WAV_DIR="$OUTPUT_DIR/wav"

    test -f "$CKPT_PATH"
    mkdir -p "$WAV_DIR"
    echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
    echo "Checkpoint: $CKPT_PATH" | tee -a "$LIVE_LOG"
    python test.py --config "$CONFIG_PATH" --save_enhanced "$WAV_DIR" --ckpt_path "$CKPT_PATH" 2>&1 | tee -a "$LIVE_LOG"

    python - "$PAIR_CSV" "$OUTPUT_DIR" <<'PY' 2>&1 | tee -a "$LIVE_LOG"
import csv
import sys
from pathlib import Path

pair_csv = Path(sys.argv[1])
out_root = Path(sys.argv[2])
rows = list(csv.DictReader(pair_csv.open()))

with (out_root / "inf.scp").open("w") as inf, (out_root / "ref.scp").open("w") as ref:
    for row in rows:
        uid = row["uid"]
        enhanced = out_root / "wav" / f"{uid}.wav"
        if not enhanced.is_file():
            raise FileNotFoundError(f"Missing enhanced wav for {uid}: {enhanced}")
        inf.write(f"{uid} {enhanced}\n")
        ref.write(f"{uid} {row['clean_filepath']}\n")

print(f"Wrote inf.scp/ref.scp for {len(rows)} utterances")
PY
    ;;

  tokenizer_oracle)
    CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/conf/tau_fixed_unise.yaml}"
    SPLIT="${SPLIT:-test}"
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/tokenizer_oracle/tau_fixed/$SPLIT}"
    MAX_BATCHES="${MAX_BATCHES:-0}"
    SAVE_EXAMPLES="${SAVE_EXAMPLES:-5}"

    mkdir -p "$OUTPUT_DIR"
    echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
    echo "Split: $SPLIT" | tee -a "$LIVE_LOG"
    echo "Output: $OUTPUT_DIR" | tee -a "$LIVE_LOG"
    python scripts/eval_tokenizer_oracle.py \
      --config "$CONFIG_PATH" \
      --split "$SPLIT" \
      --output_dir "$OUTPUT_DIR" \
      --max_batches "$MAX_BATCHES" \
      --save_examples "$SAVE_EXAMPLES" 2>&1 | tee -a "$LIVE_LOG"
    ;;

  token_similarity)
    CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/conf/tau_fixed_unise.yaml}"
    SPLIT="${SPLIT:-val}"
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/token_similarity/tau_fixed/$SPLIT}"
    MAX_BATCHES="${MAX_BATCHES:-0}"

    mkdir -p "$OUTPUT_DIR"
    echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
    echo "Split: $SPLIT" | tee -a "$LIVE_LOG"
    echo "Output: $OUTPUT_DIR" | tee -a "$LIVE_LOG"
    python scripts/eval_token_similarity.py \
      --config "$CONFIG_PATH" \
      --split "$SPLIT" \
      --output_dir "$OUTPUT_DIR" \
      --max_batches "$MAX_BATCHES" 2>&1 | tee -a "$LIVE_LOG"
    ;;

  eval_generation)
    CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/conf/tau_fixed_unise.yaml}"
    SPLIT="${SPLIT:-val}"
    CKPT_PATH="${CKPT_PATH:-$(best_checkpoint "$ROOT_DIR/checkpoints/tau_fixed")}"
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/generation_eval/tau_fixed/$SPLIT}"
    MAX_BATCHES="${MAX_BATCHES:-0}"
    SAVE_EXAMPLES="${SAVE_EXAMPLES:-5}"

    test -f "$CKPT_PATH"
    mkdir -p "$OUTPUT_DIR"
    echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
    echo "Checkpoint: $CKPT_PATH" | tee -a "$LIVE_LOG"
    echo "Split: $SPLIT" | tee -a "$LIVE_LOG"
    echo "Output: $OUTPUT_DIR" | tee -a "$LIVE_LOG"
    python scripts/eval_model_generation.py \
      --config "$CONFIG_PATH" \
      --checkpoint "$CKPT_PATH" \
      --split "$SPLIT" \
      --output_dir "$OUTPUT_DIR" \
      --max_batches "$MAX_BATCHES" \
      --save_examples "$SAVE_EXAMPLES" 2>&1 | tee -a "$LIVE_LOG"
    ;;

  smoke_sr)
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/sr_smoke}"
    RUN_CONFIG="${RUN_CONFIG:-$ROOT_DIR/conf/generated/config_sr_smoke.yaml}"
    MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/pretrained/Spark-TTS-0.5B}"
    BASE_CKPT="${BASE_CKPT:-$ROOT_DIR/checkpoints/epoch=20-step=109367.ckpt}"

    mkdir -p "$OUTPUT_DIR"
    test -f "$BASE_CKPT"
    test -f "$MODEL_DIR/config.yaml"
    test -f "$MODEL_DIR/BiCodec/config.yaml"
    test -f "$MODEL_DIR/BiCodec/model.safetensors"
    test -f "$MODEL_DIR/wav2vec2-large-xlsr-53/config.json"
    test -f "$MODEL_DIR/wav2vec2-large-xlsr-53/preprocessor_config.json"
    test -f "$MODEL_DIR/wav2vec2-large-xlsr-53/pytorch_model.bin"

    python - "$ROOT_DIR" "$RUN_CONFIG" "$BASE_CKPT" "$MODEL_DIR" <<'PY'
import sys
import yaml
from pathlib import Path

root, run_config, base_ckpt, model_dir = map(Path, sys.argv[1:])
config = yaml.safe_load((root / "conf/config.yaml").read_text())
config["accelerator"] = "gpu"
config["devices"] = [0]
config["ckpt_path"] = str(base_ckpt)
config["codec_ckpt_dir"] = str(model_dir)
config["dataset_config"]["test_kwargs"] = {
    "batch_size": 1,
    "num_workers": 1,
    "prefetch": 1,
    "mode": "se",
    "data_enroll_dir": None,
    "enroll_duration": 5.0,
    "data_src_dir": "./AudioSamples/SR/noisy",
    "data_tgt_dir": "./AudioSamples/SR/clean",
}
Path(run_config).write_text(yaml.safe_dump(config, sort_keys=False))
print("Wrote", run_config)
PY
    python test.py --config "$RUN_CONFIG" --save_enhanced "$OUTPUT_DIR" 2>&1 | tee -a "$LIVE_LOG"
    ;;

  *)
    echo "Unknown TASK: $TASK" | tee -a "$LIVE_LOG"
    exit 2
    ;;
esac

echo "Completed: $(date)" | tee -a "$LIVE_LOG"
