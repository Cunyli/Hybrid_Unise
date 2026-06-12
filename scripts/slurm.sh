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
        clean_stem = Path(row["clean_filepath"]).stem
        candidates = [
            out_root / "wav" / f"{uid}.wav",
            out_root / "wav" / f"{clean_stem}.wav",
        ]
        enhanced = next((path for path in candidates if path.is_file()), candidates[0])
        if not enhanced.is_file():
            raise FileNotFoundError(f"Missing enhanced wav for {uid}: checked {candidates}")
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

  eval_token_validation)
    CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/conf/tau_fixed_aug_same_clean_diag_fast2gpu_v1_unise.yaml}"
    CKPT_PATH="${CKPT_PATH:-$(best_checkpoint "$ROOT_DIR/checkpoints/tau_fixed_aug_same_clean_diag_fast2gpu_v1")}"
    PAIR_MANIFEST="${PAIR_MANIFEST:?PAIR_MANIFEST must be set}"
    OUTPUT_JSON="${OUTPUT_JSON:-$ROOT_DIR/outputs/token_validation/${SLURM_JOB_ID}.json}"
    SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    DEVICES="${DEVICES:-0}"

    test -f "$CKPT_PATH"
    test -f "$PAIR_MANIFEST"
    mkdir -p "$(dirname "$OUTPUT_JSON")"
    echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
    echo "Checkpoint: $CKPT_PATH" | tee -a "$LIVE_LOG"
    echo "Pair manifest: $PAIR_MANIFEST" | tee -a "$LIVE_LOG"
    echo "Output JSON: $OUTPUT_JSON" | tee -a "$LIVE_LOG"
    echo "Batch size: $BATCH_SIZE" | tee -a "$LIVE_LOG"
    if [[ -n "$SAMPLES_PER_EPOCH" ]]; then
      python scripts/eval_token_validation.py \
        --config "$CONFIG_PATH" \
        --ckpt-path "$CKPT_PATH" \
        --pair-manifest "$PAIR_MANIFEST" \
        --samples-per-epoch "$SAMPLES_PER_EPOCH" \
        --batch-size "$BATCH_SIZE" \
        --devices "$DEVICES" \
        --output-json "$OUTPUT_JSON" 2>&1 | tee -a "$LIVE_LOG"
    else
      python scripts/eval_token_validation.py \
        --config "$CONFIG_PATH" \
        --ckpt-path "$CKPT_PATH" \
        --pair-manifest "$PAIR_MANIFEST" \
        --batch-size "$BATCH_SIZE" \
        --devices "$DEVICES" \
        --output-json "$OUTPUT_JSON" 2>&1 | tee -a "$LIVE_LOG"
    fi
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

  rolling_cache_smoke)
    DRY_DATA_ROOT="${DRY_DATA_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/dry_data_lake}"
    USE_SIM_ROOT="${USE_SIM_ROOT:-/scratch/work/lil14/USE_simulation}"
    SIM_CONFIG="${SIM_CONFIG:-$ROOT_DIR/conf/use_simulation_phone_room_16k.yaml}"
    CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/tmp/rolling_cache_smoke}"
    RUN_ID="${RUN_ID:-rolling_cache_smoke_${SLURM_JOB_ID}}"
    CACHE_SIZE_GB="${CACHE_SIZE_GB:-0.000001}"
    SHARD_SIZE_MB="${SHARD_SIZE_MB:-1}"
    CUT_DURATION="${CUT_DURATION:-0.25}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_WORKERS="${NUM_WORKERS:-1}"
    INCLUDE_ARCHIVES="${INCLUDE_ARCHIVES:-1}"
    MAX_ARCHIVES_TO_INDEX="${MAX_ARCHIVES_TO_INDEX:-16}"
    MAX_ARCHIVE_MEMBERS_PER_ARCHIVE="${MAX_ARCHIVE_MEMBERS_PER_ARCHIVE:-8}"
    INCLUDE_CLEAN_ARCHIVES="${INCLUDE_CLEAN_ARCHIVES:-}"
    INCLUDE_NOISE_ARCHIVES="${INCLUDE_NOISE_ARCHIVES:-0}"
    INCLUDE_RIR_ARCHIVES="${INCLUDE_RIR_ARCHIVES:-}"
    INCLUDE_WIND_ARCHIVES="${INCLUDE_WIND_ARCHIVES:-0}"
    CLEAN_STATUSES="${CLEAN_STATUSES:-}"

    echo "Dry data root: $DRY_DATA_ROOT" | tee -a "$LIVE_LOG"
    echo "USE simulation root: $USE_SIM_ROOT" | tee -a "$LIVE_LOG"
    echo "Simulation config: $SIM_CONFIG" | tee -a "$LIVE_LOG"
    echo "Cache dir: $CACHE_DIR" | tee -a "$LIVE_LOG"
    echo "Include archives: $INCLUDE_ARCHIVES" | tee -a "$LIVE_LOG"
    echo "Max archives to index: $MAX_ARCHIVES_TO_INDEX" | tee -a "$LIVE_LOG"
    echo "Max members per archive: $MAX_ARCHIVE_MEMBERS_PER_ARCHIVE" | tee -a "$LIVE_LOG"
    if [[ -n "$CLEAN_STATUSES" ]]; then
      echo "Clean statuses: $CLEAN_STATUSES" | tee -a "$LIVE_LOG"
    fi
    python - "$DRY_DATA_ROOT" "$USE_SIM_ROOT" "$SIM_CONFIG" "$CACHE_DIR" "$RUN_ID" "$CACHE_SIZE_GB" "$SHARD_SIZE_MB" "$CUT_DURATION" "$BATCH_SIZE" "$NUM_WORKERS" "$INCLUDE_ARCHIVES" "$MAX_ARCHIVES_TO_INDEX" "$MAX_ARCHIVE_MEMBERS_PER_ARCHIVE" "$CLEAN_STATUSES" "$INCLUDE_CLEAN_ARCHIVES" "$INCLUDE_NOISE_ARCHIVES" "$INCLUDE_RIR_ARCHIVES" "$INCLUDE_WIND_ARCHIVES" <<'PY' 2>&1 | tee -a "$LIVE_LOG"
import sys
from pathlib import Path

from dataloader.rolling_cache import UseSimulationRollingCacheDataLoadIter

(
    dry_data_root,
    use_sim_root,
    sim_config,
    cache_dir,
    run_id,
    cache_size_gb,
    shard_size_mb,
    cut_duration,
    batch_size,
    num_workers,
    include_archives,
    max_archives_to_index,
    max_archive_members_per_archive,
    clean_statuses,
    include_clean_archives,
    include_noise_archives,
    include_rir_archives,
    include_wind_archives,
) = sys.argv[1:]

clean_statuses = [x for x in clean_statuses.split(",") if x] or None
def optional_bool(value):
    if value == "":
        return None
    return value not in {"0", "false", "False", "no", "NO"}

dataset = UseSimulationRollingCacheDataLoadIter(
    dry_data_root=dry_data_root,
    use_simulation_root=use_sim_root,
    simulation_config=sim_config,
    cache_dir=cache_dir,
    run_id=run_id,
    cache_size_gb=float(cache_size_gb),
    shard_size_mb=int(shard_size_mb),
    cleanup_policy="refresh",
    include_archives=include_archives not in {"0", "false", "False", "no", "NO"},
    max_archives_to_index=int(max_archives_to_index),
    max_archive_members_per_archive=int(max_archive_members_per_archive),
    clean_statuses=clean_statuses,
    include_clean_archives=optional_bool(include_clean_archives),
    include_noise_archives=optional_bool(include_noise_archives) is True,
    include_rir_archives=optional_bool(include_rir_archives),
    include_wind_archives=optional_bool(include_wind_archives) is True,
    batch_size=int(batch_size),
    cut_duration=[float(cut_duration), float(cut_duration)],
    num_workers=int(num_workers),
    samples_per_epoch=max(2, int(batch_size)),
    mode="train",
    seed=20260605,
)
batch = next(iter(dataset))
stats_path = Path(dataset.stats_path)
print("mode", batch[0])
print("mix_shape", tuple(batch[2].shape))
print("speech_shape", tuple(batch[3].shape))
print("fs", batch[5].tolist())
print("lengths", batch[6].tolist())
print("names", batch[7])
print("cache", dataset.cache_run_dir)
print("stats", stats_path)
print("noise_items", len(dataset.noise_paths))
print("rir_items", len(dataset.rir_paths))
print("clean_stats", dataset.stats_path.read_text())
print("noise_archive_items", dataset.noise_stats.get("selected_archive_audio"))
print("rir_archive_items", dataset.rir_stats.get("selected_archive_audio"))
PY
    ;;

  *)
    echo "Unknown TASK: $TASK" | tee -a "$LIVE_LOG"
    exit 2
    ;;
esac

echo "Completed: $(date)" | tee -a "$LIVE_LOG"
