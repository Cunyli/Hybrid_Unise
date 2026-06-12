#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/unified-audio/QuarkAudio-UniSE}"
AVQI_ROOT="${AVQI_ROOT:-/scratch/work/lil14/avqi}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-avqi}"
AVQI_PYTHON="${AVQI_PYTHON:-python}"
PAIR_CSV="${PAIR_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}"
ENHANCED_ROOT="${ENHANCED_ROOT:-/scratch/work/lil14/data/TAU/enhanced/unise/avqi_ablation_6models}"
RESULTS_DIR="${RESULTS_DIR:-$AVQI_ROOT/avqi_output/unise_tau_ablation_6models}"
DB_NAME="${DB_NAME:-TAU_fixed}"
SPEAKING_TYPE="${SPEAKING_TYPE:-both}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "$ROOT_DIR"

set +u
source ~/.bashrc
conda activate "$CONDA_ENV_NAME"
set -u

"$AVQI_PYTHON" scripts/score_unise_avqi_comparison.py \
  --pair-csv "$PAIR_CSV" \
  --results-dir "$RESULTS_DIR" \
  --avqi-root "$AVQI_ROOT" \
  --db-name "$DB_NAME" \
  --speaking-type "$SPEAKING_TYPE" \
  --condition "full_semw025=$ENHANCED_ROOT/full_semw025" \
  --condition "adapter_semw025=$ENHANCED_ROOT/adapter_semw025" \
  --condition "lora_qv=$ENHANCED_ROOT/lora_qv" \
  --condition "lora_attn_mlp=$ENHANCED_ROOT/lora_attn_mlp" \
  --condition "cs_only_all=$ENHANCED_ROOT/cs_only" \
  --condition "sv_only_all=$ENHANCED_ROOT/sv_only" \
  --task-ensemble "task_ensemble=$ENHANCED_ROOT/cs_only,$ENHANCED_ROOT/sv_only" \
  --write-samples

cd "$AVQI_ROOT"

"$AVQI_PYTHON" visualization.py \
  --data-dir "$RESULTS_DIR" \
  --db-name "$DB_NAME" \
  --variants "full_semw025,adapter_semw025,lora_qv,lora_attn_mlp,cs_only_all,sv_only_all,task_ensemble" \
  --speaking "$SPEAKING_TYPE"

echo "Summary: $RESULTS_DIR/tau_fixed_avqi_summary.csv"
echo "Plots: $RESULTS_DIR/plots/$DB_NAME/summary"
echo "Samples: $RESULTS_DIR/samples"
