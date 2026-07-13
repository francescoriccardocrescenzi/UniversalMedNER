#!/usr/bin/env bash
set -e

LABEL=""
for arg in "$@"; do
  case $arg in
    --label=*)
      LABEL="${arg#*=}"
      ;;
    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 --label=RUN_LABEL"
  exit 1
fi

# Hardcoded sampling grid - edit these to change what gets swept.
# Every combination (cross product) of TEMPERATURES x TOP_PS is tested, plus a
# dedicated greedy-decoding run (no --temperature/--top_p) as a baseline.
TEMPERATURES=(0.7 1.0 1.3)
TOP_PS=(0.9 0.95 1.0)

export HF_HOME=data/.cache/huggingface
set -a; source .env; set +a
source .venv/bin/activate

RUN_FOLDER="data/$LABEL"
CHECKPOINT_FOLDER_SFT="${LABEL}_sft"
CHECKPOINT_FOLDER_GRPO="${LABEL}_grpo"
OUT_DIR="$RUN_FOLDER/sampling_sweep"
mkdir -p "$OUT_DIR"

echo "[INFO] Sweeping temperature x top_p for label: $LABEL"

echo "[STEP] Testing grpo checkpoint with greedy decoding"
python src/test_pipeline.py \
  --hyperparam_path "$RUN_FOLDER/hyperparam.json" \
  --metrics_path "$OUT_DIR/grpo_metrics_greedy.json" \
  --completions_path "$OUT_DIR/grpo_completions_greedy.parquet" \
  --mode grpo \
  --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
  --grpo_checkpoint_folder "$CHECKPOINT_FOLDER_GRPO" \
  --verbose

for T in "${TEMPERATURES[@]}"; do
  for P in "${TOP_PS[@]}"; do
    TAG="temp${T}_topp${P}"
    echo "[STEP] Testing grpo checkpoint with temperature=$T, top_p=$P"
    python src/test_pipeline.py \
      --hyperparam_path "$RUN_FOLDER/hyperparam.json" \
      --metrics_path "$OUT_DIR/grpo_metrics_${TAG}.json" \
      --completions_path "$OUT_DIR/grpo_completions_${TAG}.parquet" \
      --mode grpo \
      --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
      --grpo_checkpoint_folder "$CHECKPOINT_FOLDER_GRPO" \
      --temperature "$T" \
      --top_p "$P" \
      --verbose
  done
done

echo "[STEP] Summarizing sweep metrics into a single table"
python src/summarize_sampling_sweep.py \
  --metrics_dir "$OUT_DIR" \
  --pattern "grpo_metrics_*.json" \
  --out_path "$OUT_DIR/sweep_summary.md"

echo "[DONE]"
