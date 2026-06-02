#!/usr/bin/env bash
set -e

LABEL=""
STEPS=()

for arg in "$@"; do
  case $arg in
    --label=*)
      LABEL="${arg#*=}"
      ;;
    -[0-9]*)
      STEPS+=("${arg#-}")
      ;;
    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 --label=RUN_LABEL [-1] [-2] ..."
  exit 1
fi

run_step() {
  local step=$1

  if [[ ${#STEPS[@]} -ne 0 ]]; then
    for s in "${STEPS[@]}"; do
      [[ "$s" == "$step" ]] && return 0
    done
    return 1
  fi
  return 0
}

echo "[INFO] Running pipeline for label: $LABEL"

export HF_HOME=data/.cache/huggingface
set -a; source .env; set +a
source .venv/bin/activate

RUN_FOLDER="data/$LABEL"
CHECKPOINT_FOLDER_SFT="${LABEL}_sft"

if run_step 1; then
  echo "[STEP 1] TEST BASELINE MODEL"
  python src/test_pipeline.py \
    --hyperparam_path "$RUN_FOLDER/hyperparam.json" \
    --metrics_path "$RUN_FOLDER/baseline_metrics.json" \
    --verbose
fi

if run_step 2; then
  echo "[STEP 2] SFT TRAINING"
  python src/train_pipeline.py \
    --label "$LABEL" \
    --hyperparam_path "$RUN_FOLDER/hyperparam.json" \
    --save_folder "$RUN_FOLDER/sft_out" \
    --checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --mode "sft"
fi

if run_step 3; then
  echo "[STEP 3] TEST SFT MODEL"
  python src/test_pipeline.py \
    --checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --hyperparam_path "$RUN_FOLDER/hyperparam.json" \
    --metrics_path "$RUN_FOLDER/sft_metrics.json" \
    --verbose
fi

if run_step 4; then
  echo "[STEP 4] GRPO TRAINING"
  python step4_push.py --label "$LABEL"
fi

if run_step 5; then
  echo "[STEP 5] TEST GRPO MODEL"
  python step5_eval.py --label "$LABEL"
fi

echo "[DONE]"