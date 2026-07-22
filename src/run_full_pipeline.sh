#!/usr/bin/env bash
set -e

LABEL=""
STEPS=()

# Hyperparameter defaults. This is the single source of truth for defaults —
# they mirror the last completed full GRPO run (data/full_grpo/, finished
# 2026-07-07). Override any of them with --flag=value on the command line.
RANDOM_SEED=42
VALIDATION_SIZE=0.02
TEST_SIZE=0.05
MAX_ENTITIES=6
TARGET_MODULES=all_linear
MAX_NEGATIVES=""

SFT_LEARNING_RATE=2e-4
SFT_LORA_RANK=16
SFT_BATCH_SIZE=16
SFT_GRADIENT_ACCUMULATION_STEPS=4
SFT_NUM_EPOCHS=3
SFT_MAX_TRAIN_SAMPLES=""
SFT_MAX_VALIDATION_SAMPLES=""

GRPO_LEARNING_RATE=5e-7
GRPO_LORA_RANK=16
GRPO_BATCH_SIZE=24
GRPO_GRADIENT_ACCUMULATION_STEPS=8
GRPO_NUM_GENERATIONS=8
GRPO_MAX_COMPLETION_LENGTH=128
GRPO_NUM_EPOCHS=1
GRPO_MAX_TRAIN_SAMPLES=6000
GRPO_MAX_VALIDATION_SAMPLES=100
GRPO_BETA=0.0
GRPO_TEMPERATURE=1.0

TEST_BATCH_SIZE=16
MAX_TEST_SAMPLES=""

for arg in "$@"; do
  case $arg in
    --label=*) LABEL="${arg#*=}" ;;
    -[0-9]*) STEPS+=("${arg#-}") ;;

    --random_seed=*) RANDOM_SEED="${arg#*=}" ;;
    --validation_size=*) VALIDATION_SIZE="${arg#*=}" ;;
    --test_size=*) TEST_SIZE="${arg#*=}" ;;
    --max_entities=*) MAX_ENTITIES="${arg#*=}" ;;
    --target_modules=*) TARGET_MODULES="${arg#*=}" ;;
    --max_negatives=*) MAX_NEGATIVES="${arg#*=}" ;;

    --sft_learning_rate=*) SFT_LEARNING_RATE="${arg#*=}" ;;
    --sft_lora_rank=*) SFT_LORA_RANK="${arg#*=}" ;;
    --sft_batch_size=*) SFT_BATCH_SIZE="${arg#*=}" ;;
    --sft_gradient_accumulation_steps=*) SFT_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --sft_num_epochs=*) SFT_NUM_EPOCHS="${arg#*=}" ;;
    --sft_max_train_samples=*) SFT_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --sft_max_validation_samples=*) SFT_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;

    --grpo_learning_rate=*) GRPO_LEARNING_RATE="${arg#*=}" ;;
    --grpo_lora_rank=*) GRPO_LORA_RANK="${arg#*=}" ;;
    --grpo_batch_size=*) GRPO_BATCH_SIZE="${arg#*=}" ;;
    --grpo_gradient_accumulation_steps=*) GRPO_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --grpo_num_generations=*) GRPO_NUM_GENERATIONS="${arg#*=}" ;;
    --grpo_max_completion_length=*) GRPO_MAX_COMPLETION_LENGTH="${arg#*=}" ;;
    --grpo_num_epochs=*) GRPO_NUM_EPOCHS="${arg#*=}" ;;
    --grpo_max_train_samples=*) GRPO_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --grpo_max_validation_samples=*) GRPO_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;
    --grpo_beta=*) GRPO_BETA="${arg#*=}" ;;
    --grpo_temperature=*) GRPO_TEMPERATURE="${arg#*=}" ;;

    --test_batch_size=*) TEST_BATCH_SIZE="${arg#*=}" ;;
    --max_test_samples=*) MAX_TEST_SAMPLES="${arg#*=}" ;;

    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 --label=RUN_LABEL [-1] [-2] [-3] [-4] [-5] [--random_seed=42] [--sft_learning_rate=2e-4] [--grpo_beta=0.0] ..."
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
CHECKPOINT_FOLDER_GRPO="${LABEL}_grpo"

# --flag value pairs for hyperparameters that may legitimately be left unset
# ("no limit"). Omit the flag entirely rather than passing an empty value, so
# the python side's `type=int` doesn't choke on "".
MAX_NEGATIVES_ARGS=()
[[ -n "$MAX_NEGATIVES" ]] && MAX_NEGATIVES_ARGS=(--max_negatives "$MAX_NEGATIVES")
SFT_MAX_TRAIN_SAMPLES_ARGS=()
[[ -n "$SFT_MAX_TRAIN_SAMPLES" ]] && SFT_MAX_TRAIN_SAMPLES_ARGS=(--sft_max_train_samples "$SFT_MAX_TRAIN_SAMPLES")
SFT_MAX_VALIDATION_SAMPLES_ARGS=()
[[ -n "$SFT_MAX_VALIDATION_SAMPLES" ]] && SFT_MAX_VALIDATION_SAMPLES_ARGS=(--sft_max_validation_samples "$SFT_MAX_VALIDATION_SAMPLES")
GRPO_MAX_TRAIN_SAMPLES_ARGS=()
[[ -n "$GRPO_MAX_TRAIN_SAMPLES" ]] && GRPO_MAX_TRAIN_SAMPLES_ARGS=(--grpo_max_train_samples "$GRPO_MAX_TRAIN_SAMPLES")
GRPO_MAX_VALIDATION_SAMPLES_ARGS=()
[[ -n "$GRPO_MAX_VALIDATION_SAMPLES" ]] && GRPO_MAX_VALIDATION_SAMPLES_ARGS=(--grpo_max_validation_samples "$GRPO_MAX_VALIDATION_SAMPLES")
MAX_TEST_SAMPLES_ARGS=()
[[ -n "$MAX_TEST_SAMPLES" ]] && MAX_TEST_SAMPLES_ARGS=(--max_test_samples "$MAX_TEST_SAMPLES")

if run_step 1; then
  echo "[STEP 1] TEST BASELINE MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/baseline_metrics.json" \
    --mode baseline \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --max_entities "$MAX_ENTITIES" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --grpo_max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
    "${MAX_TEST_SAMPLES_ARGS[@]}"
fi

if run_step 2; then
  echo "[STEP 2] SFT TRAINING"
  python src/train_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/sft_out" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_SFT" \
    --mode "sft" \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --max_entities "$MAX_ENTITIES" \
    --target_modules "$TARGET_MODULES" \
    --sft_learning_rate "$SFT_LEARNING_RATE" \
    --sft_lora_rank "$SFT_LORA_RANK" \
    --sft_batch_size "$SFT_BATCH_SIZE" \
    --sft_gradient_accumulation_steps "$SFT_GRADIENT_ACCUMULATION_STEPS" \
    --sft_num_epochs "$SFT_NUM_EPOCHS" \
    --grpo_learning_rate "$GRPO_LEARNING_RATE" \
    --grpo_lora_rank "$GRPO_LORA_RANK" \
    --grpo_batch_size "$GRPO_BATCH_SIZE" \
    --grpo_gradient_accumulation_steps "$GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --grpo_num_generations "$GRPO_NUM_GENERATIONS" \
    --grpo_max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
    --grpo_num_epochs "$GRPO_NUM_EPOCHS" \
    --grpo_beta "$GRPO_BETA" \
    --grpo_temperature "$GRPO_TEMPERATURE" \
    "${MAX_NEGATIVES_ARGS[@]}" \
    "${SFT_MAX_TRAIN_SAMPLES_ARGS[@]}" \
    "${SFT_MAX_VALIDATION_SAMPLES_ARGS[@]}" \
    "${GRPO_MAX_TRAIN_SAMPLES_ARGS[@]}" \
    "${GRPO_MAX_VALIDATION_SAMPLES_ARGS[@]}"
fi

if run_step 3; then
  echo "[STEP 3] TEST SFT MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/sft_metrics.json" \
    --mode sft \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --max_entities "$MAX_ENTITIES" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --grpo_max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
    "${MAX_TEST_SAMPLES_ARGS[@]}"
fi

if run_step 4; then
  echo "[STEP 4] GRPO TRAINING"
  python src/train_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/grpo_out" \
    --checkpoint_load_folder "$CHECKPOINT_FOLDER_SFT" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_GRPO" \
    --mode "grpo" \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --max_entities "$MAX_ENTITIES" \
    --target_modules "$TARGET_MODULES" \
    --sft_learning_rate "$SFT_LEARNING_RATE" \
    --sft_lora_rank "$SFT_LORA_RANK" \
    --sft_batch_size "$SFT_BATCH_SIZE" \
    --sft_gradient_accumulation_steps "$SFT_GRADIENT_ACCUMULATION_STEPS" \
    --sft_num_epochs "$SFT_NUM_EPOCHS" \
    --grpo_learning_rate "$GRPO_LEARNING_RATE" \
    --grpo_lora_rank "$GRPO_LORA_RANK" \
    --grpo_batch_size "$GRPO_BATCH_SIZE" \
    --grpo_gradient_accumulation_steps "$GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --grpo_num_generations "$GRPO_NUM_GENERATIONS" \
    --grpo_max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
    --grpo_num_epochs "$GRPO_NUM_EPOCHS" \
    --grpo_beta "$GRPO_BETA" \
    --grpo_temperature "$GRPO_TEMPERATURE" \
    "${MAX_NEGATIVES_ARGS[@]}" \
    "${SFT_MAX_TRAIN_SAMPLES_ARGS[@]}" \
    "${SFT_MAX_VALIDATION_SAMPLES_ARGS[@]}" \
    "${GRPO_MAX_TRAIN_SAMPLES_ARGS[@]}" \
    "${GRPO_MAX_VALIDATION_SAMPLES_ARGS[@]}"
fi

if run_step 5; then
  echo "[STEP 5] TEST GRPO MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/grpo_metrics.json" \
    --mode grpo \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --grpo_checkpoint_folder "$CHECKPOINT_FOLDER_GRPO" \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --max_entities "$MAX_ENTITIES" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --grpo_max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
    "${MAX_TEST_SAMPLES_ARGS[@]}"
fi

echo "[DONE]"
