#!/usr/bin/env bash

# Fail if any command fails
set -e

# **************************** ARGUMENTS *************************************

# ------ Default values -------

LABEL=""
STEPS=()
PROFILE=standard

MODEL_REPO=google/medgemma-1.5-4b-it

RANDOM_SEED=42
VALIDATION_SIZE=0.02
TEST_SIZE=0.05
TARGET_MODULES=all_linear
MAX_RAW_SAMPLES=-1

SFNER_SFT_LEARNING_RATE=2e-4
SFNER_SFT_LORA_RANK=16
SFNER_SFT_BATCH_SIZE=16
SFNER_SFT_GRADIENT_ACCUMULATION_STEPS=4
SFNER_SFT_NUM_EPOCHS=3
SFNER_SFT_MAX_TRAIN_SAMPLES=-1
SFNER_SFT_MAX_VALIDATION_SAMPLES=-1
SFNER_SFT_SAVE_STEPS=200
SFNER_SFT_EVAL_STEPS=200
SFNER_SFT_MAX_STEPS=-1

SFNER_GRPO_LEARNING_RATE=5e-7
SFNER_GRPO_LORA_RANK=16
SFNER_GRPO_BATCH_SIZE=24
SFNER_GRPO_GRADIENT_ACCUMULATION_STEPS=8
SFNER_GRPO_NUM_GENERATIONS=8
SFNER_GRPO_MAX_COMPLETION_LENGTH=128
SFNER_GRPO_NUM_EPOCHS=1
SFNER_GRPO_MAX_TRAIN_SAMPLES=6000
SFNER_GRPO_MAX_VALIDATION_SAMPLES=100
SFNER_GRPO_BETA=0.0
SFNER_GRPO_TEMPERATURE=1.0
SFNER_GRPO_EVAL_STEPS=50
SFNER_GRPO_MAX_STEPS=-1

SFNER_TEST_BATCH_SIZE=16
SFNER_MAX_TEST_SAMPLES=-1

# ----- Passed values -----

for arg in "$@"; do
  case $arg in
    --label=*) LABEL="${arg#*=}" ;;
    -[0-9]*) STEPS+=("${arg#-}") ;;
    --profile=*) PROFILE="${arg#*=}" ;;

    --model_repo=*) MODEL_REPO="${arg#*=}" ;;

    --random_seed=*) RANDOM_SEED="${arg#*=}" ;;
    --validation_size=*) VALIDATION_SIZE="${arg#*=}" ;;
    --test_size=*) TEST_SIZE="${arg#*=}" ;;
    --target_modules=*) TARGET_MODULES="${arg#*=}" ;;
    --max_raw_samples=*) MAX_RAW_SAMPLES="${arg#*=}" ;;

    --sfner_sft_learning_rate=*) SFNER_SFT_LEARNING_RATE="${arg#*=}" ;;
    --sfner_sft_lora_rank=*) SFNER_SFT_LORA_RANK="${arg#*=}" ;;
    --sfner_sft_batch_size=*) SFNER_SFT_BATCH_SIZE="${arg#*=}" ;;
    --sfner_sft_gradient_accumulation_steps=*) SFNER_SFT_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --sfner_sft_num_epochs=*) SFNER_SFT_NUM_EPOCHS="${arg#*=}" ;;
    --sfner_sft_max_train_samples=*) SFNER_SFT_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --sfner_sft_max_validation_samples=*) SFNER_SFT_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;
    --sfner_sft_save_steps=*) SFNER_SFT_SAVE_STEPS="${arg#*=}" ;;
    --sfner_sft_eval_steps=*) SFNER_SFT_EVAL_STEPS="${arg#*=}" ;;
    --sfner_sft_max_steps=*) SFNER_SFT_MAX_STEPS="${arg#*=}" ;;

    --sfner_grpo_learning_rate=*) SFNER_GRPO_LEARNING_RATE="${arg#*=}" ;;
    --sfner_grpo_lora_rank=*) SFNER_GRPO_LORA_RANK="${arg#*=}" ;;
    --sfner_grpo_batch_size=*) SFNER_GRPO_BATCH_SIZE="${arg#*=}" ;;
    --sfner_grpo_gradient_accumulation_steps=*) SFNER_GRPO_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --sfner_grpo_num_generations=*) SFNER_GRPO_NUM_GENERATIONS="${arg#*=}" ;;
    --sfner_grpo_max_completion_length=*) SFNER_GRPO_MAX_COMPLETION_LENGTH="${arg#*=}" ;;
    --sfner_grpo_num_epochs=*) SFNER_GRPO_NUM_EPOCHS="${arg#*=}" ;;
    --sfner_grpo_max_train_samples=*) SFNER_GRPO_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --sfner_grpo_max_validation_samples=*) SFNER_GRPO_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;
    --sfner_grpo_beta=*) SFNER_GRPO_BETA="${arg#*=}" ;;
    --sfner_grpo_temperature=*) SFNER_GRPO_TEMPERATURE="${arg#*=}" ;;
    --sfner_grpo_eval_steps=*) SFNER_GRPO_EVAL_STEPS="${arg#*=}" ;;
    --sfner_grpo_max_steps=*) SFNER_GRPO_MAX_STEPS="${arg#*=}" ;;

    --sfner_test_batch_size=*) SFNER_TEST_BATCH_SIZE="${arg#*=}" ;;
    --sfner_max_test_samples=*) SFNER_MAX_TEST_SAMPLES="${arg#*=}" ;;

    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 --label=RUN_LABEL [-1] [-2] [-3] [-4] [-5] [--profile=standard|smoke] [--random_seed=42] [--sfner_sft_learning_rate=2e-4] [--sfner_grpo_beta=0.0] ..."
  exit 1
fi

# ---------- Smoke-test profile ----------

if [[ "$PROFILE" == "smoke" ]]; then
  for arg in "$@"; do
    case $arg in
      --label=*|-[0-9]*|--profile=*) ;;
      *)
        echo "--profile=smoke uses a fixed built-in configuration and does not accept hyperparameter overrides: $arg"
        exit 1
        ;;
    esac
  done

  RANDOM_SEED=42
  VALIDATION_SIZE=0.2
  TEST_SIZE=0.2
  TARGET_MODULES=all_linear
  MAX_RAW_SAMPLES=200

  SFNER_SFT_LEARNING_RATE=2e-4
  SFNER_SFT_LORA_RANK=16
  SFNER_SFT_BATCH_SIZE=16
  SFNER_SFT_GRADIENT_ACCUMULATION_STEPS=1
  SFNER_SFT_NUM_EPOCHS=1
  SFNER_SFT_MAX_TRAIN_SAMPLES=-1
  SFNER_SFT_MAX_VALIDATION_SAMPLES=24
  SFNER_SFT_SAVE_STEPS=1
  SFNER_SFT_EVAL_STEPS=1
  SFNER_SFT_MAX_STEPS=2

  SFNER_GRPO_LEARNING_RATE=5e-7
  SFNER_GRPO_LORA_RANK=16
  SFNER_GRPO_BATCH_SIZE=24
  SFNER_GRPO_GRADIENT_ACCUMULATION_STEPS=1
  SFNER_GRPO_NUM_GENERATIONS=8
  SFNER_GRPO_MAX_COMPLETION_LENGTH=128
  SFNER_GRPO_NUM_EPOCHS=1
  SFNER_GRPO_MAX_TRAIN_SAMPLES=-1
  SFNER_GRPO_MAX_VALIDATION_SAMPLES=24
  SFNER_GRPO_BETA=0.0
  SFNER_GRPO_TEMPERATURE=1.0
  SFNER_GRPO_EVAL_STEPS=1
  SFNER_GRPO_MAX_STEPS=2

  SFNER_TEST_BATCH_SIZE=16
  SFNER_MAX_TEST_SAMPLES=16

  export WANDB_MODE=disabled
elif [[ "$PROFILE" != "standard" ]]; then
  echo "Unknown --profile: $PROFILE (expected 'standard' or 'smoke')"
  exit 1
fi

# ***************************** PIPELINE *************************

# Returns true only if the step was passed
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

# Load env and env variables
# Move HF cache to data folder (keeps cache in the network volume)
export HF_HOME=data/.cache/huggingface
set -a; source .env; set +a
source .venv/bin/activate

# Set up paths
RUN_FOLDER="data/$LABEL"
CHECKPOINT_FOLDER_SFT="${LABEL}_sfner_sft"
CHECKPOINT_FOLDER_GRPO="${LABEL}_sfner_grpo"

if run_step 1; then
  echo "[STEP 1] TEST BASELINE MODEL"
  python src/test_sfner_pipeline.py \
    --metrics_path "$RUN_FOLDER/baseline_metrics.json" \
    --completions_path "$RUN_FOLDER/baseline_completions.parquet" \
    --model_repo "$MODEL_REPO" \
    --mode baseline \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --test_batch_size "$SFNER_TEST_BATCH_SIZE" \
    --grpo_max_completion_length "$SFNER_GRPO_MAX_COMPLETION_LENGTH" \
    --max_test_samples "$SFNER_MAX_TEST_SAMPLES" \
    --max_raw_samples "$MAX_RAW_SAMPLES"
fi

if run_step 2; then
  echo "[STEP 2] SFT TRAINING"
  python src/train_sfner_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/sft_out" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_SFT" \
    --model_repo "$MODEL_REPO" \
    --mode "sft" \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --target_modules "$TARGET_MODULES" \
    --sft_learning_rate "$SFNER_SFT_LEARNING_RATE" \
    --sft_lora_rank "$SFNER_SFT_LORA_RANK" \
    --sft_batch_size "$SFNER_SFT_BATCH_SIZE" \
    --sft_gradient_accumulation_steps "$SFNER_SFT_GRADIENT_ACCUMULATION_STEPS" \
    --sft_num_epochs "$SFNER_SFT_NUM_EPOCHS" \
    --sft_save_steps "$SFNER_SFT_SAVE_STEPS" \
    --sft_eval_steps "$SFNER_SFT_EVAL_STEPS" \
    --sft_max_steps "$SFNER_SFT_MAX_STEPS" \
    --grpo_learning_rate "$SFNER_GRPO_LEARNING_RATE" \
    --grpo_lora_rank "$SFNER_GRPO_LORA_RANK" \
    --grpo_batch_size "$SFNER_GRPO_BATCH_SIZE" \
    --grpo_gradient_accumulation_steps "$SFNER_GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --grpo_num_generations "$SFNER_GRPO_NUM_GENERATIONS" \
    --grpo_max_completion_length "$SFNER_GRPO_MAX_COMPLETION_LENGTH" \
    --grpo_num_epochs "$SFNER_GRPO_NUM_EPOCHS" \
    --grpo_beta "$SFNER_GRPO_BETA" \
    --grpo_temperature "$SFNER_GRPO_TEMPERATURE" \
    --grpo_eval_steps "$SFNER_GRPO_EVAL_STEPS" \
    --grpo_max_steps "$SFNER_GRPO_MAX_STEPS" \
    --max_raw_samples "$MAX_RAW_SAMPLES" \
    --sft_max_train_samples "$SFNER_SFT_MAX_TRAIN_SAMPLES" \
    --sft_max_validation_samples "$SFNER_SFT_MAX_VALIDATION_SAMPLES" \
    --grpo_max_train_samples "$SFNER_GRPO_MAX_TRAIN_SAMPLES" \
    --grpo_max_validation_samples "$SFNER_GRPO_MAX_VALIDATION_SAMPLES"
fi

if run_step 3; then
  echo "[STEP 3] TEST SFT MODEL"
  python src/test_sfner_pipeline.py \
    --metrics_path "$RUN_FOLDER/sft_metrics.json" \
    --completions_path "$RUN_FOLDER/sft_completions.parquet" \
    --model_repo "$MODEL_REPO" \
    --mode sft \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --test_batch_size "$SFNER_TEST_BATCH_SIZE" \
    --grpo_max_completion_length "$SFNER_GRPO_MAX_COMPLETION_LENGTH" \
    --max_test_samples "$SFNER_MAX_TEST_SAMPLES" \
    --max_raw_samples "$MAX_RAW_SAMPLES"
fi

if run_step 4; then
  echo "[STEP 4] GRPO TRAINING"
  python src/train_sfner_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/grpo_out" \
    --checkpoint_load_folder "$CHECKPOINT_FOLDER_SFT" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_GRPO" \
    --model_repo "$MODEL_REPO" \
    --mode "grpo" \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --target_modules "$TARGET_MODULES" \
    --sft_learning_rate "$SFNER_SFT_LEARNING_RATE" \
    --sft_lora_rank "$SFNER_SFT_LORA_RANK" \
    --sft_batch_size "$SFNER_SFT_BATCH_SIZE" \
    --sft_gradient_accumulation_steps "$SFNER_SFT_GRADIENT_ACCUMULATION_STEPS" \
    --sft_num_epochs "$SFNER_SFT_NUM_EPOCHS" \
    --sft_save_steps "$SFNER_SFT_SAVE_STEPS" \
    --sft_eval_steps "$SFNER_SFT_EVAL_STEPS" \
    --sft_max_steps "$SFNER_SFT_MAX_STEPS" \
    --grpo_learning_rate "$SFNER_GRPO_LEARNING_RATE" \
    --grpo_lora_rank "$SFNER_GRPO_LORA_RANK" \
    --grpo_batch_size "$SFNER_GRPO_BATCH_SIZE" \
    --grpo_gradient_accumulation_steps "$SFNER_GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --grpo_num_generations "$SFNER_GRPO_NUM_GENERATIONS" \
    --grpo_max_completion_length "$SFNER_GRPO_MAX_COMPLETION_LENGTH" \
    --grpo_num_epochs "$SFNER_GRPO_NUM_EPOCHS" \
    --grpo_beta "$SFNER_GRPO_BETA" \
    --grpo_temperature "$SFNER_GRPO_TEMPERATURE" \
    --grpo_eval_steps "$SFNER_GRPO_EVAL_STEPS" \
    --grpo_max_steps "$SFNER_GRPO_MAX_STEPS" \
    --max_raw_samples "$MAX_RAW_SAMPLES" \
    --sft_max_train_samples "$SFNER_SFT_MAX_TRAIN_SAMPLES" \
    --sft_max_validation_samples "$SFNER_SFT_MAX_VALIDATION_SAMPLES" \
    --grpo_max_train_samples "$SFNER_GRPO_MAX_TRAIN_SAMPLES" \
    --grpo_max_validation_samples "$SFNER_GRPO_MAX_VALIDATION_SAMPLES"
fi

if run_step 5; then
  echo "[STEP 5] TEST GRPO MODEL"
  python src/test_sfner_pipeline.py \
    --metrics_path "$RUN_FOLDER/grpo_metrics.json" \
    --completions_path "$RUN_FOLDER/grpo_completions.parquet" \
    --model_repo "$MODEL_REPO" \
    --mode grpo \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --grpo_checkpoint_folder "$CHECKPOINT_FOLDER_GRPO" \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --test_batch_size "$SFNER_TEST_BATCH_SIZE" \
    --grpo_max_completion_length "$SFNER_GRPO_MAX_COMPLETION_LENGTH" \
    --max_test_samples "$SFNER_MAX_TEST_SAMPLES" \
    --max_raw_samples "$MAX_RAW_SAMPLES"
fi

echo "[DONE]"
