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

LBL_SFT_LEARNING_RATE=2e-4
LBL_SFT_LORA_RANK=16
LBL_SFT_BATCH_SIZE=16
LBL_SFT_GRADIENT_ACCUMULATION_STEPS=4
LBL_SFT_NUM_EPOCHS=3
LBL_SFT_MAX_TRAIN_SAMPLES=-1
LBL_SFT_MAX_VALIDATION_SAMPLES=-1
LBL_SFT_SAVE_STEPS=200
LBL_SFT_EVAL_STEPS=200
LBL_SFT_MAX_STEPS=-1

LBL_GRPO_LEARNING_RATE=5e-7
LBL_GRPO_LORA_RANK=16
LBL_GRPO_BATCH_SIZE=24
LBL_GRPO_GRADIENT_ACCUMULATION_STEPS=8
LBL_GRPO_NUM_GENERATIONS=8
LBL_GRPO_MAX_COMPLETION_LENGTH=128
LBL_GRPO_NUM_EPOCHS=1
LBL_GRPO_MAX_TRAIN_SAMPLES=6000
LBL_GRPO_MAX_VALIDATION_SAMPLES=100
LBL_GRPO_BETA=0.0
LBL_GRPO_TEMPERATURE=1.0
LBL_GRPO_EVAL_STEPS=50
LBL_GRPO_MAX_STEPS=-1

LBL_TEST_BATCH_SIZE=16
LBL_MAX_TEST_SAMPLES=-1

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

    --lbl_sft_learning_rate=*) LBL_SFT_LEARNING_RATE="${arg#*=}" ;;
    --lbl_sft_lora_rank=*) LBL_SFT_LORA_RANK="${arg#*=}" ;;
    --lbl_sft_batch_size=*) LBL_SFT_BATCH_SIZE="${arg#*=}" ;;
    --lbl_sft_gradient_accumulation_steps=*) LBL_SFT_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --lbl_sft_num_epochs=*) LBL_SFT_NUM_EPOCHS="${arg#*=}" ;;
    --lbl_sft_max_train_samples=*) LBL_SFT_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --lbl_sft_max_validation_samples=*) LBL_SFT_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;
    --lbl_sft_save_steps=*) LBL_SFT_SAVE_STEPS="${arg#*=}" ;;
    --lbl_sft_eval_steps=*) LBL_SFT_EVAL_STEPS="${arg#*=}" ;;
    --lbl_sft_max_steps=*) LBL_SFT_MAX_STEPS="${arg#*=}" ;;

    --lbl_grpo_learning_rate=*) LBL_GRPO_LEARNING_RATE="${arg#*=}" ;;
    --lbl_grpo_lora_rank=*) LBL_GRPO_LORA_RANK="${arg#*=}" ;;
    --lbl_grpo_batch_size=*) LBL_GRPO_BATCH_SIZE="${arg#*=}" ;;
    --lbl_grpo_gradient_accumulation_steps=*) LBL_GRPO_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --lbl_grpo_num_generations=*) LBL_GRPO_NUM_GENERATIONS="${arg#*=}" ;;
    --lbl_grpo_max_completion_length=*) LBL_GRPO_MAX_COMPLETION_LENGTH="${arg#*=}" ;;
    --lbl_grpo_num_epochs=*) LBL_GRPO_NUM_EPOCHS="${arg#*=}" ;;
    --lbl_grpo_max_train_samples=*) LBL_GRPO_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --lbl_grpo_max_validation_samples=*) LBL_GRPO_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;
    --lbl_grpo_beta=*) LBL_GRPO_BETA="${arg#*=}" ;;
    --lbl_grpo_temperature=*) LBL_GRPO_TEMPERATURE="${arg#*=}" ;;
    --lbl_grpo_eval_steps=*) LBL_GRPO_EVAL_STEPS="${arg#*=}" ;;
    --lbl_grpo_max_steps=*) LBL_GRPO_MAX_STEPS="${arg#*=}" ;;

    --lbl_test_batch_size=*) LBL_TEST_BATCH_SIZE="${arg#*=}" ;;
    --lbl_max_test_samples=*) LBL_MAX_TEST_SAMPLES="${arg#*=}" ;;

    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 --label=RUN_LABEL [-1] [-2] [-3] [-4] [-5] [--profile=standard|smoke] [--random_seed=42] [--lbl_sft_learning_rate=2e-4] [--lbl_grpo_beta=0.0] ..."
  exit 1
fi

# ---------- Smoke-test profile ----------
# Fast end-to-end check on the real model (on the VM), using very little data
# Does not accept hyperparameter overrides

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

  LBL_SFT_LEARNING_RATE=2e-4
  LBL_SFT_LORA_RANK=16
  LBL_SFT_BATCH_SIZE=16
  LBL_SFT_GRADIENT_ACCUMULATION_STEPS=1
  LBL_SFT_NUM_EPOCHS=1
  LBL_SFT_MAX_TRAIN_SAMPLES=-1
  LBL_SFT_MAX_VALIDATION_SAMPLES=24
  LBL_SFT_SAVE_STEPS=1
  LBL_SFT_EVAL_STEPS=1
  LBL_SFT_MAX_STEPS=2

  LBL_GRPO_LEARNING_RATE=5e-7
  LBL_GRPO_LORA_RANK=16
  LBL_GRPO_BATCH_SIZE=24
  LBL_GRPO_GRADIENT_ACCUMULATION_STEPS=1
  LBL_GRPO_NUM_GENERATIONS=8
  LBL_GRPO_MAX_COMPLETION_LENGTH=128
  LBL_GRPO_NUM_EPOCHS=1
  LBL_GRPO_MAX_TRAIN_SAMPLES=-1
  LBL_GRPO_MAX_VALIDATION_SAMPLES=24
  LBL_GRPO_BETA=0.0
  LBL_GRPO_TEMPERATURE=1.0
  LBL_GRPO_EVAL_STEPS=1
  LBL_GRPO_MAX_STEPS=2

  LBL_TEST_BATCH_SIZE=16
  LBL_MAX_TEST_SAMPLES=16

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
CHECKPOINT_FOLDER_SFT="${LABEL}_lbl_sft"
CHECKPOINT_FOLDER_GRPO="${LABEL}_lbl_grpo"

if run_step 1; then
  echo "[STEP 1] TEST BASELINE MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/baseline_metrics.json" \
    --model_repo "$MODEL_REPO" \
    --task lbl \
    --mode baseline \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --lbl_test_batch_size "$LBL_TEST_BATCH_SIZE" \
    --lbl_grpo_max_completion_length "$LBL_GRPO_MAX_COMPLETION_LENGTH" \
    --lbl_max_test_samples "$LBL_MAX_TEST_SAMPLES" \
    --max_raw_samples "$MAX_RAW_SAMPLES"
fi

if run_step 2; then
  echo "[STEP 2] SFT TRAINING"
  python src/train_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/sft_out" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_SFT" \
    --model_repo "$MODEL_REPO" \
    --task lbl \
    --mode "sft" \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --target_modules "$TARGET_MODULES" \
    --lbl_sft_learning_rate "$LBL_SFT_LEARNING_RATE" \
    --lbl_sft_lora_rank "$LBL_SFT_LORA_RANK" \
    --lbl_sft_batch_size "$LBL_SFT_BATCH_SIZE" \
    --lbl_sft_gradient_accumulation_steps "$LBL_SFT_GRADIENT_ACCUMULATION_STEPS" \
    --lbl_sft_num_epochs "$LBL_SFT_NUM_EPOCHS" \
    --lbl_sft_save_steps "$LBL_SFT_SAVE_STEPS" \
    --lbl_sft_eval_steps "$LBL_SFT_EVAL_STEPS" \
    --lbl_sft_max_steps "$LBL_SFT_MAX_STEPS" \
    --lbl_grpo_learning_rate "$LBL_GRPO_LEARNING_RATE" \
    --lbl_grpo_lora_rank "$LBL_GRPO_LORA_RANK" \
    --lbl_grpo_batch_size "$LBL_GRPO_BATCH_SIZE" \
    --lbl_grpo_gradient_accumulation_steps "$LBL_GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --lbl_grpo_num_generations "$LBL_GRPO_NUM_GENERATIONS" \
    --lbl_grpo_max_completion_length "$LBL_GRPO_MAX_COMPLETION_LENGTH" \
    --lbl_grpo_num_epochs "$LBL_GRPO_NUM_EPOCHS" \
    --lbl_grpo_beta "$LBL_GRPO_BETA" \
    --lbl_grpo_temperature "$LBL_GRPO_TEMPERATURE" \
    --lbl_grpo_eval_steps "$LBL_GRPO_EVAL_STEPS" \
    --lbl_grpo_max_steps "$LBL_GRPO_MAX_STEPS" \
    --max_raw_samples "$MAX_RAW_SAMPLES" \
    --lbl_sft_max_train_samples "$LBL_SFT_MAX_TRAIN_SAMPLES" \
    --lbl_sft_max_validation_samples "$LBL_SFT_MAX_VALIDATION_SAMPLES" \
    --lbl_grpo_max_train_samples "$LBL_GRPO_MAX_TRAIN_SAMPLES" \
    --lbl_grpo_max_validation_samples "$LBL_GRPO_MAX_VALIDATION_SAMPLES"
fi

if run_step 3; then
  echo "[STEP 3] TEST SFT MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/sft_metrics.json" \
    --model_repo "$MODEL_REPO" \
    --task lbl \
    --mode sft \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --lbl_test_batch_size "$LBL_TEST_BATCH_SIZE" \
    --lbl_grpo_max_completion_length "$LBL_GRPO_MAX_COMPLETION_LENGTH" \
    --lbl_max_test_samples "$LBL_MAX_TEST_SAMPLES" \
    --max_raw_samples "$MAX_RAW_SAMPLES"
fi

if run_step 4; then
  echo "[STEP 4] GRPO TRAINING"
  python src/train_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/grpo_out" \
    --checkpoint_load_folder "$CHECKPOINT_FOLDER_SFT" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_GRPO" \
    --model_repo "$MODEL_REPO" \
    --task lbl \
    --mode "grpo" \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --target_modules "$TARGET_MODULES" \
    --lbl_sft_learning_rate "$LBL_SFT_LEARNING_RATE" \
    --lbl_sft_lora_rank "$LBL_SFT_LORA_RANK" \
    --lbl_sft_batch_size "$LBL_SFT_BATCH_SIZE" \
    --lbl_sft_gradient_accumulation_steps "$LBL_SFT_GRADIENT_ACCUMULATION_STEPS" \
    --lbl_sft_num_epochs "$LBL_SFT_NUM_EPOCHS" \
    --lbl_sft_save_steps "$LBL_SFT_SAVE_STEPS" \
    --lbl_sft_eval_steps "$LBL_SFT_EVAL_STEPS" \
    --lbl_sft_max_steps "$LBL_SFT_MAX_STEPS" \
    --lbl_grpo_learning_rate "$LBL_GRPO_LEARNING_RATE" \
    --lbl_grpo_lora_rank "$LBL_GRPO_LORA_RANK" \
    --lbl_grpo_batch_size "$LBL_GRPO_BATCH_SIZE" \
    --lbl_grpo_gradient_accumulation_steps "$LBL_GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --lbl_grpo_num_generations "$LBL_GRPO_NUM_GENERATIONS" \
    --lbl_grpo_max_completion_length "$LBL_GRPO_MAX_COMPLETION_LENGTH" \
    --lbl_grpo_num_epochs "$LBL_GRPO_NUM_EPOCHS" \
    --lbl_grpo_beta "$LBL_GRPO_BETA" \
    --lbl_grpo_temperature "$LBL_GRPO_TEMPERATURE" \
    --lbl_grpo_eval_steps "$LBL_GRPO_EVAL_STEPS" \
    --lbl_grpo_max_steps "$LBL_GRPO_MAX_STEPS" \
    --max_raw_samples "$MAX_RAW_SAMPLES" \
    --lbl_sft_max_train_samples "$LBL_SFT_MAX_TRAIN_SAMPLES" \
    --lbl_sft_max_validation_samples "$LBL_SFT_MAX_VALIDATION_SAMPLES" \
    --lbl_grpo_max_train_samples "$LBL_GRPO_MAX_TRAIN_SAMPLES" \
    --lbl_grpo_max_validation_samples "$LBL_GRPO_MAX_VALIDATION_SAMPLES"
fi

if run_step 5; then
  echo "[STEP 5] TEST GRPO MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/grpo_metrics.json" \
    --model_repo "$MODEL_REPO" \
    --task lbl \
    --mode grpo \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --grpo_checkpoint_folder "$CHECKPOINT_FOLDER_GRPO" \
    --verbose \
    --random_seed "$RANDOM_SEED" \
    --validation_size "$VALIDATION_SIZE" \
    --test_size "$TEST_SIZE" \
    --lbl_test_batch_size "$LBL_TEST_BATCH_SIZE" \
    --lbl_grpo_max_completion_length "$LBL_GRPO_MAX_COMPLETION_LENGTH" \
    --lbl_max_test_samples "$LBL_MAX_TEST_SAMPLES" \
    --max_raw_samples "$MAX_RAW_SAMPLES"
fi

echo "[DONE]"
