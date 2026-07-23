#!/usr/bin/env bash

# Fail if any command fails
set -e

# **************************** ARGUMENTS *************************************

# ------ Default values -------

LABEL=""
STEPS=()
PROFILE=standard

MODEL_REPO=google/medgemma-1.5-4b-it

NER_RANDOM_SEED=42
NER_VALIDATION_SIZE=0.02
NER_TEST_SIZE=0.05
NER_MAX_ENTITIES=6
NER_TARGET_MODULES=all_linear
NER_MAX_NEGATIVES=-1
NER_MAX_RAW_SAMPLES=-1

NER_SFT_LEARNING_RATE=2e-4
NER_SFT_LORA_RANK=16
NER_SFT_BATCH_SIZE=16
NER_SFT_GRADIENT_ACCUMULATION_STEPS=4
NER_SFT_NUM_EPOCHS=3
NER_SFT_MAX_TRAIN_SAMPLES=-1
NER_SFT_MAX_VALIDATION_SAMPLES=-1
NER_SFT_SAVE_STEPS=200
NER_SFT_EVAL_STEPS=200
NER_SFT_MAX_STEPS=-1

NER_GRPO_LEARNING_RATE=5e-7
NER_GRPO_LORA_RANK=16
NER_GRPO_BATCH_SIZE=24
NER_GRPO_GRADIENT_ACCUMULATION_STEPS=8
NER_GRPO_NUM_GENERATIONS=8
NER_GRPO_MAX_COMPLETION_LENGTH=128
NER_GRPO_NUM_EPOCHS=1
NER_GRPO_MAX_TRAIN_SAMPLES=6000
NER_GRPO_MAX_VALIDATION_SAMPLES=100
NER_GRPO_BETA=0.0
NER_GRPO_TEMPERATURE=1.0
NER_GRPO_EVAL_STEPS=50
NER_GRPO_MAX_STEPS=-1

NER_TEST_BATCH_SIZE=16
NER_MAX_TEST_SAMPLES=-1

# ----- Passed values -----

for arg in "$@"; do
  case $arg in
    --label=*) LABEL="${arg#*=}" ;;
    -[0-9]*) STEPS+=("${arg#-}") ;;
    --profile=*) PROFILE="${arg#*=}" ;;

    --model_repo=*) MODEL_REPO="${arg#*=}" ;;

    --ner_random_seed=*) NER_RANDOM_SEED="${arg#*=}" ;;
    --ner_validation_size=*) NER_VALIDATION_SIZE="${arg#*=}" ;;
    --ner_test_size=*) NER_TEST_SIZE="${arg#*=}" ;;
    --ner_max_entities=*) NER_MAX_ENTITIES="${arg#*=}" ;;
    --ner_target_modules=*) NER_TARGET_MODULES="${arg#*=}" ;;
    --ner_max_negatives=*) NER_MAX_NEGATIVES="${arg#*=}" ;;
    --ner_max_raw_samples=*) NER_MAX_RAW_SAMPLES="${arg#*=}" ;;

    --ner_sft_learning_rate=*) NER_SFT_LEARNING_RATE="${arg#*=}" ;;
    --ner_sft_lora_rank=*) NER_SFT_LORA_RANK="${arg#*=}" ;;
    --ner_sft_batch_size=*) NER_SFT_BATCH_SIZE="${arg#*=}" ;;
    --ner_sft_gradient_accumulation_steps=*) NER_SFT_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --ner_sft_num_epochs=*) NER_SFT_NUM_EPOCHS="${arg#*=}" ;;
    --ner_sft_max_train_samples=*) NER_SFT_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --ner_sft_max_validation_samples=*) NER_SFT_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;
    --ner_sft_save_steps=*) NER_SFT_SAVE_STEPS="${arg#*=}" ;;
    --ner_sft_eval_steps=*) NER_SFT_EVAL_STEPS="${arg#*=}" ;;
    --ner_sft_max_steps=*) NER_SFT_MAX_STEPS="${arg#*=}" ;;

    --ner_grpo_learning_rate=*) NER_GRPO_LEARNING_RATE="${arg#*=}" ;;
    --ner_grpo_lora_rank=*) NER_GRPO_LORA_RANK="${arg#*=}" ;;
    --ner_grpo_batch_size=*) NER_GRPO_BATCH_SIZE="${arg#*=}" ;;
    --ner_grpo_gradient_accumulation_steps=*) NER_GRPO_GRADIENT_ACCUMULATION_STEPS="${arg#*=}" ;;
    --ner_grpo_num_generations=*) NER_GRPO_NUM_GENERATIONS="${arg#*=}" ;;
    --ner_grpo_max_completion_length=*) NER_GRPO_MAX_COMPLETION_LENGTH="${arg#*=}" ;;
    --ner_grpo_num_epochs=*) NER_GRPO_NUM_EPOCHS="${arg#*=}" ;;
    --ner_grpo_max_train_samples=*) NER_GRPO_MAX_TRAIN_SAMPLES="${arg#*=}" ;;
    --ner_grpo_max_validation_samples=*) NER_GRPO_MAX_VALIDATION_SAMPLES="${arg#*=}" ;;
    --ner_grpo_beta=*) NER_GRPO_BETA="${arg#*=}" ;;
    --ner_grpo_temperature=*) NER_GRPO_TEMPERATURE="${arg#*=}" ;;
    --ner_grpo_eval_steps=*) NER_GRPO_EVAL_STEPS="${arg#*=}" ;;
    --ner_grpo_max_steps=*) NER_GRPO_MAX_STEPS="${arg#*=}" ;;

    --ner_test_batch_size=*) NER_TEST_BATCH_SIZE="${arg#*=}" ;;
    --ner_max_test_samples=*) NER_MAX_TEST_SAMPLES="${arg#*=}" ;;

    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 --label=RUN_LABEL [-1] [-2] [-3] [-4] [-5] [--profile=standard|smoke] [--ner_random_seed=42] [--ner_sft_learning_rate=2e-4] [--ner_grpo_beta=0.0] ..."
  exit 1
fi

# ---------- Smoke-test profile ----------
# Uses specific argument combination to run a small scale smoke test on local machine
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

  # NOTE: verify this repo loads via AutoModelForImageTextToText and carries
  # a Gemma-compatible tokenizer (<end_of_turn> etc.) before relying on this;
  # swap to a different tiny model repo here if it doesn't.
  MODEL_REPO=hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration

  NER_RANDOM_SEED=42
  NER_VALIDATION_SIZE=0.2
  NER_TEST_SIZE=0.2
  NER_MAX_ENTITIES=6
  NER_TARGET_MODULES=all_linear
  NER_MAX_NEGATIVES=-1
  NER_MAX_RAW_SAMPLES=40

  NER_SFT_LEARNING_RATE=2e-4
  NER_SFT_LORA_RANK=4
  NER_SFT_BATCH_SIZE=2
  NER_SFT_GRADIENT_ACCUMULATION_STEPS=1
  NER_SFT_NUM_EPOCHS=1
  NER_SFT_MAX_TRAIN_SAMPLES=8
  NER_SFT_MAX_VALIDATION_SAMPLES=4
  NER_SFT_SAVE_STEPS=1
  NER_SFT_EVAL_STEPS=1
  NER_SFT_MAX_STEPS=2

  NER_GRPO_LEARNING_RATE=5e-7
  NER_GRPO_LORA_RANK=4
  NER_GRPO_BATCH_SIZE=2
  NER_GRPO_GRADIENT_ACCUMULATION_STEPS=1
  NER_GRPO_NUM_GENERATIONS=2
  NER_GRPO_MAX_COMPLETION_LENGTH=16
  NER_GRPO_NUM_EPOCHS=1
  NER_GRPO_MAX_TRAIN_SAMPLES=8
  NER_GRPO_MAX_VALIDATION_SAMPLES=4
  NER_GRPO_BETA=0.0
  NER_GRPO_TEMPERATURE=1.0
  NER_GRPO_EVAL_STEPS=1
  NER_GRPO_MAX_STEPS=2

  NER_TEST_BATCH_SIZE=2
  NER_MAX_TEST_SAMPLES=4

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
CHECKPOINT_FOLDER_SFT="${LABEL}_ner_sft"
CHECKPOINT_FOLDER_GRPO="${LABEL}_ner_grpo"

if run_step 1; then
  echo "[STEP 1] TEST BASELINE MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/baseline_metrics.json" \
    --model_repo "$MODEL_REPO" \
    --ner_mode baseline \
    --verbose \
    --ner_random_seed "$NER_RANDOM_SEED" \
    --ner_validation_size "$NER_VALIDATION_SIZE" \
    --ner_test_size "$NER_TEST_SIZE" \
    --ner_max_entities "$NER_MAX_ENTITIES" \
    --ner_test_batch_size "$NER_TEST_BATCH_SIZE" \
    --ner_grpo_max_completion_length "$NER_GRPO_MAX_COMPLETION_LENGTH" \
    --ner_max_test_samples "$NER_MAX_TEST_SAMPLES" \
    --ner_max_raw_samples "$NER_MAX_RAW_SAMPLES"
fi

if run_step 2; then
  echo "[STEP 2] SFT TRAINING"
  python src/train_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/sft_out" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_SFT" \
    --model_repo "$MODEL_REPO" \
    --ner_mode "sft" \
    --ner_random_seed "$NER_RANDOM_SEED" \
    --ner_validation_size "$NER_VALIDATION_SIZE" \
    --ner_test_size "$NER_TEST_SIZE" \
    --ner_max_entities "$NER_MAX_ENTITIES" \
    --ner_target_modules "$NER_TARGET_MODULES" \
    --ner_sft_learning_rate "$NER_SFT_LEARNING_RATE" \
    --ner_sft_lora_rank "$NER_SFT_LORA_RANK" \
    --ner_sft_batch_size "$NER_SFT_BATCH_SIZE" \
    --ner_sft_gradient_accumulation_steps "$NER_SFT_GRADIENT_ACCUMULATION_STEPS" \
    --ner_sft_num_epochs "$NER_SFT_NUM_EPOCHS" \
    --ner_sft_save_steps "$NER_SFT_SAVE_STEPS" \
    --ner_sft_eval_steps "$NER_SFT_EVAL_STEPS" \
    --ner_sft_max_steps "$NER_SFT_MAX_STEPS" \
    --ner_grpo_learning_rate "$NER_GRPO_LEARNING_RATE" \
    --ner_grpo_lora_rank "$NER_GRPO_LORA_RANK" \
    --ner_grpo_batch_size "$NER_GRPO_BATCH_SIZE" \
    --ner_grpo_gradient_accumulation_steps "$NER_GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --ner_grpo_num_generations "$NER_GRPO_NUM_GENERATIONS" \
    --ner_grpo_max_completion_length "$NER_GRPO_MAX_COMPLETION_LENGTH" \
    --ner_grpo_num_epochs "$NER_GRPO_NUM_EPOCHS" \
    --ner_grpo_beta "$NER_GRPO_BETA" \
    --ner_grpo_temperature "$NER_GRPO_TEMPERATURE" \
    --ner_grpo_eval_steps "$NER_GRPO_EVAL_STEPS" \
    --ner_grpo_max_steps "$NER_GRPO_MAX_STEPS" \
    --ner_max_negatives "$NER_MAX_NEGATIVES" \
    --ner_max_raw_samples "$NER_MAX_RAW_SAMPLES" \
    --ner_sft_max_train_samples "$NER_SFT_MAX_TRAIN_SAMPLES" \
    --ner_sft_max_validation_samples "$NER_SFT_MAX_VALIDATION_SAMPLES" \
    --ner_grpo_max_train_samples "$NER_GRPO_MAX_TRAIN_SAMPLES" \
    --ner_grpo_max_validation_samples "$NER_GRPO_MAX_VALIDATION_SAMPLES"
fi

if run_step 3; then
  echo "[STEP 3] TEST SFT MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/sft_metrics.json" \
    --model_repo "$MODEL_REPO" \
    --ner_mode sft \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --verbose \
    --ner_random_seed "$NER_RANDOM_SEED" \
    --ner_validation_size "$NER_VALIDATION_SIZE" \
    --ner_test_size "$NER_TEST_SIZE" \
    --ner_max_entities "$NER_MAX_ENTITIES" \
    --ner_test_batch_size "$NER_TEST_BATCH_SIZE" \
    --ner_grpo_max_completion_length "$NER_GRPO_MAX_COMPLETION_LENGTH" \
    --ner_max_test_samples "$NER_MAX_TEST_SAMPLES" \
    --ner_max_raw_samples "$NER_MAX_RAW_SAMPLES"
fi

if run_step 4; then
  echo "[STEP 4] GRPO TRAINING"
  python src/train_pipeline.py \
    --label "$LABEL" \
    --save_folder "$RUN_FOLDER/grpo_out" \
    --checkpoint_load_folder "$CHECKPOINT_FOLDER_SFT" \
    --checkpoint_save_folder "$CHECKPOINT_FOLDER_GRPO" \
    --model_repo "$MODEL_REPO" \
    --ner_mode "grpo" \
    --ner_random_seed "$NER_RANDOM_SEED" \
    --ner_validation_size "$NER_VALIDATION_SIZE" \
    --ner_test_size "$NER_TEST_SIZE" \
    --ner_max_entities "$NER_MAX_ENTITIES" \
    --ner_target_modules "$NER_TARGET_MODULES" \
    --ner_sft_learning_rate "$NER_SFT_LEARNING_RATE" \
    --ner_sft_lora_rank "$NER_SFT_LORA_RANK" \
    --ner_sft_batch_size "$NER_SFT_BATCH_SIZE" \
    --ner_sft_gradient_accumulation_steps "$NER_SFT_GRADIENT_ACCUMULATION_STEPS" \
    --ner_sft_num_epochs "$NER_SFT_NUM_EPOCHS" \
    --ner_sft_save_steps "$NER_SFT_SAVE_STEPS" \
    --ner_sft_eval_steps "$NER_SFT_EVAL_STEPS" \
    --ner_sft_max_steps "$NER_SFT_MAX_STEPS" \
    --ner_grpo_learning_rate "$NER_GRPO_LEARNING_RATE" \
    --ner_grpo_lora_rank "$NER_GRPO_LORA_RANK" \
    --ner_grpo_batch_size "$NER_GRPO_BATCH_SIZE" \
    --ner_grpo_gradient_accumulation_steps "$NER_GRPO_GRADIENT_ACCUMULATION_STEPS" \
    --ner_grpo_num_generations "$NER_GRPO_NUM_GENERATIONS" \
    --ner_grpo_max_completion_length "$NER_GRPO_MAX_COMPLETION_LENGTH" \
    --ner_grpo_num_epochs "$NER_GRPO_NUM_EPOCHS" \
    --ner_grpo_beta "$NER_GRPO_BETA" \
    --ner_grpo_temperature "$NER_GRPO_TEMPERATURE" \
    --ner_grpo_eval_steps "$NER_GRPO_EVAL_STEPS" \
    --ner_grpo_max_steps "$NER_GRPO_MAX_STEPS" \
    --ner_max_negatives "$NER_MAX_NEGATIVES" \
    --ner_max_raw_samples "$NER_MAX_RAW_SAMPLES" \
    --ner_sft_max_train_samples "$NER_SFT_MAX_TRAIN_SAMPLES" \
    --ner_sft_max_validation_samples "$NER_SFT_MAX_VALIDATION_SAMPLES" \
    --ner_grpo_max_train_samples "$NER_GRPO_MAX_TRAIN_SAMPLES" \
    --ner_grpo_max_validation_samples "$NER_GRPO_MAX_VALIDATION_SAMPLES"
fi

if run_step 5; then
  echo "[STEP 5] TEST GRPO MODEL"
  python src/test_pipeline.py \
    --metrics_path "$RUN_FOLDER/grpo_metrics.json" \
    --model_repo "$MODEL_REPO" \
    --ner_mode grpo \
    --sft_checkpoint_folder "$CHECKPOINT_FOLDER_SFT" \
    --grpo_checkpoint_folder "$CHECKPOINT_FOLDER_GRPO" \
    --verbose \
    --ner_random_seed "$NER_RANDOM_SEED" \
    --ner_validation_size "$NER_VALIDATION_SIZE" \
    --ner_test_size "$NER_TEST_SIZE" \
    --ner_max_entities "$NER_MAX_ENTITIES" \
    --ner_test_batch_size "$NER_TEST_BATCH_SIZE" \
    --ner_grpo_max_completion_length "$NER_GRPO_MAX_COMPLETION_LENGTH" \
    --ner_max_test_samples "$NER_MAX_TEST_SAMPLES" \
    --ner_max_raw_samples "$NER_MAX_RAW_SAMPLES"
fi

echo "[DONE]"
