#!/bin/bash

export HF_HOME=data/.cache/huggingface
set -a; source .env; set +a
python3 src/train_pipeline.py "$@"