#!/bin/bash

# RUN THIS IN THE ROOT OF THE REPO SO THAT PWD WORKS CORRECTLY
# MAKE SURE TO INCLUDE A .env FILE IN THE ROOT OF THE REPO WITH YOUR HF_TOKEN

# Run docker container with the src directory mounted and dataset directory mounted

docker run \
    -v "$PWD/src":/workspace/src \
    -v "$PWD/data":/workspace/data \
    --rm \
    --memory="30g" \
    --gpus all \
    --env-file .env \
    universal-med-ner \
    bash src/test.sh