#!/bin/bash

# RUN THIS IN THE ROOT OF THE REPO SO THAT PWD WORKS CORRECTLY

# Run docker container with the src directory mounted and dataset directory mounted

docker run \
    -v "$PWD/src":/workspace/src \
    -v "$PWD/data":/workspace/data \
    --rm \
    --memory="30g" \
    --gpus all \
    universal-med-ner \
    bash src/test.sh