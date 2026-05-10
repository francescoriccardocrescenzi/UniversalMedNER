#!/bin/bash

# RUN THIS IN THE ROOT OF THE REPO SO THAT THE DOCKERFILE CAN BE FOUND

# Builds docker image

docker build -f Dockerfile -t universal-med-ner .