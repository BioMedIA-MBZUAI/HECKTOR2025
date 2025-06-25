#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example-algorithm-sanity-check-task-2"


# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    DOCKER_IMAGE_TAG="$1"
fi

# Build the container
docker build "$SCRIPT_DIR" \
  --tag "$DOCKER_IMAGE_TAG" 2>&1

# Build the Container when developing with macOS
# docker build "$SCRIPT_DIR" \
#   --platform=linux/arm64/v8 \
#   --tag "$DOCKER_IMAGE_TAG" 2>&1