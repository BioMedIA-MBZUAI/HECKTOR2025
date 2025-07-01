#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example-algorithm-sanity-check-task-1"

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    DOCKER_IMAGE_TAG="$1"
fi

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh" "$DOCKER_IMAGE_TAG"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"
}


echo "=+= Cleaning up any earlier output"
if [ -d "$OUTPUT_DIR" ]; then
  # Ensure permissions are setup correctly
  # This allows for the Docker user to write to this location
  rm -rf "${OUTPUT_DIR}"/*
  chmod -f o+rwx "$OUTPUT_DIR"
else
  mkdir -m o+rwx "$OUTPUT_DIR"
fi

trap cleanup EXIT

echo "=+= Doing a forward pass"
## Note the extra arguments that are passed here:
# '--network none'
#    entails there is no internet connection
# 'gpus all'
#    enables access to any GPUs present
# '--volume <NAME>:/tmp'
#   is added because on Grand Challenge this directory cannot be used to store permanent files
docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    $DOCKER_IMAGE_TAG
docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null

# Ensure permissions are set correctly on the output
# This allows the host user (e.g. you) to access and handle these files
docker run --rm \
    --quiet \
    --env HOST_UID=`id -u` \
    --env HOST_GID=`id -g` \
    --volume "$OUTPUT_DIR":/output \
    alpine:latest \
    /bin/sh -c 'chown -R ${HOST_UID}:${HOST_GID} /output'

echo "=+= Wrote results to ${OUTPUT_DIR}"

echo "=+= Save this image for uploading via ./do_save.sh \"${DOCKER_IMAGE_TAG}\""