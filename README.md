# HECKTOR2025 

This repository contains instructions and examples for creating a valid docker image for the [HECKTOR2025 challenge](https://hecktor25.grand-challenge.org/). 


# About

This repository contains the submission template and instructions for the [Grand Challenge 2025](https://hecktor25.grand-challenge.org/hecktor25/) docker-based inference task. Follow this guide to install Docker, run the baseline inference, observe challenge restrictions, save your container, and prepare your submission.

# Table of Contents

* [Installation](#installation)
* [Baseline Inference](#baseline-inference)
* [Restrictions and Submission Tips](#restrictions-and-submission-tips)
* [Saving and Uploading Containers](#saving-and-uploading-containers)
* [Reminders, Error Handling & Deadlines](#reminders-error-handling--deadlines)

# Installation

## Windows

1. Download Docker Desktop for Windows: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Run the installer and follow on-screen instructions.
3. Ensure Docker is running by opening PowerShell and executing:

   ```bash
   docker --version
   ```

## macOS

1. Download Docker Desktop for Mac: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Open the `.dmg` file and drag the Docker app to Applications.
3. Launch Docker and verify:

   ```bash
   docker --version
   ```

## Linux

1. Install prerequisites:

   ```bash
   sudo apt-get update && sudo apt-get install -y \
     apt-transport-https \
     ca-certificates \
     curl \
     software-properties-common
   ```
2. Add Docker’s GPG key and repository:

   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   sudo add-apt-repository \
     "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   ```
3. Install Docker Engine:

   ```bash
   sudo apt-get update && sudo apt-get install -y docker-ce
   ```
4. Verify installation:

   ```bash
   docker --version
   ```

# Baseline Inference

This section explains how to run the provided baseline inference container.

1. **Build or pull the image**:

   ```bash
   # To build locally
   docker build -t challenge-baseline .

   # Or pull from Docker Hub
   docker pull your-dockerhub-username/challenge-baseline:latest
   ```

2. **Prepare input data**

   * Place all test images in a local folder, e.g. `~/data/images/`.
   * **Do NOT** modify the `Dockerfile`, entrypoint script (`inference.sh`), or `/app/config/` directory.

3. **Run inference**:

   ```bash
   docker run --rm \
     -v ~/data/images:/tmp/images:ro \
     -v ~/data/output:/tmp/output:rw \
     challenge-baseline
   ```

   * Inputs must mount to `/tmp/images`
   * Outputs will be written to `/tmp/output`

# Restrictions and Submission Tips

* **No network access**: All downloads must occur before container startup.
* **No GPU**: Inference runs on CPU only.
* **I/O paths**: Read inputs only from `/tmp/images` and write outputs only to `/tmp/output`.
* **Time limit**: Entire inference must finish within **30 minutes**.
* **File writes**: Do not create or modify files outside `/tmp`

# Saving and Uploading Containers

1. **Commit the running container** (after testing):

   ```bash
   CONTAINER_ID=$(docker ps -lq)
   docker commit $CONTAINER_ID submission-image
   ```

2. **Save to tarball**:

   ```bash
   docker save submission-image -o my_submission.tar
   ```

3. **Upload to Sanity Check**:

   * Log in to the challenge portal.
   * Navigate to **My Submissions** → **Upload Container**.
   * Select `my_submission.tar` and submit.

# Reminders, Error Handling & Deadlines

* **Submission deadline**: July 31, 2025, 23:59 UTC.
* **Common errors**:

  * *Permission denied*: Ensure volume mounts use correct permissions (`:ro` for inputs).
  * *Timeout exceeded*: Optimize your model or reduce dataset size.
  * *File not found*: Verify that inputs are in `/tmp/images`.
* **Logging**: Capture logs by adding `-e DEBUG=1` or redirecting stdout to `/tmp/output/log.txt`.
* **Support**: Post questions on the challenge forum under **"Docker Inference"**.

Good luck with your submission!
