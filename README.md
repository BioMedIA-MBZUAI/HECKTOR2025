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

### Install prerequisites

  As most of the participants might be using Linux, so we provide detailed steps to set-up the docker.
  To create and test your docker setup, you will need to install [Docker Engine](https://docs.docker.com/engine/install/)
  and [NVIDIA-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (in case you need GPU computation).

- **Docker Engine:**
  - You can follow the instructions [here](https://docs.docker.com/engine/install/) to install `docker` on your system
    - ```command: sudo pacman -S docker```
    - Alternatively, you can also install using following steps:

      ```bash
      sudo apt-get update && sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        software-properties-common
      ```
    - Add Docker’s GPG key and repository:

      ```bash
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
      sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
      ```
    - Install Docker Engine:

      ```bash
      sudo apt-get update && sudo apt-get install -y docker-ce
      ```
    - Verify installation:

      ```bash
      docker --version
      ```
  - To verify that docker has been successfully installed, please run ```docker run hello-world```
  - If the above command does not work run ```systemctl status docker``` to check docker status. If inactive run ```systemctl start docker```.
  - If you can not run docker run ```hello-world``` without writing sudo first, you need to follow the instruction [here](https://docs.docker.com/engine/install/linux-postinstall/)
    - ```sudo groupadd docker```
    - ```sudo usermod -aG docker $USER```
    - restart your machine
- **Nvidia-container-toolkit:**
  - You can follow the instruction [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install `nvidia-container-toolkit` on you system
    - Following commands can be executed:
      - ```sudo pacman -Syu nvidia-container-toolkit```
      - ```sudo nvidia-ctk runtime configure --runtime=docker```
      - ```sudo systemctl restart docker```
  - To verify you can access gpu inside docker by running ```docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi```
  - git and git-lfs and a github account (grand challenge website to clone a repo):
    - make sure you have access to git by running ```git --version```


# Baseline Inference

# Baseline Inference

The **main** branch hosts the baseline model setup. Users can add their model code here, then build, test, and save their Docker container. We support three separate tasks: **Task1**, **Task2**, and **Task3**. For each task, create a dedicated model under its folder.

1. **Repository Structure**

   ```text
   ├── Task1/
   │   ├── resources/       # Place your model files here
   │   ├── requirements.txt # Modify only to add new packages
   │   └── ...
   ├── Task2/
   │   └── resources/
   ├── Task3/
   │   └── resources/
   ├── Dockerfile.template # Base Dockerfile for reference
   ├── do_build.sh         # Script to build container
   ├── do_test_run.sh      # Script to test container locally
   ├── do_save.sh          # Script to save container as tarball
   └── inference.py        # Entry point: loads models, runs inference
   ```

2. **Model Files and Packages**

   * Place all model weights, configuration files, or auxiliary code inside the `resources/` folder of the corresponding Task directory.
   * You **may** update `requirements.txt` within each Task folder to install any additional Python packages needed by your model.
   * **Do NOT** modify any other files or directories.

3. **Working Directory**

   * All input and output files during inference must be read from or written to the `/tmp/` directory inside the container.

4. **Build the Container**

   ```bash
   # From repo root
   ./do_build.sh main           # builds an image tagged `challenge:<branch>`
   ```

5. **Local Test Run**

   ```bash
   # Runs inference locally mounting /tmp data
   ./do_test_run.sh main
   ```

6. **Commit and Save**

   ```bash
   ./do_save.sh main my_submission.tar
   ```

7. **Entry Point**

   * `inference.py` is the script executed at container runtime. Implement or call your model-loading and prediction code here.


   <!-- * Inputs must mount to `/tmp/images`
   * Outputs will be written to `/tmp/output` -->

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
