#!/bin/bash

# ==============================================================================
# Bash script to run 4-fold cross-validation training in parallel using 'screen'.
# Each fold is assigned to a separate GPU.
#
# Each fold will be launched in its own detached 'screen' session, allowing you
# to safely close your terminal while the training continues.
#
# USAGE:
# 1. Save this file as `run_training.sh`.
# 2. Make it executable: `chmod +x run_training.sh`
# 3. Modify the `TRAIN_SCRIPT` and `COMMON_ARGS` variables below if needed.
# 4. Ensure you have at least 4 GPUs available.
# 5. Run the script: `./run_training.sh`
#
# MONITORING:
# - To see GPU usage: `watch nvidia-smi`
# - To see all running screen sessions: `screen -ls`
# - To attach to a specific session (e.g., fold 0): `screen -r fold_0`
# - To detach from a session: Press `Ctrl+A`, then `d`.
# ==============================================================================

# --- Configuration ---
# The main Python script to execute for training.
TRAIN_SCRIPT="train.py"

# Any other command-line arguments that are common to all folds.
# For example: "--config /path/to/config.yaml --epochs 100"
COMMON_ARGS="--config unetr"

# Number of folds to run (from 0 to 3 for a total of 4)
NUM_FOLDS=4

echo "--- Starting 4-Fold Cross-Validation Training on 4 GPUs ---"

# Loop through the folds from 0 to 3
for (( fold=0; fold<${NUM_FOLDS}; fold++ ))
do
  # Define a unique name for the screen session for this fold
  SESSION_NAME="fold_${fold}"

  # Define the full command to be executed.
  # We set CUDA_VISIBLE_DEVICES to the fold number to assign each fold
  # to a unique GPU device (Fold 0 -> GPU 0, Fold 1 -> GPU 1, etc.)
  FULL_COMMAND="CUDA_VISIBLE_DEVICES=${fold} python ${TRAIN_SCRIPT} --fold ${fold} ${COMMON_ARGS}"

  echo "Launching training for Fold ${fold} on GPU ${fold} in screen session '${SESSION_NAME}'..."
  
  # Create a new detached screen session and run the training command inside it.
  # -d -m: Starts the screen in a detached mode.
  # -S: Specifies the session name.
  screen -d -m -S ${SESSION_NAME} bash -c "${FULL_COMMAND}"

done

echo ""
echo "--- All training sessions have been launched. ---"
echo "Use 'screen -ls' to see the list of running sessions."
echo "Use 'watch nvidia-smi' to monitor GPU utilization."
echo "Use 'screen -r fold_0' (or fold_1, etc.) to attach and monitor a specific fold."