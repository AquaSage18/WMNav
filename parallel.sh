#!/bin/bash

# Configuration Variables
ROOT_DIR=/home/yangnan/projects/WMNav
CONDA_PATH=/opt/anaconda3/etc/profile.d/conda.sh
NUM_GPU=2
INSTANCES=20
NUM_EPISODES_PER_INSTANCE=40
MAX_STEPS_PER_EPISODE=40
TASK="ObjectNav"
DATASET="hm3d_v0.1"
CFG="WMNav"
NAME="wmnav-qwen3.6-hm3dv1-20*40eps"
SESSION_NAME_PREFIX="wmnav_qwen36_hm3dv1_20x40eps"
PROJECT_NAME="WMNav"
VENV_NAME="wmnav" # Name of the conda environment
GPU_LIST=(6 7) # List of GPU IDs to use
SLEEP_INTERVAL=200
LOG_FREQ=1
PORT=2000
QWEN_BASE_URL="http://127.0.0.1:8001/v1"
QWEN_API_KEY="EMPTY"
DETACH_AFTER_START=${DETACH_AFTER_START:-0}
WMNAV_RUN_TIMESTAMP=${WMNAV_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --dataset ${DATASET}"

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="aggregator_${SESSION_NAME_PREFIX}"

# Start Aggregator Session
tmux new-session -d -s "$AGGREGATOR_SESSION" "bash --noprofile --norc"
tmux send-keys -t "$AGGREGATOR_SESSION" "source ${CONDA_PATH} && conda activate ${VENV_NAME} && cd ${ROOT_DIR} && export QWEN_BASE_URL=${QWEN_BASE_URL} QWEN_API_KEY=${QWEN_API_KEY} WMNAV_RUN_TIMESTAMP=${WMNAV_RUN_TIMESTAMP} && python scripts/aggregator.py --name ${TASK}_${NAME} --project ${PROJECT_NAME} --sleep ${SLEEP_INTERVAL} --config ${CFG} --port ${PORT}" C-m
SESSION_NAMES+=("$AGGREGATOR_SESSION")

# Cleanup Function
cleanup() {
  echo "\nCaught interrupt signal. Cleaning up tmux sessions..."

  for session in "${SESSION_NAMES[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      echo "Killed session: $session"
    fi
  done

}

# Trap SIGINT to Run Cleanup
trap cleanup SIGINT

# Start Tmux Sessions for Each Instance
for instance_id in $(seq 0 $((INSTANCES - 1))); do
  #GPU_ID=$((instance_id % NUM_GPU))
  GPU_ID=${GPU_LIST[$((instance_id % ${#GPU_LIST[@]}))]}
  SESSION_NAME="${TASK}_${SESSION_NAME_PREFIX}_${instance_id}_of_${INSTANCES}"

  tmux new-session -d -s "$SESSION_NAME" "bash --noprofile --norc"
  tmux send-keys -t "$SESSION_NAME" "source ${CONDA_PATH} && conda activate ${VENV_NAME} && cd ${ROOT_DIR} && export QWEN_BASE_URL=${QWEN_BASE_URL} QWEN_API_KEY=${QWEN_API_KEY} WMNAV_RUN_TIMESTAMP=${WMNAV_RUN_TIMESTAMP} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id" C-m
  SESSION_NAMES+=("$SESSION_NAME")
done

if [ "$DETACH_AFTER_START" = "1" ]; then
  echo "Started ${INSTANCES} worker sessions and aggregator session. Detaching without monitoring."
  exit 0
fi

# Monitor Tmux Sessions
while true; do
  sleep $SLEEP_INTERVAL

  ALL_DONE=true

  for instance_id in $(seq 0 $((INSTANCES - 1))); do
    SESSION_NAME="${TASK}_${SESSION_NAME_PREFIX}_${instance_id}_of_${INSTANCES}"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "$SESSION_NAME finished"
    else
      ALL_DONE=false
    fi
  done

  if $ALL_DONE; then
    echo "DONE"
    echo "$(date): Sending termination signal to aggregator."
    curl -X POST http://localhost:${PORT}/terminate
    if [ $? -eq 0 ]; then
      echo "$(date): Termination signal sent successfully."
    else
      echo "$(date): Failed to send termination signal."
    fi

    sleep 10
    if tmux has-session -t "$AGGREGATOR_SESSION" 2>/dev/null; then
      tmux kill-session -t "$AGGREGATOR_SESSION"
      echo "Killed session: $AGGREGATOR_SESSION"
    fi
    break
  fi

done
