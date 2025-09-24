#!/bin/bash
# Run SpecForge EAGLE3 training

set -e

# Configuration
TARGET_MODEL="sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k"
CHAT_TEMPLATE="qwen"
NUM_GPUS=${1}  # Optional: override GPU count

# Paths
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
TRAIN_DATA="/home/ronaksagtani/artifacts/spec-forge/prepared_data/train_data.jsonl"
EVAL_DATA="/home/ronaksagtani/artifacts/spec-forge/prepared_data/eval_data.jsonl"
DRAFT_CONFIG="$ROOT_DIR/experiments/amp-tab-draft-config.json"
OUTPUT_BASE="/home/ronaksagtani/artifacts/spec-forge/outputs"

# Training parameters
NUM_EPOCHS=1
BATCH_SIZE=1
DRAFT_GLOBAL_BATCH_SIZE=32   # Reasonable global batch size for gradient updates
LEARNING_RATE=1e-4
MAX_LENGTH=8192
EMBEDDING_KEY="model.embed_tokens.weight"
ATTENTION_BACKEND="flex_attention"
TTT_LENGTH=7
LOG_STEPS=1
SAVE_INTERVAL=100
EVAL_INTERVAL=20
REPORT_TO="wandb"
WANDB_PROJECT="spec-forge-training"
WANDB_NAME="amp-tab-eagle3-poc"

echo "Starting SpecForge EAGLE3 training..."

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "spec-forge-env" ]; then
    echo "Activating spec-forge-env conda environment..."
    source /opt/conda/bin/activate /opt/conda/envs/spec-forge-env
fi

# Load environment variables
if [ -f "$ROOT_DIR/.env" ]; then
    export $(grep -v '^#' "$ROOT_DIR/.env" | xargs)
fi

# Check required files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found: $TRAIN_DATA"
    echo "Run: ./create_online_dataset.sh first"
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "❌ Error: Evaluation data not found: $EVAL_DATA"
    echo "Run: ./create_online_dataset.sh first"
    exit 1
fi

if [ ! -f "$DRAFT_CONFIG" ]; then
    echo "❌ Error: Draft config not found: $DRAFT_CONFIG"
    exit 1
fi

# Run training via Python wrapper
python "$SCRIPT_DIR/run_eagle3_training.py" \
    --target-model "$TARGET_MODEL" \
    --train-data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --draft-config "$DRAFT_CONFIG" \
    --output-dir "$OUTPUT_BASE" \
    --chat-template "$CHAT_TEMPLATE" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --draft-global-batch-size "$DRAFT_GLOBAL_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --max-length "$MAX_LENGTH" \
    --embedding-key "$EMBEDDING_KEY" \
    --attention-backend "$ATTENTION_BACKEND" \
    --ttt-length "$TTT_LENGTH" \
    --log-steps "$LOG_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --eval-interval "$EVAL_INTERVAL" \
    --report-to "$REPORT_TO" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$WANDB_NAME" \
    $([ -n "$NUM_GPUS" ] && echo "--num-gpus $NUM_GPUS")

echo "✅ Training script completed!"
