#!/bin/bash
# Pre-process dataset offline to avoid distributed training timeouts

set -e

# Configuration
TARGET_MODEL="sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k"
CHAT_TEMPLATE="qwen"
MAX_LENGTH=8192
BUILD_NUM_PROC=8

# Paths
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
TRAIN_DATA="/home/ronaksagtani/artifacts/spec-forge/prepared_data/train_data.jsonl"
EVAL_DATA="/home/ronaksagtani/artifacts/spec-forge/prepared_data/eval_data.jsonl"
DRAFT_CONFIG="$ROOT_DIR/experiments/amp-tab-draft-config.json"
CACHE_DIR="/home/ronaksagtani/artifacts/spec-forge/cache"

echo "Pre-processing dataset for faster training..."

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "spec-forge-env" ]; then
    echo "Activating spec-forge-env conda environment..."
    source /opt/conda/bin/activate /opt/conda/envs/spec-forge-env
fi

# Load environment variables
if [ -f "$ROOT_DIR/.env" ]; then
    export $(grep -v '^#' "$ROOT_DIR/.env" | xargs)
fi

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found: $TRAIN_DATA"
    echo "Run: ./create_online_dataset.sh first"
    exit 1
fi

# Check if eval data exists (optional)
if [ ! -f "$EVAL_DATA" ]; then
    echo "⚠️ Warning: Evaluation data not found: $EVAL_DATA"
    echo "Will process only training data"
    EVAL_DATA=""
fi

# Create cache directory
mkdir -p "$CACHE_DIR"

# Pre-process datasets
if [ -n "$EVAL_DATA" ]; then
    python "$SCRIPT_DIR/preprocess_dataset.py" \
        --target-model "$TARGET_MODEL" \
        --train-data "$TRAIN_DATA" \
        --eval-data "$EVAL_DATA" \
        --draft-config "$DRAFT_CONFIG" \
        --cache-dir "$CACHE_DIR" \
        --chat-template "$CHAT_TEMPLATE" \
        --max-length "$MAX_LENGTH" \
        --build-dataset-num-proc "$BUILD_NUM_PROC"
else
    python "$SCRIPT_DIR/preprocess_dataset.py" \
        --target-model "$TARGET_MODEL" \
        --train-data "$TRAIN_DATA" \
        --draft-config "$DRAFT_CONFIG" \
        --cache-dir "$CACHE_DIR" \
        --chat-template "$CHAT_TEMPLATE" \
        --max-length "$MAX_LENGTH" \
        --build-dataset-num-proc "$BUILD_NUM_PROC"
fi

echo "✅ Dataset pre-processing completed!"
echo "You can now run training with reduced startup time."
