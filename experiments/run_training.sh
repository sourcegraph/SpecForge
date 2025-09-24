#!/bin/bash
# Run SpecForge online training

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Configuration
TARGET_MODEL="sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k"
CHAT_TEMPLATE="qwen"
NUM_GPUS=${1:-1}

# Paths
TRAIN_DATA="$ROOT_DIR/cache/dataset/train_data.jsonl"
DRAFT_CONFIG="$ROOT_DIR/experiments/draft_model_config.json"
OUTPUT_DIR="$ROOT_DIR/outputs/qwen-draft-model"
CACHE_DIR="$ROOT_DIR/cache"

# Create directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "Starting SpecForge training..."
echo "Target model: $TARGET_MODEL"
echo "Training data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"

# Run training
torchrun --standalone --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_online.py \
    --target-model-path "$TARGET_MODEL" \
    --draft-model-config "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template "$CHAT_TEMPLATE" \
    --cache-dir "$CACHE_DIR" \
    --attention-backend flex_attention \
    --ttt-length 7
