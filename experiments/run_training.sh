#!/bin/bash
# Run SpecForge online training

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Configuration
TARGET_MODEL="sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k"
CHAT_TEMPLATE="qwen"

# Auto-detect GPUs (allow override with first argument)
if [ "$1" != "" ]; then
    NUM_GPUS=$1
    echo "Using manually specified GPU count: $NUM_GPUS"
else
    # Auto-detect available GPUs
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NUM_GPUS GPUs"
    else
        echo "nvidia-smi not found, falling back to Python detection..."
        NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
        echo "Python detected $NUM_GPUS GPUs"
    fi
    
    # Fallback to 1 if detection failed
    if [ "$NUM_GPUS" -eq 0 ] 2>/dev/null || [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=1
        echo "GPU detection failed, using 1 GPU"
    fi
fi

# Paths
TRAIN_DATA="/home/ronaksagtani/artifacts/spec-forge/prepared_data/train_data_split.jsonl"
EVAL_DATA="/home/ronaksagtani/artifacts/spec-forge/prepared_data/eval_data.jsonl"
DRAFT_CONFIG="$ROOT_DIR/experiments/amp-tab-draft-config.json"
OUTPUT_DIR="/home/ronaksagtani/artifacts/spec-forge/outputs/qwen-draft-model"
CACHE_DIR="/home/ronaksagtani/artifacts/spec-forge/cache"

# Create directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "Starting SpecForge training..."
echo "Target model: $TARGET_MODEL"
echo "Training data: $TRAIN_DATA (400 samples)"
echo "Eval data: $EVAL_DATA (100 samples)"
echo "GPUs: $NUM_GPUS (tensor parallel)"
echo "TTT Length: 7 (predicting 7 tokens ahead)"
echo "Output: $OUTPUT_DIR"

# Activate conda environment
source /opt/conda/bin/activate /opt/conda/envs/spec-forge-env

# Load environment variables from .env file if it exists
if [ -f "$ROOT_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$ROOT_DIR/.env" | xargs)
elif [ -f ".env" ]; then
    echo "Loading environment variables from local .env file..."
    export $(grep -v '^#' ".env" | xargs)
fi

# Set up Weights & Biases
if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ ERROR: WANDB_API_KEY not found in environment or .env file"
    echo ""
    echo "Training requires Weights & Biases for experiment tracking."
    echo "Please set up authentication using one of these methods:"
    echo "   1. Set WANDB_API_KEY environment variable"
    echo "   2. Create a .env file with WANDB_API_KEY=your_key"
    echo "   3. Run 'wandb login' to authenticate"
    echo ""
    echo "You can find your API key at: https://wandb.ai/authorize"
    exit 1
fi
export WANDB_PROJECT="${WANDB_PROJECT:-spec-forge-training}"

# Run training
torchrun --standalone --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_online.py \
    --target-model-path "$TARGET_MODEL" \
    --draft-model-config "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --eval-data-path "$EVAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 1024 \
    --chat-template "$CHAT_TEMPLATE" \
    --cache-dir "$CACHE_DIR" \
    --embedding-key model.embed_tokens.weight \
    --attention-backend flex_attention \
    --ttt-length 7 \
    --tp-size $NUM_GPUS \
    --log-steps 1 \
    --save-interval 100 \
    --eval-interval 20 \
    --report-to wandb \
    --wandb-project spec-forge-training \
    --wandb-name "amp-tab-eagle3-poc"
