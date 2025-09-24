#!/bin/bash
# Create offline dataset with hidden states for SpecForge training

set -e

# Configuration  
BASE_DATASET="/home/ronaksagtani/artifacts/spec-forge/dataset/data.jsonl"
NUM_SAMPLES=${1:-100}   # None means use all samples
EVAL_SAMPLES=${2:-10}   # Default 500 eval samples
OUTPUT_DIR="/home/ronaksagtani/artifacts/spec-forge/prepared_data"
MODEL_PATH="sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k"

CHAT_TEMPLATE="qwen"
MAX_LENGTH=10000      # Accommodate typical 8024 token sequences
BATCH_SIZE=1          # Lower batch size for longer sequences
MEM_FRAC=0.75
MAX_LENGTH_FILTER=9000        # Filter out conversations longer than this
TOKENIZER_MODEL="$MODEL_PATH"  # Use same model for tokenizer

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Creating offline dataset ($NUM_SAMPLES samples, $EVAL_SAMPLES eval)..."

# Check if base dataset exists
if [ ! -f "$BASE_DATASET" ]; then
    echo "❌ Error: Base dataset not found: $BASE_DATASET"
    exit 1
fi

# Create offline dataset with hidden states
python "$SCRIPT_DIR/create_dataset.py" \
    --mode offline \
    --base-dataset "$BASE_DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --eval-samples "$EVAL_SAMPLES" \
    --model-path "$MODEL_PATH" \
    --chat-template "$CHAT_TEMPLATE" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --mem-frac "$MEM_FRAC" \
    --max-length-filter "$MAX_LENGTH_FILTER" \
    --tokenizer-model "$TOKENIZER_MODEL"

echo "✅ Offline dataset ready at: $OUTPUT_DIR"
