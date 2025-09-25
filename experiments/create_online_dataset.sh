#!/bin/bash
# Create online dataset for SpecForge training

set -e

# Configuration  
BASE_DATASET="/home/ronaksagtani/artifacts/spec-forge/dataset/data.jsonl"
NUM_SAMPLES=${1:-None}   # None means use all samples
EVAL_SAMPLES=${2:-10000}   # Default 1000 eval samples
OUTPUT_DIR="/home/ronaksagtani/artifacts/spec-forge/prepared_data"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Creating online dataset ($NUM_SAMPLES samples, $EVAL_SAMPLES eval)..."

# Check if base dataset exists
if [ ! -f "$BASE_DATASET" ]; then
    echo "❌ Error: Base dataset not found: $BASE_DATASET"
    exit 1
fi

# Create online dataset (no hidden states)
python "$SCRIPT_DIR/create_dataset.py" \
    --mode online \
    --base-dataset "$BASE_DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --eval-samples "$EVAL_SAMPLES"

echo "✅ Online dataset ready at: $OUTPUT_DIR"
