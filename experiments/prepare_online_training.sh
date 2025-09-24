#!/bin/bash
# Prepare dataset for SpecForge online training

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Configuration
INPUT_DATA=${1:-"/home/ronaksagtani/artifacts/spec-forge/dataset/data_sample_500.jsonl"}
OUTPUT_DIR=${2:-"/home/ronaksagtani/artifacts/spec-forge/prepared_data"}
OUTPUT_FILE="$OUTPUT_DIR/train_data.jsonl"

echo "Converting dataset for online training..."
echo "Input: $INPUT_DATA"
echo "Output: $OUTPUT_FILE"

# Create output directory
mkdir -p $OUTPUT_DIR

# Convert dataset directly to training location
source /opt/conda/bin/activate /opt/conda/envs/spec-forge-env
python $SCRIPT_DIR/convert_dataset.py "$INPUT_DATA" "$OUTPUT_FILE"

echo "Dataset ready for training: $OUTPUT_FILE"
