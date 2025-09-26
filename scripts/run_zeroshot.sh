#!/bin/bash

# Script to reproduce all BixBench paper results
# This script runs the zero-shot evaluations

set -e # Exit on error

echo "=========================================="
echo "BixBench Reproduction Script"
echo "=========================================="
echo ""

# Configuration
RESULTS_DIR="bixbench-v1.5_results"
ZERO_SHOT_DIR="${RESULTS_DIR}/zero_shot_baselines"
TRAJECTORIES_DIR="${RESULTS_DIR}/trajectories"
CONFIG_DIR="bixbench/run_configuration"
POSTPROCESS_CONFIG="bixbench/run_configuration/v1.5_paper_results.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required directories exist
if [ ! -d "$RESULTS_DIR" ]; then
    print_status "Creating results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"/{zero_shot_baselines,trajectories}
fi

# ==========================================
# PART 1: ZERO-SHOT EVALUATIONS
# ==========================================

print_status "Starting ZERO-SHOT evaluations..."
echo ""

# Function to run zero-shot evaluation
run_zero_shot() {
    local answer_mode=$1
    local model=$2
    local refusal_flag=$3
    local output_file=$4

    print_status "Running zero-shot: $model - $answer_mode $refusal_flag"

    if [ "$refusal_flag" = "--with-refusal" ]; then
        python generate_zeroshot_evals.py \
            --answer-mode "$answer_mode" \
            --model "$model" \
            --with-refusal \
            --output-dir "$ZERO_SHOT_DIR" \
            --output-file "$output_file" \
            --dataset-split "train"
    else
        python generate_zeroshot_evals.py \
            --answer-mode "$answer_mode" \
            --model "$model" \
            --output-dir "$ZERO_SHOT_DIR" \
            --output-file "$output_file" \
            --dataset-split "train"
    fi
}

# Run all zero-shot evaluations


if true; then
    print_status "Running GPT-4o zero-shot evaluations..."
    run_zero_shot "openanswer" "gpt-4o" "" "gpt-4o-grader-openended.csv"
    run_zero_shot "mcq" "gpt-4o" "--with-refusal" "gpt-4o-grader-mcq-refusal-True.csv"
    run_zero_shot "mcq" "gpt-4o" "" "gpt-4o-grader-mcq-refusal-False.csv"

    print_status "Running Claude-3.5-Sonnet zero-shot evaluations..."
    run_zero_shot "openanswer" "claude-3-5-sonnet-latest" "" "claude-3-5-sonnet-latest-grader-openended.csv"
    run_zero_shot "mcq" "claude-3-5-sonnet-latest" "--with-refusal" "claude-3-5-sonnet-latest-grader-mcq-refusal-True.csv"
    run_zero_shot "mcq" "claude-3-5-sonnet-latest" "" "claude-3-5-sonnet-latest-grader-mcq-refusal-False.csv"
fi

echo ""
print_status "Zero-shot evaluations complete!"
echo ""

# ==========================================
# PART 2: GRADE ZERO-SHOT RESULTS
# ==========================================

print_status "Grading zero-shot results..."
echo ""

# Function to grade zero-shot results
grade_zero_shot() {
    local input_file=$1
    local answer_mode=$2

    print_status "Grading: $input_file"

    python grade_outputs.py \
        --input-file "${ZERO_SHOT_DIR}/${input_file}" \
        --answer-mode "$answer_mode" \
        --output-dir "$ZERO_SHOT_DIR"
}

# Grade all zero-shot results
grade_zero_shot "gpt-4o-grader-openended.csv" "openanswer"
grade_zero_shot "gpt-4o-grader-mcq-refusal-True.csv" "mcq"
grade_zero_shot "gpt-4o-grader-mcq-refusal-False.csv" "mcq"
grade_zero_shot "claude-3-5-sonnet-latest-grader-openended.csv" "openanswer"
grade_zero_shot "claude-3-5-sonnet-latest-grader-mcq-refusal-True.csv" "mcq"
grade_zero_shot "claude-3-5-sonnet-latest-grader-mcq-refusal-False.csv" "mcq"

# Aggregate zero-shot results into a single JSON file
print_status "Aggregating zero-shot results..."
python -c "
import json
import glob
import os

results = {}
for file in glob.glob('${ZERO_SHOT_DIR}/*.json'):
    if 'zero_shot_baselines.json' not in file:
        with open(file, 'r') as f:
            data = json.load(f)
            basename = os.path.basename(file).replace('.json', '')
            results[basename] = data

with open('${ZERO_SHOT_DIR}/zero_shot_baselines.json', 'w') as f:
    json.dump(results, f, indent=4)
print(f'Aggregated {len(results)} zero-shot results')
"

echo ""
print_status "Zero-shot grading complete!"
echo ""
