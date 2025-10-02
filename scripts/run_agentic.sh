#!/bin/bash

# Script to reproduce all BixBench paper results
# This script runs the main agentic evaluations

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
NUM_REPLICAS=5

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
# AGENTIC EVALUATIONS
# ==========================================

print_status "Starting AGENTIC evaluations..."
echo ""
print_warning "This will take a long time (24-48 hours total for all runs)"
echo ""

# List of all configurations
CONFIGS=(
    "4o_image"
    "4o_no_image"
    "claude_image"
    "claude_no_image"
)

# Function to run agentic evaluation
run_agentic() {
    local config_name=$1
    local replica_id=$2
    local config_file="${CONFIG_DIR}/${config_name}.yaml"

    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        return 1
    fi

    print_status "Running agentic evaluation: $config_name"
    print_status "Config file: $config_file"

    # Create a subdirectory for this run's trajectories
    mkdir -p "${TRAJECTORIES_DIR}/${config_name}"

    # Run the evaluation
    echo "Running replica $replica_id"
    python bixbench/generate_trajectories.py --config_file "$config_file" --replica_id "$replica_id"

    print_status "Completed: $config_name"
    echo ""
}

# Ask user if they want to run all agentic evaluations
echo "Do you want to run all agentic evaluations? (y/n)"
echo "WARNING: This will take 24-48 hours and requires API credits for both OpenAI and Anthropic"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    # Run all agentic evaluations
    for replica_id in $(seq 0 $NUM_REPLICAS); do
        for config in "${CONFIGS[@]}"; do
            run_agentic "$config" "$replica_id"
        done
    done

    print_status "All agentic evaluations complete!"
else
    print_warning "Skipping agentic evaluations. You can run them individually using:"
    echo "python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/<config_name>.yaml"
fi

echo ""

# ==========================================
# PART 4: POSTPROCESSING
# ==========================================

print_status "Running postprocessing to generate figures..."
echo ""

# Check if postprocessing config exists
if [ ! -f "$POSTPROCESS_CONFIG" ]; then
    print_error "Postprocessing config not found. Please check the config file path."
    exit 1
fi

# Run postprocessing
python bixbench/postprocessing.py --config_file "$POSTPROCESS_CONFIG"

print_status "Postprocessing complete! Figures saved to ${RESULTS_DIR}/figures"
echo ""

# ==========================================
# SUMMARY
# ==========================================

echo "=========================================="
echo "REPRODUCTION COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Zero-shot baselines: ${ZERO_SHOT_DIR}/"
echo "  - Trajectories: ${TRAJECTORIES_DIR}/"
echo ""
echo "To view the figures:"
echo "  - Performance comparison: ${RESULTS_DIR}/bixbench_results_comparison.png"
echo "  - Majority vote (refusal): ${RESULTS_DIR}/majority_vote_accuracy_refusal_option_comparison.png"
echo "  - Majority vote (images): ${RESULTS_DIR}/majority_vote_accuracy_image_comparison.png"
echo ""
