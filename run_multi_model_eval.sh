#!/bin/bash
# Script to run BixBench evaluations on multiple models

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting BixBench multi-model evaluation${NC}"

# Get current date and time
DATETIME=$(date +"%Y%m%d_%H%M%S")

# Prompt for run name
echo -e "${YELLOW}Please enter a name for this evaluation run (will be combined with datetime):${NC}"
read RUN_NAME

# If run name is empty, use "run" as default
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="run"
fi

# Create the run ID by combining name and datetime
RUN_ID="${RUN_NAME}_${DATETIME}"
echo -e "${GREEN}Using run ID: ${RUN_ID}${NC}"

# Create necessary directories
mkdir -p data/trajectories
mkdir -p "bixbench_results/${RUN_ID}/multi_model"
mkdir -p "bixbench_results/${RUN_ID}/multi_model_mcq"

# Update configuration files with the new paths
sed -i.bak "s|results_dir: \"bixbench_results/multi_model\"|results_dir: \"bixbench_results/${RUN_ID}/multi_model\"|g" bixbench/run_configuration/multi_model_postprocessing.yaml
sed -i.bak "s|results_dir: \"bixbench_results/multi_model_mcq\"|results_dir: \"bixbench_results/${RUN_ID}/multi_model_mcq\"|g" bixbench/run_configuration/multi_model_mcq_postprocessing.yaml

# Function to run a model evaluation
run_model() {
    MODEL_NAME=$1
    CONFIG_FILE=$2
    
    echo -e "${YELLOW}Running ${MODEL_NAME} evaluation...${NC}"
    uv run python bixbench/generate_trajectories.py --config_file "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}${MODEL_NAME} evaluation completed successfully${NC}"
    else
        echo -e "${RED}${MODEL_NAME} evaluation failed${NC}"
    fi
}

# Run open-ended question evaluations
echo -e "${YELLOW}Running open-ended question evaluations...${NC}"

run_model "GPT-4o" "bixbench/run_configuration/gpt4o_trajectories.yaml"
run_model "Claude 3.5" "bixbench/run_configuration/claude35_trajectories.yaml"
run_model "Claude 3.7" "bixbench/run_configuration/claude37_trajectories.yaml"
# DeepSeek R1 commented out for initial experiment
# run_model "DeepSeek R1" "bixbench/run_configuration/deepseek_trajectories.yaml"

# Run MCQ evaluations
echo -e "${YELLOW}Running multiple-choice question evaluations...${NC}"

run_model "GPT-4o MCQ" "bixbench/run_configuration/gpt4o_mcq_trajectories.yaml"
run_model "Claude 3.5 MCQ" "bixbench/run_configuration/claude35_mcq_trajectories.yaml"
run_model "Claude 3.7 MCQ" "bixbench/run_configuration/claude37_mcq_trajectories.yaml"
# DeepSeek R1 commented out for initial experiment
# run_model "DeepSeek R1 MCQ" "bixbench/run_configuration/deepseek_mcq_trajectories.yaml"

# Run postprocessing for open-ended questions
echo -e "${YELLOW}Running postprocessing for open-ended questions...${NC}"
uv run python bixbench/postprocessing.py --config_file bixbench/run_configuration/multi_model_postprocessing.yaml

# Run postprocessing for MCQ
echo -e "${YELLOW}Running postprocessing for multiple-choice questions...${NC}"
uv run python bixbench/postprocessing.py --config_file bixbench/run_configuration/multi_model_mcq_postprocessing.yaml

# Restore original configuration files
mv bixbench/run_configuration/multi_model_postprocessing.yaml.bak bixbench/run_configuration/multi_model_postprocessing.yaml 2>/dev/null || true
mv bixbench/run_configuration/multi_model_mcq_postprocessing.yaml.bak bixbench/run_configuration/multi_model_mcq_postprocessing.yaml 2>/dev/null || true

echo -e "${GREEN}Evaluation complete! Results are available in:${NC}"
echo -e "  - bixbench_results/${RUN_ID}/multi_model/ (open-ended questions)"
echo -e "  - bixbench_results/${RUN_ID}/multi_model_mcq/ (multiple-choice questions)"

# Log the run details
echo "Completed run: ${RUN_ID} at $(date)" >> bixbench_results/run_history.log