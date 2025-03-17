#!/bin/bash
# Script to list all BixBench evaluation runs

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}BixBench Evaluation Run History${NC}"
echo "------------------------------------"

if [ -f "bixbench_results/run_history.log" ]; then
    cat bixbench_results/run_history.log
else
    echo -e "${YELLOW}No run history found. Run the evaluation script first.${NC}"
fi

echo ""
echo -e "${GREEN}Available Run Directories:${NC}"
echo "------------------------------------"
ls -ld bixbench_results/*/ 2>/dev/null | grep -v "bixbench_results/multi_model" | grep -v "bixbench_results/multi_model_mcq" || echo -e "${YELLOW}No run directories found.${NC}"

echo ""
echo -e "${YELLOW}To view results for a specific run, navigate to:${NC}"
echo "  - bixbench_results/YOUR_RUN_ID/multi_model/ (open-ended questions)"
echo "  - bixbench_results/YOUR_RUN_ID/multi_model_mcq/ (multiple-choice questions)"