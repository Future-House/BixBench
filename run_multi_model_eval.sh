#!/bin/bash
# Script to run BixBench evaluations on multiple models

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_OPENENDED=true
RUN_MCQ=true
RUN_GPT4O=true
RUN_CLAUDE35=true
RUN_CLAUDE37=true
RUN_DEEPSEEK=false
MAX_CONCURRENT=20  # Default concurrency level for API calls
RUN_MINI=false     # Mini mode for quick testing with only 10 questions per type
DOCKER_IMAGE="aviary-notebook-env"  # Default Docker image for Aviary

# Display help function
show_help() {
    echo -e "${BLUE}BixBench Multi-Model Evaluation Script${NC}"
    echo -e "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help                  Show this help message"
    echo "  --open-ended-only       Run only open-ended evaluations"
    echo "  --mcq-only              Run only multiple-choice questions evaluations"
    echo "  --gpt4o-only            Run only GPT-4o model"
    echo "  --claude35-only         Run only Claude 3.5 model"
    echo "  --claude37-only         Run only Claude 3.7 model"
    echo "  --deepseek              Include DeepSeek model in the evaluation"
    echo "  --models=[list]         Comma-separated list of models to run (gpt4o,claude35,claude37,deepseek)"
    echo "  --concurrency=[num]     Set maximum concurrent API requests (default: 20)"
    echo "  --mini                  Run in mini mode with only 10 questions per type (fast testing)"
    echo "  --docker-image=[name]   Specify Docker image for Aviary (default: aviary-notebook-env)"
    echo ""
    echo "Examples:"
    echo "  $0 --mcq-only --claude35-only  # Run only Claude 3.5 MCQ evaluations"
    echo "  $0 --models=gpt4o,claude37     # Run only GPT-4o and Claude 3.7 (both open-ended and MCQ)"
    echo "  $0 --models=gpt4o --concurrency=25  # Run GPT-4o with 25 concurrent API requests"
    echo "  $0 --mini --gpt4o-only         # Run a quick test with GPT-4o on only 10 examples per type"
    echo "  $0 --docker-image=aviary-notebook-env-fixed  # Use fixed version of Aviary notebook environment"
    exit 0
}

# Process command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            ;;
        --open-ended-only)
            RUN_OPENENDED=true
            RUN_MCQ=false
            ;;
        --mcq-only)
            RUN_OPENENDED=false
            RUN_MCQ=true
            ;;
        --gpt4o-only)
            RUN_GPT4O=true
            RUN_CLAUDE35=false
            RUN_CLAUDE37=false
            RUN_DEEPSEEK=false
            ;;
        --claude35-only)
            RUN_GPT4O=false
            RUN_CLAUDE35=true
            RUN_CLAUDE37=false
            RUN_DEEPSEEK=false
            ;;
        --claude37-only)
            RUN_GPT4O=false
            RUN_CLAUDE35=false
            RUN_CLAUDE37=true
            RUN_DEEPSEEK=false
            ;;
        --deepseek)
            RUN_DEEPSEEK=true
            ;;
        --mini)
            RUN_MINI=true
            ;;
        --docker-image=*)
            DOCKER_IMAGE="${1#*=}"
            ;;
        --concurrency=*)
            MAX_CONCURRENT="${1#*=}"
            # Validate that it's a positive integer
            if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [ "$MAX_CONCURRENT" -eq 0 ]; then
                echo -e "${RED}Error: Concurrency must be a positive integer${NC}"
                exit 1
            fi
            ;;
        --models=*)
            # Reset all models to false first
            RUN_GPT4O=false
            RUN_CLAUDE35=false
            RUN_CLAUDE37=false
            RUN_DEEPSEEK=false
            
            # Parse the comma-separated list
            IFS=',' read -ra MODELS <<< "${1#*=}"
            for model in "${MODELS[@]}"; do
                case "$model" in
                    gpt4o)
                        RUN_GPT4O=true
                        ;;
                    claude35)
                        RUN_CLAUDE35=true
                        ;;
                    claude37)
                        RUN_CLAUDE37=true
                        ;;
                    deepseek)
                        RUN_DEEPSEEK=true
                        ;;
                    *)
                        echo -e "${RED}Error: Unknown model '$model'${NC}"
                        exit 1
                        ;;
                esac
            done
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            show_help
            ;;
    esac
    shift
done

echo -e "${GREEN}Starting BixBench multi-model evaluation${NC}"

# Display what will be run
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Docker image: ${GREEN}${DOCKER_IMAGE}${NC}"
echo -e "  Open-ended evaluations: $([ "$RUN_OPENENDED" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}")"
echo -e "  MCQ evaluations: $([ "$RUN_MCQ" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}")"
echo -e "  Mini mode (10 questions per type): $([ "$RUN_MINI" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}")"
echo -e "  Max concurrent API requests: ${GREEN}${MAX_CONCURRENT}${NC}"
echo -e "  Models:"
echo -e "    GPT-4o: $([ "$RUN_GPT4O" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}")"
echo -e "    Claude 3.5: $([ "$RUN_CLAUDE35" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}")"
echo -e "    Claude 3.7: $([ "$RUN_CLAUDE37" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}")"
echo -e "    DeepSeek: $([ "$RUN_DEEPSEEK" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}")"

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
echo -e "${YELLOW}Creating run directories for ${RUN_ID}...${NC}"

# Base directories
mkdir -p "data/trajectories/${RUN_ID}"
mkdir -p "bixbench_results/${RUN_ID}/multi_model"
mkdir -p "bixbench_results/${RUN_ID}/multi_model_mcq"

# Create model-specific subdirectories for selected models only
if [ "$RUN_GPT4O" = true ]; then
    echo -e "${BLUE}Creating GPT-4o directories${NC}"
    mkdir -p "data/trajectories/${RUN_ID}/gpt4o"
    if [ "$RUN_MCQ" = true ]; then
        mkdir -p "data/trajectories/${RUN_ID}/gpt4o_mcq"
    fi
fi

if [ "$RUN_CLAUDE35" = true ]; then
    echo -e "${BLUE}Creating Claude 3.5 directories${NC}"
    mkdir -p "data/trajectories/${RUN_ID}/claude35"
    if [ "$RUN_MCQ" = true ]; then
        mkdir -p "data/trajectories/${RUN_ID}/claude35_mcq"
    fi
fi

if [ "$RUN_CLAUDE37" = true ]; then
    echo -e "${BLUE}Creating Claude 3.7 directories${NC}"
    mkdir -p "data/trajectories/${RUN_ID}/claude37"
    if [ "$RUN_MCQ" = true ]; then
        mkdir -p "data/trajectories/${RUN_ID}/claude37_mcq"
    fi
fi

if [ "$RUN_DEEPSEEK" = true ]; then
    echo -e "${BLUE}Creating DeepSeek directories${NC}"
    mkdir -p "data/trajectories/${RUN_ID}/deepseek"
    if [ "$RUN_MCQ" = true ]; then
        mkdir -p "data/trajectories/${RUN_ID}/deepseek_mcq"
    fi
fi

# Create dynamic configuration directory for this run
RUN_CONFIG_DIR="bixbench/run_configuration/run_${RUN_ID}"
mkdir -p "$RUN_CONFIG_DIR"
echo -e "${YELLOW}Creating dynamic configuration files in ${RUN_CONFIG_DIR}${NC}"

# Function to generate model-specific configuration
generate_model_config() {
    MODEL_NAME=$1
    MODEL_ID=$2
    LLM_NAME=$3
    BATCH_SIZE=$4
    MODE=$5  # "open" or "mcq"
    MAX_TOKENS=${6:-""}  # Optional max_tokens parameter
    
    # Set up mode-specific settings
    if [ "$MODE" = "mcq" ]; then
        RUN_NAME="bixbench-${MODEL_ID}-mcq"
        SYSTEM_PROMPT="CAPSULE_SYSTEM_PROMPT_MCQ"
        TRAJECTORY_DIR="data/trajectories/${RUN_ID}/${MODEL_ID}_mcq"
    else
        RUN_NAME="bixbench-${MODEL_ID}"
        SYSTEM_PROMPT="CAPSULE_SYSTEM_PROMPT_OPEN"
        TRAJECTORY_DIR="data/trajectories/${RUN_ID}/${MODEL_ID}"
    fi
    
    # Set mini mode batch size if needed
    if [ "$RUN_MINI" = true ]; then
        BATCH_SIZE=4
    fi
    
    # Create the configuration file
    CONFIG_FILE="${RUN_CONFIG_DIR}/${MODEL_ID}_${MODE}_trajectories.yaml"
    
    echo -e "${BLUE}Generating configuration for ${MODEL_NAME} (${MODE} mode)${NC}"
    
    # Generate the YAML content
    cat > "$CONFIG_FILE" << EOL
# Dynamically generated config for ${MODEL_NAME} (${MODE} mode)
# Docker image: ${DOCKER_IMAGE}
run_name: "${RUN_NAME}"

agent:
  agent_type: "ReActAgent"
  agent_kwargs:
    llm_model:
      name: "${LLM_NAME}"
      parallel_tool_calls: false
      num_retries: 6
      temperature: 1.0
      request_timeout: 180  # Increased timeout to handle rate limit delays
EOL

    # Add max_tokens if provided (mainly for Claude models)
    if [ -n "$MAX_TOKENS" ]; then
        echo "      max_tokens: ${MAX_TOKENS}  # Set max output tokens to help stay within output token limits" >> "$CONFIG_FILE"
    fi
    
    # Continue with the rest of the config
    cat >> "$CONFIG_FILE" << EOL
    hide_old_env_states: true
rollout:
  max_steps: 40
  batch_size: ${BATCH_SIZE}
  rollout_type: "aviary"

notebook:
  name: "notebook.ipynb"
  language: "python"

capsule:
  mode: "${MODE}"
  include_refusal_option: true
  system_prompt: "${SYSTEM_PROMPT}"
  prompt_templates:
    mcq: "MCQ_PROMPT_TEMPLATE"
    open: "OPEN_PROMPT_TEMPLATE"
    hypothesis: "HYPOTHESIS_PROMPT_TEMPLATE"
  eval_mode: null  # When set to None, the capsule will not evaluate the answer
  avoid_images: true

paths:
  workspace_dir: "data/workspace"
  trajectories_dir: "${TRAJECTORY_DIR}"
  data_folder: "data/capsules"
  hf_repo_id: "futurehouse/bixbench"
EOL

    # Add mini mode settings if needed
    if [ "$RUN_MINI" = true ]; then
        echo -e "\n# Mini mode settings for faster testing\nmini_mode: true\nmax_problems: 10" >> "$CONFIG_FILE"
    fi
    
    echo -e "${GREEN}Created configuration file: ${CONFIG_FILE}${NC}"
    return "$CONFIG_FILE"
}

# Generate model-specific configurations for selected models
if [ "$RUN_OPENENDED" = true ]; then
    if [ "$RUN_GPT4O" = true ]; then
        generate_model_config "GPT-4o" "gpt4o" "gpt-4o" 8 "open"
    fi
    
    if [ "$RUN_CLAUDE35" = true ]; then
        generate_model_config "Claude 3.5 Sonnet" "claude35" "claude-3-5-sonnet-20241022" 6 "open" 4000
    fi
    
    if [ "$RUN_CLAUDE37" = true ]; then
        generate_model_config "Claude 3.7 Sonnet" "claude37" "claude-3-7-sonnet-20250219" 6 "open" 4000
    fi
    
    if [ "$RUN_DEEPSEEK" = true ]; then
        generate_model_config "DeepSeek R1" "deepseek" "deepseek-coder:1.5-live" 5 "open"
    fi
fi

if [ "$RUN_MCQ" = true ]; then
    if [ "$RUN_GPT4O" = true ]; then
        generate_model_config "GPT-4o" "gpt4o" "gpt-4o" 8 "mcq"
    fi
    
    if [ "$RUN_CLAUDE35" = true ]; then
        generate_model_config "Claude 3.5 Sonnet" "claude35" "claude-3-5-sonnet-20241022" 6 "mcq" 4000
    fi
    
    if [ "$RUN_CLAUDE37" = true ]; then
        generate_model_config "Claude 3.7 Sonnet" "claude37" "claude-3-7-sonnet-20250219" 6 "mcq" 4000
    fi
    
    if [ "$RUN_DEEPSEEK" = true ]; then
        generate_model_config "DeepSeek R1" "deepseek" "deepseek-coder:1.5-live" 5 "mcq"
    fi
fi

# Function to run a model evaluation
run_model() {
    MODEL_NAME=$1
    MODEL_ID=$2
    MODE=$3  # "open" or "mcq"
    
    # Use our dynamically generated config files
    CONFIG_FILE="${RUN_CONFIG_DIR}/${MODEL_ID}_${MODE}_trajectories.yaml"
    
    # Add mini indicator to model name if in mini mode
    if [ "$RUN_MINI" = true ]; then
        MODEL_NAME="${MODEL_NAME} (Mini)"
    fi
    
    echo -e "${YELLOW}Running ${MODEL_NAME} evaluation...${NC}"
    uv run python bixbench/generate_trajectories.py --config_file "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}${MODEL_NAME} evaluation completed successfully${NC}"
    else
        echo -e "${RED}${MODEL_NAME} evaluation failed${NC}"
    fi
}

# Run open-ended question evaluations
if [ "$RUN_OPENENDED" = true ]; then
    echo -e "${YELLOW}Running open-ended question evaluations...${NC}"
    
    if [ "$RUN_GPT4O" = true ]; then
        run_model "GPT-4o" "gpt4o" "open"
    fi
    
    if [ "$RUN_CLAUDE35" = true ]; then
        run_model "Claude 3.5" "claude35" "open"
    fi
    
    if [ "$RUN_CLAUDE37" = true ]; then
        run_model "Claude 3.7" "claude37" "open"
    fi
    
    if [ "$RUN_DEEPSEEK" = true ]; then
        run_model "DeepSeek R1" "deepseek" "open"
    fi
fi

# Run MCQ evaluations
if [ "$RUN_MCQ" = true ]; then
    echo -e "${YELLOW}Running multiple-choice question evaluations...${NC}"
    
    if [ "$RUN_GPT4O" = true ]; then
        run_model "GPT-4o MCQ" "gpt4o" "mcq"
    fi
    
    if [ "$RUN_CLAUDE35" = true ]; then
        run_model "Claude 3.5 MCQ" "claude35" "mcq"
    fi
    
    if [ "$RUN_CLAUDE37" = true ]; then
        run_model "Claude 3.7 MCQ" "claude37" "mcq"
    fi
    
    if [ "$RUN_DEEPSEEK" = true ]; then
        run_model "DeepSeek R1 MCQ" "deepseek" "mcq"
    fi
fi

# Function to create filtered postprocessing config
create_filtered_config() {
    OUTPUT_CONFIG=$1
    IS_MCQ=$2
    
    # Initialize an empty config file
    echo -e "${BLUE}Creating completely new filtered config for active models only${NC}"
    
    # Create the base configuration with only the relevant paths
    cat > "$OUTPUT_CONFIG" << EOL
# Generated filtered config for BixBench postprocessing - $(date)
# This is a clean configuration that only includes selected models
results_dir: "bixbench_results/${RUN_ID}/$(if [ "$IS_MCQ" = "true" ]; then echo "multi_model_mcq"; else echo "multi_model"; fi)"
data_path: "data/trajectories/${RUN_ID}"
debug: true

EOL

    # Add majority voting configuration only for selected models
    cat >> "$OUTPUT_CONFIG" << EOL
# Majority vote configuration
majority_vote:
  run: true
  k_value: 10
  groups:
    model_comparison:
EOL

    # Add only enabled models to majority_vote config
    if [ "$RUN_GPT4O" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "      - \"bixbench-gpt4o-mcq\"" >> "$OUTPUT_CONFIG"
        else
            echo "      - \"bixbench-gpt4o\"" >> "$OUTPUT_CONFIG"
        fi
    fi
    if [ "$RUN_CLAUDE35" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "      - \"bixbench-claude35-mcq\"" >> "$OUTPUT_CONFIG"
        else
            echo "      - \"bixbench-claude35\"" >> "$OUTPUT_CONFIG"
        fi
    fi
    if [ "$RUN_CLAUDE37" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "      - \"bixbench-claude37-mcq\"" >> "$OUTPUT_CONFIG"
        else
            echo "      - \"bixbench-claude37\"" >> "$OUTPUT_CONFIG"
        fi
    fi
    if [ "$RUN_DEEPSEEK" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "      - \"bixbench-deepseek-mcq\"" >> "$OUTPUT_CONFIG"
        else
            echo "      - \"bixbench-deepseek\"" >> "$OUTPUT_CONFIG"
        fi
    fi

    # Add run comparison configuration for selected models
    cat >> "$OUTPUT_CONFIG" << EOL

# Run comparison configuration
run_comparison:
  run: true
  run_name_groups:
EOL

    # Add only enabled models to run_name_groups
    if [ "$RUN_GPT4O" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "    - [\"bixbench-gpt4o-mcq\"]" >> "$OUTPUT_CONFIG"
        else
            echo "    - [\"bixbench-gpt4o\"]" >> "$OUTPUT_CONFIG"
        fi
    fi
    if [ "$RUN_CLAUDE35" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "    - [\"bixbench-claude35-mcq\"]" >> "$OUTPUT_CONFIG"
        else
            echo "    - [\"bixbench-claude35\"]" >> "$OUTPUT_CONFIG"
        fi
    fi
    if [ "$RUN_CLAUDE37" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "    - [\"bixbench-claude37-mcq\"]" >> "$OUTPUT_CONFIG"
        else
            echo "    - [\"bixbench-claude37\"]" >> "$OUTPUT_CONFIG"
        fi
    fi
    if [ "$RUN_DEEPSEEK" = true ]; then
        if [ "$IS_MCQ" = "true" ]; then
            echo "    - [\"bixbench-deepseek-mcq\"]" >> "$OUTPUT_CONFIG"
        else
            echo "    - [\"bixbench-deepseek\"]" >> "$OUTPUT_CONFIG"
        fi
    fi

    # Add titles for enabled models
    echo "  group_titles:" >> "$OUTPUT_CONFIG"
    if [ "$RUN_GPT4O" = true ]; then
        echo "    - \"GPT-4o\"" >> "$OUTPUT_CONFIG"
    fi
    if [ "$RUN_CLAUDE35" = true ]; then
        echo "    - \"Claude 3.5\"" >> "$OUTPUT_CONFIG"
    fi
    if [ "$RUN_CLAUDE37" = true ]; then
        echo "    - \"Claude 3.7\"" >> "$OUTPUT_CONFIG"
    fi
    if [ "$RUN_DEEPSEEK" = true ]; then
        echo "    - \"DeepSeek R1\"" >> "$OUTPUT_CONFIG"
    fi

    # Add color groups for enabled models
    echo "  color_groups:" >> "$OUTPUT_CONFIG"
    if [ "$RUN_GPT4O" = true ]; then
        echo "    - \"gpt4o\"" >> "$OUTPUT_CONFIG"
    fi
    if [ "$RUN_CLAUDE35" = true ]; then
        echo "    - \"claude35\"" >> "$OUTPUT_CONFIG"
    fi
    if [ "$RUN_CLAUDE37" = true ]; then
        echo "    - \"claude37\"" >> "$OUTPUT_CONFIG"
    fi
    if [ "$RUN_DEEPSEEK" = true ]; then
        echo "    - \"deepseek\"" >> "$OUTPUT_CONFIG"
    fi

    # Set default total questions count
    TOTAL_QUESTIONS=296  # Default
    echo "  total_questions_per_run: $TOTAL_QUESTIONS" >> "$OUTPUT_CONFIG"
    
    # Add baselines configuration
    echo "  use_zero_shot_baselines: false" >> "$OUTPUT_CONFIG"
    echo "  baseline_name_mappings: {}" >> "$OUTPUT_CONFIG"
    
    # Add random baselines based on mode
    if [ "$IS_MCQ" = "true" ]; then
        echo "  random_baselines: [0.2, 0.25]  # Random baselines for MCQ (with refusal: 0.2, without: 0.25)" >> "$OUTPUT_CONFIG"
    else
        echo "  random_baselines: []" >> "$OUTPUT_CONFIG"
    fi
    
    echo -e "${GREEN}Created clean filtered config at $OUTPUT_CONFIG${NC}"
}

# Run postprocessing for open-ended questions
if [ "$RUN_OPENENDED" = true ]; then
    echo -e "${YELLOW}Running postprocessing for open-ended questions...${NC}"
    
    # Create a filtered config for selected models (open-ended)
    FILTERED_CONFIG="bixbench/run_configuration/filtered_multi_model_postprocessing.yaml"
    create_filtered_config "$FILTERED_CONFIG" "false"
    
    # Add the --max_concurrent flag to limit API calls
    # Use higher default concurrency (20) which will be automatically adjusted by model
    uv run python bixbench/postprocessing.py "$FILTERED_CONFIG" --max_concurrent ${MAX_CONCURRENT:-20}
    
    # Cleanup
    rm "$FILTERED_CONFIG" 2>/dev/null || true
fi

# Run postprocessing for MCQ
if [ "$RUN_MCQ" = true ]; then
    echo -e "${YELLOW}Running postprocessing for multiple-choice questions...${NC}"
    
    # Create a filtered config for selected models (MCQ)
    FILTERED_CONFIG="bixbench/run_configuration/filtered_multi_model_mcq_postprocessing.yaml"
    create_filtered_config "$FILTERED_CONFIG" "true"
    
    # Add the --max_concurrent flag to limit API calls
    # Use higher default concurrency (20) which will be automatically adjusted by model
    uv run python bixbench/postprocessing.py "$FILTERED_CONFIG" --max_concurrent ${MAX_CONCURRENT:-20}
    
    # Cleanup
    rm "$FILTERED_CONFIG" 2>/dev/null || true
fi

# No need to restore any files since we're generating everything dynamically
# and not modifying any original configuration files

# Optionally clean up our dynamic config directory (disabled by default)
# rm -rf "$RUN_CONFIG_DIR" 2>/dev/null || true
# Instead we keep it for debugging or reference purposes

echo -e "${GREEN}Evaluation complete! Results are available in:${NC}"
if [ "$RUN_OPENENDED" = true ]; then
    echo -e "  - bixbench_results/${RUN_ID}/multi_model/ (open-ended questions)"
fi
if [ "$RUN_MCQ" = true ]; then
    echo -e "  - bixbench_results/${RUN_ID}/multi_model_mcq/ (multiple-choice questions)"
fi

# Summarize the run configuration
echo -e "${BLUE}Run configuration:${NC}"
echo -e "  - Run ID: ${RUN_ID}"
echo -e "  - Docker image: ${DOCKER_IMAGE}"
echo -e "  - Generated configs: ${RUN_CONFIG_DIR}/"
echo -e "  - Models evaluated: $([ "$RUN_GPT4O" = true ] && echo "GPT-4o " || echo "")$([ "$RUN_CLAUDE35" = true ] && echo "Claude3.5 " || echo "")$([ "$RUN_CLAUDE37" = true ] && echo "Claude3.7 " || echo "")$([ "$RUN_DEEPSEEK" = true ] && echo "DeepSeek " || echo "")"
echo -e "  - Question types: $([ "$RUN_OPENENDED" = true ] && echo "Open-ended " || echo "")$([ "$RUN_MCQ" = true ] && echo "MCQ " || echo "")"
echo -e "  - Mini mode: $([ "$RUN_MINI" = true ] && echo "Yes" || echo "No")"

# Log the run details
echo "Completed run: ${RUN_ID} at $(date)" >> bixbench_results/run_history.log