# BixBench Initial Experiment: Running Instructions

This document outlines how to run BixBench evaluations with GPT-4o, Claude 3.5, and Claude 3.7.

## Prerequisites

1. **API Keys**: Ensure you have valid API keys set up in your `.env` file:
   ```
   OPENAI_API_KEY=your-openai-key
   ANTHROPIC_API_KEY=your-anthropic-key
   ```

2. **Docker**: Ensure Docker is running, as BixBench runs evaluations in a containerized environment:
   ```bash
   # Verify Docker is running
   docker ps
   
   # Pull the required image if you haven't already
   docker pull futurehouse/bixbench:aviary-notebook-env
   ```

3. **Hugging Face Authentication**: Authenticate with Hugging Face to access the BixBench dataset:
   ```bash
   huggingface-cli login
   # Follow the prompts to provide your token
   ```

## Running the Experiment

### Automated Execution

The easiest approach is to use the shell script:

```bash
# Make sure the script is executable
chmod +x run_multi_model_eval.sh

# Run the entire experiment
./run_multi_model_eval.sh
```

When prompted, enter a name for this evaluation run (e.g., "baseline" or "temperature_0.7"). This name will be combined with the current date and time to create a unique run ID.

The script will:
1. Run trajectory generation for all models (both open-ended and MCQ)
2. Process the results
3. Generate visualizations comparing the models
4. Store results in a timestamped directory

### Manual Execution

If you prefer to run steps individually:

#### 1. Generate Open-Ended Question Trajectories

```bash
# GPT-4o
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/gpt4o_trajectories.yaml

# Claude 3.5
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/claude35_trajectories.yaml

# Claude 3.7
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/claude37_trajectories.yaml
```

#### 2. Generate Multiple-Choice Question Trajectories

```bash
# GPT-4o
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/gpt4o_mcq_trajectories.yaml

# Claude 3.5
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/claude35_mcq_trajectories.yaml

# Claude 3.7
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/claude37_mcq_trajectories.yaml
```

#### 3. Run Postprocessing for Both Question Types

```bash
# For open-ended questions
python bixbench/postprocessing.py --config_file bixbench/run_configuration/multi_model_postprocessing.yaml

# For MCQ
python bixbench/postprocessing.py --config_file bixbench/run_configuration/multi_model_mcq_postprocessing.yaml
```

## Managing and Comparing Results

### Listing Previous Runs

To view a list of all previous evaluation runs:

```bash
./list_runs.sh
```

This will show:
1. A history log of completed runs with timestamps
2. The available run directories in the filesystem

### Comparing Different Runs

To compare results from different evaluation runs:

```bash
# Compare all runs
python compare_runs.py

# Compare specific runs
python compare_runs.py --runs baseline_20240317_120000 modified_20240318_140000

# Compare only MCQ results
python compare_runs.py --type multi_model_mcq

# Save comparison plots to a directory
python compare_runs.py --output comparison_results
```

## Monitoring Progress

The trajectory generation can take significant time. You can monitor progress through:

1. **Terminal Output**: The script will show a progress bar
2. **Output Directories**: Check if files are being created in the trajectories directories

## Comparing to Original Paper Results

To compare your results with the original paper:

```bash
# Download the paper's data
wget https://storage.googleapis.com/bixbench-results/raw_trajectory_data.csv -P bixbench_results/
wget https://storage.googleapis.com/bixbench-results/eval_df.csv -P bixbench_results/

# Generate the paper's charts
python bixbench/postprocessing.py --config_file bixbench/run_configuration/bixbench_paper_results.yaml
```

## Troubleshooting

If you encounter issues:

1. **API Errors**: Check your API keys and rate limits
2. **Docker Issues**: Ensure Docker is running and has sufficient resources
3. **Out of Memory**: The process can be memory-intensive; try running models one at a time
4. **Missing Dependencies**: Run `pip install -e .` or `uv sync` again to ensure all dependencies are installed
5. **File Path Issues**: If you get errors about file paths, make sure you're running commands from the BixBench root directory

## Customizing Configurations

To modify model configurations:

1. Edit the YAML files in `bixbench/run_configuration/`
2. Key parameters to consider changing:
   - `temperature` - Adjust model temperature (higher = more creative/random)
   - `max_steps` - Maximum number of steps for each task
   - `batch_size` - Number of tasks to process in parallel
   - `include_refusal_option` - Whether to include a refusal option in MCQ tests