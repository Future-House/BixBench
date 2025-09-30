<p align="center">  <!-- markdownlint-disable MD041 -->
    <a href="https://arxiv.org/abs/2503.00096">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-arXiv:2409.11363-b31b1b.svg">
    <a href = "https://github.com/Future-House/BixBench">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-181717.svg">
    <a href="https://huggingface.co/datasets/futurehouse/BixBench">
    <img alt="Dataset" src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg">
    <img alt="Tests" src="https://github.com/Future-House/BixBench/actions/workflows/tests.yml/badge.svg">
    <a href="https://github.com/Future-House/BixBench/actions/workflows/tests.yml">
</p>

# BixBench: A Comprehensive Benchmark for LLM-based Agents in Computational Biology

BixBench is a benchmark designed to evaluate AI agents on real-world bioinformatics tasks.
This benchmark tests AI agents' ability to:

- Explore biological datasets
- Perform long, multi-step computational analyses
- Interpret nuanced results in the context of a research question

BixBench presents AI agents with open-ended or multiple-choice tasks, requiring them to navigate datasets, execute code (Python, R, Bash), generate scientific hypotheses, and validate them.
The dataset contains 205 questions derived from 60 real-world, published Jupyter notebooks and related data (capsules).

You can find the BixBench dataset on [Hugging Face](https://huggingface.co/datasets/futurehouse/BixBench), read details in the [paper](https://arxiv.org/abs/2503.00096), and read our [blog post announcement](https://www.futurehouse.org/research-announcements/bixbench).

This repository enables three separate functions:

1. Agentic evaluations of LLMs on BixBench
2. Zero-shot evaluations of LLMs on BixBench
3. Replicating the BixBench paper results

## Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agentic Evaluations](#agentic-evaluations)
- [Using Your Own Agent](#using-your-own-agent)
- [Zero-shot Evaluations & Grading](#zero-shot-evaluations--grading)
- [Replicating the BixBench Paper Results](#replicating-the-bixbench-paper-results)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the repository, first clone it and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/Future-House/BixBench.git
cd BixBench

# Install dependencies
pip install -e .  # or `uv sync` if you are using uv
```

Next, you will need to be able to access the BixBench dataset. To do this, you will have to authenticate with Hugging Face:

```bash
# Authenticate with Hugging Face
huggingface-cli login
```

See the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) for how to get started with the Hugging Face CLI and [Hugging Face Security Tokens](https://huggingface.co/docs/huggingface_hub/en/guides/security-tokens) for more information on how to create a token.

Finally, the agent executes its data analysis code in a containerized environment. So to run it, you will need to pull the docker image:

```bash
# Pull the docker image
docker pull futurehouse/bixbench:aviary-notebook-env
```

See [Docker's Getting Started Guide](https://docs.docker.com/get-started/) for instructions on how to set up Docker.

## Quick Start

For quick reproduction of BixBench results, we provide automated scripts that handle the entire evaluation pipeline:

### Option 1: Automated Evaluation (Recommended)

```bash
# Run zero-shot evaluations and grading (fastest)
bash scripts/run_zeroshot.sh

# Run agentic evaluations with multiple replicas (takes 24-48 hours)
bash scripts/run_agentic.sh
```

### Option 2: Manual Configuration

```bash
# Generate trajectories with specific configuration
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/4o_image.yaml

# Run postprocessing to generate results
python bixbench/postprocessing.py --config_file bixbench/run_configuration/v1.5_paper_results.yaml
```

⚠️ **Note**: The automated agentic evaluation script will run multiple models (GPT-4o, Claude) across 5 replicas each, which requires significant API credits and 24-48 hours to complete.

## Prerequisites

### API Keys

We support all LLMs that are supported by [litellm](https://github.com/BerriAI/litellm). Create a `.env` file with the API keys for the LLMs you want to evaluate. For example:

```env
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

## Agentic Evaluations

BixBench evaluates agents' ability to create complex Jupyter notebooks for real-world bioinformatics research questions. To evaluate an LLM on BixBench we separate the process into two steps:

1. Generate trajectories
2. Evaluate the trajectories via postprocessing

### Generate Trajectories

BixBench evaluates agents' ability to create complex Jupyter notebooks for real-world bioinformatics research questions. To generate these trajectories, it's as simple as configuring a YAML file and running the following command:

```bash
python bixbench/generate_trajectories.py --config_file bixbench/run_configuration/generate_trajectories.yaml
```

This will:

1. Download the BixBench dataset from Hugging Face (only needed once)
2. Preprocess each capsule in the dataset
3. Generate and store trajectories including the final agent answer and Jupyter notebook in the directory specified in the YAML file

Trajectories are saved in the specified `trajectories_dir` in the YAML file as json files (default is `data/trajectories/`).

### Customization

Edit or create a new YAML file to modify:

- Model configurations
- System prompts
- Batch sizes
- File paths
- Evaluation modes
- Rollout configuration

### Using Your Own Agent

To use your own agent, use the `generate_trajectories.py` script by editing the [`custom_rollout`](https://github.com/Future-House/BixBench/blob/6c28217959d5d7dd6f48c59894534fced7c6c040/bixbench/generate_trajectories.py#L239) function to generate trajectories in the same format as the BixBench trajectories, then use the `postprocessing.py` script to evaluate your agent's performance.

### Evaluate trajectories

Similarly, to evaluate the trajectories, we use the `postprocessing.py` script alongside a YAML configuration file:

```bash
python bixbench/postprocessing.py --config_file bixbench/run_configuration/postprocessing.yaml
```

This script will:

1. Load the raw trajectory data
2. Create an evaluation dataframe
3. Run majority vote analysis (for MCQ questions)
4. Compare model performance across different run groups defined in the YAML file
5. Generate visualizations

The script will save the evaluation dataframe as a CSV file in the `bixbench_results/` directory as well as the plots.

## Zero-shot Evaluations & Grading

You can run zero-shot evaluations using the `generate_zeroshot_evals.py` script and then grade the responses using the `grade_outputs.py` script. These two scripts:

1. Loads the BixBench dataset from Hugging Face
2. Evaluates the LLM on the dataset, outputting a CSV file with the results
3. Grades the responses using LLM-based graders for open-ended answer or exact match for MCQs
4. Saves the final results as a JSON file

The scripts can be configured to run with open-ended questions, multiple-choice questions (with or without a refusal option), different models, and different temperatures. To explore the different options, run the scripts with the `--help` flag.

### Example: Generate zero-shot answers in MCQ setting with the "refusal option" (in addition to the original distractors)

```bash
python generate_zeroshot_evals.py \
    --answer-mode "mcq" \
    --model "gpt-4o" \
    --with-refusal
```

### Example: Grade the zero-shot answers from the previous step

```bash
python grade_outputs.py \
    --input-file path/to/zeroshot.csv \
    --answer-mode "mcq"
```

## Replicating the BixBench Paper Results

### v1.5 Results (Latest)

For the latest BixBench v1.5 results using the enhanced 205-question dataset, use the automated scripts or the v1.5 configuration:

#### Quick Reproduction

```bash
# Complete reproduction using automation scripts
bash scripts/run_zeroshot.sh  # Zero-shot baselines
bash scripts/run_agentic.sh   # Agentic evaluations (24-48 hours)
```

#### Manual v1.5 Reproduction

```bash
# Generate v1.5 results
python bixbench/postprocessing.py --config_file bixbench/run_configuration/v1.5_paper_results.yaml
```

The v1.5 configuration includes:

- **Majority vote analysis** with k=5 replicas
- **Image comparison analysis** (with/without image support)
- **Refusal option comparison** (with/without refusal options in MCQs)
- **Zero-shot baseline integration** for comprehensive model comparison
- **Updated result paths** using `bixbench-v1.5_results/` directory structure

### Original Paper Results

To replicate the original BixBench paper results, you can download the raw data from 2,120 trajectories and its respective postprocessed evaluation dataframe:

```bash
wget https://storage.googleapis.com/bixbench-results/raw_trajectory_data.csv -P bixbench_results/
wget https://storage.googleapis.com/bixbench-results/eval_df.csv -P bixbench_results/
```

You can then run the postprocessing script to generate the evaluation dataframe and analysis plots using the original configuration file:

```bash
python bixbench/postprocessing.py --config_file bixbench/run_configuration/bixbench_paper_results.yaml
```

### Generated Figures

The evaluation process will generate the following comparative visualizations:

**v1.5 Results:**

You will see the following figures from the paper:
![Performance Comparison](bixbench-v1.5_results/bixbench_results_comparison.png)

![Majority Vote Accuracy](bixbench-v1.5_results/majority_vote_accuracy_refusal_option_comparison.png)

**Original Results:**

- `bixbench_results/bixbench_results_comparison.png` - Original performance comparison
- `bixbench_results/majority_vote_accuracy_refusal_option_comparison.png` - Original majority vote analysis

## Gotchas

- The BixBench dataset is large and may take several minutes to download.
- **API Costs**: The automated agentic evaluation script (`run_agentic.sh`) will incur significant API costs as it runs 5 replicas across multiple model configurations (GPT-4o, Claude-3.5-Sonnet). Estimate your costs before running.
- **Execution Time**: Complete agentic evaluations take 24-48 hours. Use the zero-shot script (`run_zeroshot.sh`) for faster results if you only need baseline comparisons.
- When generating trajectories manually, the default batch size is set to 4 to optimize processing speed. You may need to adjust this value in the [configuration file](https://github.com/Future-House/BixBench/blob/8c57d3562044e4ce574a09438066033e21155f54/bixbench/run_configuration/generate_trajectories.yaml#L14) based on your API rate limits and available compute resources.
- While the agent can use a local Jupyter kernel, we recommend using our custom Docker environment for improved performance (default is `USE_DOCKER=true`). Be sure to pull the Docker image as described in the [Installation](#installation) section. If you would like to use a local Jupyter kernel, set use_docker to false in the notebook section of the configuration file.
- **Directory Structure**: The v1.5 automation scripts create and use `bixbench-v1.5_results/` while manual configurations may use `bixbench_results/` or other directories as specified in the YAML files.

## Acknowledgments

BixBench is the product of a collaboration between [FutureHouse](https://futurehouse.org) and [ScienceMachine](https://www.sciencemachine.ai/).
