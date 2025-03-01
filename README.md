<p align="center">
    <a href="https://arxiv.org/abs/">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-arXiv:2409.11363-b31b1b.svg">
    <a href = "https://github.com/Future-House/BixBench">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-181717.svg">
    <a href="https://huggingface.co/datasets/futurehouse/BixBench">
    <img alt="Dataset" src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg">
</p>

# BixBench: A Comprehensive Benchmark for LLM-based Agents in Computational Biology

BixBench is a benchmark designed to evaluate AI agents on real-world bioinformatics tasks.
This benchmark tests AI agents' ability to:

- Explore biological datasets
- Perform long, multi-step computational analyses
- Interpret nuanced results in the context of a research question

BixBench presents AI agents with open-ended or multiple-choice tasks, requiring them to navigate datasets, execute code (Python, R, Bash), generate scientific hypotheses, and validate them.
The dataset contains 296 questions derived from 53 real-world, published Jupyter notebooks and related data (capsules).

You can find the BixBench dataset in [Hugging Face](https://huggingface.co/datasets/futurehouse/BixBench), the paper [here](), and the blog post [here](https://futurehouse.org/blog/bixbench/).

This repository enables three separate functions:

1. Agentic evaluations of LLMs on BixBench
2. Zero-shot evaluations of LLMs on BixBench
3. Replicating the BixBench paper results

## Links

- [Installation](#installation)
- [Quick start](#quick-start)
- [Results](#latest-results)
- [Use your own agent](#use-your-own-agent)
- [Baselines](#baselines)
- [Ackowledge](#acknowledge)

## Installation

```bash
# Clone the repository
git clone https://github.com/Future-House/bixbench.git
cd bixbench

# Install dependencies
pip install -e .

# Authenticate with Hugging Face
huggingface-cli login

# Pull the docker for agentic evaluations
docker pull futurehouse/bixbench:aviary-notebook-env
```

## Prerequisites

### API Keys

We support all LLMs that are supported by [litellm](https://github.com/BerriAI/litellm). Create a `.env` file with the API keys for the LLMs you want to evaluate. For example:

```
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

## Agentic Evaluations

BixBench evaluates agents' ability to create complex Jupyter notebooks for real-world bioinformatics research questions. To evaluate an LLM on BixBench we separate the process into two steps:

1. Generate trajectories
2. Evaluate the trajectories via postprocessing

### Generate Trajectories

BixBench evaluates agents' ability to create complex Jupyter notebooks for real-world bioinformatics research questions. To generate these trajectories, it's as simple as configuring the `config.yaml` file and running the following command:

```bash
python bixbench/generate_trajectories.py
```

This will:

1. Download the BixBench dataset from Hugging Face (only needed once)
2. Preprocess each capsule in the dataset
3. Generate and store trajectories including the final agent answer and Jupyter notebook in the directory specified in `config.yaml`

Trajectories are saved in the `bixbench_results/` directory as json files.

### Customization

Edit `bixbench/config.yaml` to modify:

- Model configurations
- System prompts
- Batch sizes
- File paths
- Evaluation modes

## Using Your Own Agent

To use your own agent, use the `generate_trajectories.py` script to generate trajectories in the same format as the BixBench trajectories, then use the `postprocessing.py` script to evaluate your agent's performance.

### Evaluate trajectories

To evaluate the trajectories, we use the `postprocessing.py` script:

```bash
python bixbench/postprocessing.py --data_path bixbench_results/raw_trajectory_data.csv
```

This script will:

1. Load the raw trajectory data
2. Create an evaluation dataframe
3. Run majority vote analysis (for MCQ questions)
4. Compare model performance across different run groups defined in `config.py`
5. Generate visualizations

The script will save the evaluation dataframe as a CSV file in the `bixbench_results/` directory as well as the plots.

## Zero-shot Evaluations & Grading

You can run zero-shot evaluations using the `run_zeroshot_evals.py` script and then automatically grade the responses using the `grade_outputs.py` script. This code:

1. Loads the BixBench dataset from Hugging Face
2. Evaluates the LLM on the dataset, outputting a CSV file with the results
3. Grades the responses using LLM-based graders for open-ended answer or exact match for MCQs
4. Saves the final results as a JSON file

The scripts can be configured to run with open-ended questions, multiple-choice questions (with or without a refusal option), different models, and different temperatures. To explore the different options, run the scripts with the `--help` flag.

## Replicating the BixBench Paper Results

To replicate the BixBench paper results for agentic evaluations, you can download the raw data from 2,120 trajectories and its respective postprocessed evaluation dataframe:

```bash
wget https://storage.googleapis.com/bixbench-results/raw_trajectory_data.csv -P bixbench_results/
wget https://storage.googleapis.com/bixbench-results/eval_df.csv -P bixbench_results/
```

You can then run the postprocessing script to generate the evaluation dataframe and analysis plots using the `--checkpointing` flag to load the evaluation dataframe directly:

```bash
python bixbench/postprocessing.py --data_path bixbench_results/raw_trajectory_data.csv --checkpointing
```

## Acknowledge

BixBench is the product of a collaboration between [FutureHouse](https://futurehouse.org) and [ScienceMachine](https://www.sciencemachine.ai/).

