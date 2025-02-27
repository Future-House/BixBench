<p align="center">
    <a href="https://arxiv.org/abs/">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-arXiv:2409.11363-b31b1b.svg">
    <a href = "https://github.com/Future-House/BixBench">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-181717.svg">
    <a href="https://huggingface.co/datasets/futurehouse/BixBench-internal">
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

You can find the BixBench dataset in [Hugging Face]() and the paper [here]().

## Installation

```bash
# Clone the repository
git clone https://github.com/Future-House/bixbench.git
cd bixbench

# Install dependencies
pip install -e .
```

## Prerequisites

### API Keys
Create a `.env` file with your API keys:

```
HF_TOKEN = "your-hf-token"
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

For more details on Hugging Face tokens, see https://huggingface.co/settings/tokens.

You can also set environment variables directly:

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export HUGGING_FACE_TOKEN=your_hf_token
```

Authenticate with Hugging Face:

```bash
huggingface-cli login
```

Install environment docker image:

```bash
docker pull futurehouse/bixbench:aviary-notebook-env
export NB_ENVIRONMENT_DOCKER_IMAGE=futurehouse/bixbench/aviary-notebook-env:latest
```

## Running Zero-shot Evaluations

You can run zero-shot evaluations using the `run_zeroshot_evals.py` script. This code automatically loads the BixBench dataset from Hugging Face.

The script supports two task types:
1. Multiple-choice question (MCQ) type
2. Open-ended question type

You can also evaluate LLMs with the option to refuse answering when information is insufficient. The `--with-refusal` flag adds "Insufficient information to answer the question" to the choices. This option is NOT enabled by default.

### MCQ (Multiple Choice Question) Mode

Run with refusal option:

```bash
python run_zeroshot_evals.py --eval-mode mcq --with-refusal
```

Run without refusal option:

```bash
python run_zeroshot_evals.py --eval-mode mcq
```

### Open-answer Mode

Run with a specific model:

```bash
python run_zeroshot_evals.py --eval-mode openanswer --model "Claude 3.5 Sonnet" --temperature 0.5
```

Evaluation results are saved as CSV files in the `bixbench_results/` directory.

## Grading Responses

After running evaluations, you can grade the model responses using the `grade_outputs.py` script.

### For MCQ responses:

```bash
python grade_outputs.py --input-file bixbench_results/results_mcq_False_gpt-4o_1.0.csv --eval-mode mcq
```

### For open-ended responses:

```bash
python grade_outputs.py --input-file bixbench_results/results_openanswer_False_gpt-4o_1.0.csv --eval-mode openanswer --model "Claude 3.5 Sonnet"
```

By default, the script uses `gpt-4o` at `temp=1.0` for grading open-ended responses.

## Generating Traces

BixBench evaluates agents' ability to create complex Jupyter notebooks for real-world bioinformatics research questions. To generate these traces:

```bash
python bixbench/generate_traces.py
```

This will:
1. Download the BixBench dataset from Hugging Face (only needed once)
2. Preprocess each capsule in the dataset
3. Generate and store traces including the final agent answer and Jupyter notebook

Traces are saved in the directory specified in `config.yaml`.

## Running Post-processing and Analysis

Process raw traces to evaluate agent performance:

```bash
python bixbench/postprocessing.py --data_path bixbench_results/raw_trajectory_data.csv
```

This will:
1. Load and process raw data
2. Create an evaluation dataframe
3. Run majority vote analysis (for MCQ questions)
4. Compare model performance across different configurations
5. Generate visualizations

Results are saved to the `bixbench_results/` directory.

### Key Outputs:

- `eval_df.csv`: Processed evaluation dataframe
- Visualization plots comparing model performance with/without vision capabilities
- Visualization plots comparing performance with/without refusal options

## Customization

Edit `bixbench/config.yaml` to modify:
- Model configurations
- System prompts
- Batch sizes
- File paths
- Evaluation modes

## Using Your Own Agent

To use your own agent, use the `generate_traces.py` script to generate traces in the same format as the BixBench traces, then use the `postprocessing.py` script to evaluate your agent's performance.

