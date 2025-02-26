# BixBench: A a Comprehensive Benchmark for LLM-based Agents in Computational Biology

BixBench is a benchmark for evaluating Large Language Models on bioinformatics tasks. This README explains how to run zero-shot evaluations, generate traces, and analyze results.

## Installation

```bash
# Clone the repository
git clone https://github.com/Future-House/bixbench.git
cd bixbench

# Install dependencies
pip install -e .
```


## Prerequisites

Set the following environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export HUGGING_FACE_TOKEN=your_hf_token
export USE_DOCKER=true  # if using Docker
export USE_R=true       # if using R
```


Authenticate with Hugging Face:

```bash
huggingface-cli login
```


## Running Zero-shot Evaluations

BixBench supports multiple evaluation modes and model configurations.

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


Results are saved to CSV files in the `bixbench_results/` directory.

## Grading Responses

After running evaluations, grade the model responses:

### For MCQ responses:

```bash
python grade_outputs.py --input-file bixbench_results/results_mcq_False_gpt-4o_1.0.csv --eval-mode mcq
```


### For open-ended responses:

```bash
python grade_outputs.py --input-file bixbench_results/results_openanswer_False_gpt-4o_1.0.csv --eval-mode openanswer --model "Claude 3.5 Sonnet"
```


## Generating Traces

In BixBench we evaluate the ability for agents to create complex Jupyter notebooks to answer real-world bioinformatics research questions. To generate these traces, you can use the `generate_traces.py` script:

```bash
python bixbench/generate_traces.py
```


This will:
1. Download the BixBench dataset from Hugging Face (only needed once)
2. Preprocess each capsule in the dataset
3. Generate and store traces for each problem including the final agent answer and jupyter notebook

Traces are saved in the directory specified in `config.yaml`.

## Running Post-processing and Analysis

Process raw traces to evaluate the performance of the agent:

```bash
python bixbench/postprocessing.py --data_path bixbench_results/raw_trajectory_data.csv
```


This will:
1. Load and process raw data
2. Create an evaluation dataframes
3. Run majority vote analysis (for MCQ questions)
4. Compare model performance across different configurations
5. Generate visualizations

Results are saved to the `bixbench_results/` directory.

### Key Outputs:

- `eval_loop_results.csv`: Raw evaluation results
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

## Advanced Usage

For more detailed control over the evaluation process, refer to the configuration options and command-line arguments in each script.

# Using your own agent

To use your own agent, you can use the `generate_traces.py` script to generate traces in the same format as the BixBench traces and then use the `postprocessing.py` script to evaluate the performance of your agent.






