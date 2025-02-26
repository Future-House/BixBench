# ❗❗ Under construction -- to be released to the public ❗❗

## BixBench Overview

BixBench is a benchmark designed to evaluate AI agents on real-world bioinformatics tasks. 
This benchmark is designed to test AI agents' ability to:
- Explore and analyze diverse datasets
- Perform multi-step computational analyses
- Interpret results in the context of the research question

BixBench presents AI agents with open-ended or multiple-choice type tasks, requiring them to navigate datasets, execute code (Python, R, Bash), and generate scientific hypotheses and validate them. 
This dataset has 296 questions derived from real-world, published 53 jupyter notebooks and related data (capsules).

You can find the BixBench  dataset in [Hugging Face]() and the paper [here]().  

### Step 1: Create a .env file
Before you run the evaluations you must create a .env file with your api keys.

eg setup your Hugging Face API token in the `.env` file as follows. See https://huggingface.co/settings/tokens for more details.
```
HF_TOKEN = "your-hf-token"
```

### Setp 2: Running BixBench zeroshot evaluations 
You can run zeroshot evaluations for our to answer `BixBench` dataset using the `run_zeroshot_evals.py` script. This code will automatically load our `BixBench` dataset from Hugging Face. Therefore, make sure you have followed step 1.

This script runs for 2 task types:
1. Multiple-choice question (MCQ) type
2. Open-ended question type

Additionally, you can evaluate LLMs performance by prompting it to refuse to answer if sufficient information is available. In this case, the `with-refusal` flag will add the choice "Insufficient information to answer the question" to the list of choices. The refusal option is NOT set by default.

Let's look at an example. Say you want to run zeroshot evaluations for our `BixBench` dataset in `MCQ` setting and with the option to refuse.
Let's use the default LLM setting `gpt-4o`. 
This will write the evaluation results to a `csv` file inside the `results/` folder.
```
python run_zeroshot_evals.py --eval-mode mcq --with-refusal
```

You can also run as with open-answer setting. Let's see an example with  different LLM settings. eg:
```
python run_zeroshot_evals.py --eval-mode openanswer --model Claude 3.5 Sonnet --temperature 0.5
```



### Setp 3: Grading the responses 
After running the evaluation (getting model performances) you may want to grade their performances. For this purpose you can use the `grade_output.py` script.
You have to provide the path to the input file generated from step 1.

Example usage:
```
python grade_outputs.py --input-file results/results_mcq_False_gpt-4o_1.0.csv --eval-mode mcq
```

If you have open-ended answers, you can use an LLM grader. In this case you have to specify the model name and other paramters as you wish. By default, our script will use `gpt-4o` at `temp=1.0`
```
python grade_outputs.py --input-file results/results_openanswer_False_gpt-4o_1.0.csv --eval-mode openanswer --model Claude 3.5 Sonnet
```
