# ❗❗ Under construction -- to be released to the public ❗❗
### Setp 1: Running BixBench zeroshot evaluations 
Let's say you want to run zeroshot evaluations for our `BixBench` dataset in `MCQ` setting and with the option to refuse to answer. Let's use the default LLM setting `gpt-4o`
This will write the evaluation results to a `csv` file inside the `results/` folder.
```
python run_zeroshot_evals.py --eval-mode mcq --with-refusal
```

You can also run as with open-answer setting. Let's see an example with  different LLM settings. eg:
```
python run_zeroshot_evals.py --eval-mode openanswer --model Claude 3.5 Sonnet --temperature 0.5
```

### Setp 2: Grading the responses 
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
