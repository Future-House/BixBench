# ❗❗ Under construction -- to be released to the public ❗❗
### Example run zeroshot evaluations with `gpt-4o`
```python
from bixbench import ZeroshotBaseline, AgentInput, EvalMode, LLMConfig

# Initialize the baseline agent
baseline_agent = ZeroshotBaseline(
    eval_mode=EvalMode.mcq,
    with_refusal=True,
    llm_config=LLMConfig(
        model_name="gpt-4",
        temperature=1.0
    )
)

# Example dataframe structure:
# df = pd.DataFrame({
#     'id': ['uuid1', 'uuid2'],
#     'question': ['q1', 'q2'],
#     'target': ['4', 'DNA'],
#     'choices': [['2', '3', '5'], ['RNA', 'mRNA', 'tRNA']]
# })

async def run_zeroshot_baseline(df):
    """
    Run zero-shot evaluation on a dataframe of questions.
    
    Args:
        df: pandas DataFrame with columns: id, question, target, choices
        
    Yields:
        tuple: (question_id, predicted_answer)
    """
    for idx, row in df.iterrows():
        input = AgentInput(
            id=row['id'],
            question=row['question'],
            target=row['target'],
            choices=row['choices']
        )
        answer, target, unsure_answer = await baseline_agent.generate_zeroshot_answers(input)
        yield id, answer, target, unsure_answer

# Usage example
async def main():
    results = []
    async for id, answer, target, unsure_answer in run_zeroshot_baseline(df):
        results.append({'id': id, 'answer': answer, 'target': target, 'unsure': unsure_answer})
    return results
```
