MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL = (
    "Extract the single letter answer to the following question from the given options. You must pick one answer even if you are unsure."  # noqa: E501
    "\n\nQuestion: {question}"
    "\n\nOptions:\n{options}"
    "IMPORTANT: You must only output a single letter answer in XML format."
    "\n\n Example Output: <answer> X </answer>"
)

MCQ_PROMPT_TEMPLATE_WITH_REFUSAL = (
    "Extract the single letter answer to the following question from the given options given below."
    "\n\nQuestion: {question}"
    "\n\nOptions:\n{options}"
    "IMPORTANT: You must only output a single letter answer in XML format."
    "\n\nExample Output: <answer> X </answer>"
)

OPEN_ENDED_PROMPT_TEMPLATE = (
    "Answer following question to the best of your knowledge."
    "Keep your answer concise and to the point."
    "\n\nQuestion: {question}"
    "IMPORTANT: You must only output your answer in XML format."
    "\n\nExample Output: <answer> Your answer </answer>"
)

OPEN_ENDED_GRADING_PROMPT = """You are given a question, target answer and a predicted answer. Your task is to compare the target answer with the predicted and assess if the predicted answer is correct, incorrect or it refused to answer.
Question: {question}
Target Answer: {target}
Predicted Answer: {predicted}

Important: You must only output one from `correct`, `incorrect` or `refused` between <grade> tags.
Example Output: <grade> correct </grade>
"""  # noqa: E501

MCQ_EVAL_PROMPT = """
First, carefully examine the following notebook:

<notebook>
{{notebook}}
</notebook>

Now, consider the following multiple-choice question:

<question>
{{question}}
</question>

For reference, this was an open response answer submitted to the question:

<proposed_answer>
{{proposed_answer}}
</proposed_answer>

You are allowed to use the proposed answer as a reference, but you don't have to use it when selecting your final answer.

Your goal is to select the best answer for the MCQ based on the information provided in the notebook and any associated images.
To ensure accuracy, please follow these steps:

1. Carefully read and analyze the content of the notebook.
2. Review any associated images mentioned in the notebook.
3. Identify the relevant information from the notebook and images.
4. Consider each answer option carefully.
5. Select the best answer based on the available information.

Before providing your final answer, wrap your analysis inside <question_analysis> tags:
1. Quote the most relevant parts of the notebook for answering the question.
2. List arguments for and against each answer option.
3. Conclude with your chosen answer and a brief explanation.

This process will help you arrive at the most accurate answer. It's OK for this section to be quite long.

After completing your analysis, provide your final answer in the exact format shown below:

<answer>A</answer>

Remember:
- Only include the corresponding letter answers in your final output.
- DO NOT PROVIDE ANY ADDITIONAL EXPLANATIONS OR TEXT OUTSIDE OF THE LETTER.

Please proceed with your analysis and answer selection.
"""  # noqa: E501

OPEN_ENDED_EVAL_PROMPT = """
Here is a question, the correct answer to the question, and a proposed answer.
Question: {question}
Correct answer: {correct_answer}
Proposed answer: {proposed_answer}
You must respond with a binary score (0 or 1) for whether the proposed answer is equivalent  to the correct answer.
\nNothing else is permitted.
"""
