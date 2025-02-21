MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL = (
    "Extract the single letter answer to the following question from the given options. You must pick one answer even if you are unsure."
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

OPEN_ENDED_GRADING_PROMPT = """You are given a question, target answer and a predicted answer. Your task is to compare the target answer with the predicted and assess if the predicted answer is correct, incorrect or it refused or could not answer. 
Question: {question}
Target Answer: {target}
Predicted Answer: {predicted}

Important: You must only output one from `correct`, `incorrect` or `refused` between <grade> tags.
Example Output: <grade> correct </grade>
"""
