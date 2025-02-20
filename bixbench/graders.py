from prompts import OPEN_ENDED_GRADING_PROMPT
import re

def grade_mcq_answer(target,predicted,unsure):  
    predicted = predicted.upper()
    target = target.upper()
    unsure = unsure.upper()

    correct = predicted == target
    sure = predicted != unsure

    if correct:
        grade = 1 
    else:
        grade = 0
    return grade, correct, sure

def grade_open_ended_answer(question, target, predicted,llm_client):
    query = OPEN_ENDED_GRADING_PROMPT.format(question=question, target=target, predicted=predicted)
    
    response,_,_ = llm_client.get_response(query=query)
    # parse response
    match = re.search(r'<grade>\s*(.*?)\s*</grade>', response , re.DOTALL)
    grade = match.group(1).strip().lower() if match else None
    if grade == "correct":
        grade = 1
        correct = True
    elif grade == "incorrect":
        grade = 0
        correct = False
    else:
        grade = None
    return grade, correct, True # always sure for open ended questions

def compute_metrics(df):
    n_total = len(df)
    n_correct = df['correct'].sum()
    n_sure = df['sure'].sum()
    
    # Calculate metrics
    accuracy = n_correct / n_total if n_total > 0 else 0
    precision = n_correct / n_sure if n_sure > 0 else 0
    coverage = n_sure / n_total if n_total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'coverage': coverage,
        'n_total': n_total,
        'n_correct': n_correct,
        'n_sure': n_sure
    }