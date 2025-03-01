import re

from aviary.core import Message

from .prompts import OPEN_ENDED_GRADING_PROMPT


def grade_mcq_answer(target, predicted, unsure):
    predicted = predicted.upper()
    target = target.upper()
    unsure = unsure.upper()

    correct = predicted == target
    # checks if the predicted answer is the refusal option. (LLm is unsure)
    # Only for MCQ + w/resusal setting.Used to compute precision
    refusal = predicted == unsure

    grade = 1 if correct else 0
    return grade, correct, refusal


async def grade_open_ended_answer(question, target, predicted, llm_client):
    query = OPEN_ENDED_GRADING_PROMPT.format(
        question=question, target=target, predicted=predicted
    )

    completion = await llm_client.call_single([Message(content=query)])
    response = completion.model_dump()["text"]
    # parse response
    match = re.search(r"<grade>\s*(.*?)\s*</grade>", response, re.DOTALL)
    grade = match.group(1).strip().lower() if match else None

    if grade == "correct":
        grade = 1
        correct = True
        refusal = False

    if grade == "incorrect":
        grade = 0
        correct = False
        refusal = False

    if grade == "refused":
        grade = 0
        correct = False
        refusal = True

    if grade is None:
        grade = 0
        correct = False
        refusal = False
    return grade, correct, refusal


def compute_metrics(grades: list[bool], is_refued: list[bool]) -> dict:
    """Calculate metrics for question answering evaluation.

    Accuracy = (num correct) / (num questions)
    precision = (num correct) / ((num questions) - (num unsure)).
    """
    if len(grades) != len(is_refued):
        raise ValueError("is_correct and is_refued must have the same length")

    n_total = len(grades)
    n_correct = sum(grades)
    n_unsure = sum(1 for x in is_refued if x)
    n_sure = n_total - n_unsure
    # Calculate metrics
    accuracy = n_correct / n_total if n_total > 0 else 0
    precision = n_correct / n_sure if n_sure > 0 else 0
    coverage = n_sure / n_total if n_total > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "coverage": coverage,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_sure": n_sure,
    }
