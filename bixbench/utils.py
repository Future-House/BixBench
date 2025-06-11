import random
import string
import uuid
from enum import StrEnum, auto
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AnswerMode(StrEnum):
    mcq = auto()
    openanswer = auto()


class BaseModelWithID(BaseModel):
    def model_dump(self, **kwargs) -> dict:
        dump = super().model_dump(**kwargs)
        dump["id"] = str(dump["id"])
        return dump


class Query(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    id: uuid.UUID
    question: str
    target: str
    choices: list[str] | None = None
    predicted: str | None = None
    unsure: str | None = None
    evaluation_mode: Optional[str] = None


class LLMConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 1.0


def parse_response(
    text: str, tag: str = "answer", answer_mode: AnswerMode = AnswerMode.openanswer
) -> str:
    start = text.find(f"<{tag}>") + len(f"<{tag}>")
    end = text.find(f"</{tag}>")
    answer = text[start:end]
    if answer_mode == AnswerMode.openanswer:
        return answer
    return answer.strip().upper()


def randomize_choices(
    ideal: str, distractors: list[str], with_refusal: bool = True
) -> tuple[list[str], str, str]:
    REFUSE_CHOICE = "Insufficient information to answer the question"
    ALPHABET = string.ascii_uppercase
    choices = (
        [ideal, REFUSE_CHOICE, *distractors] if with_refusal else [ideal, *distractors]
    )
    n_choices = len(choices)
    assert n_choices <= len(ALPHABET), "Too many choices"

    perm = list(range(n_choices))
    random.shuffle(perm)
    shuffled_choices = [
        f"({letter}) {choices[sigma_i]}"
        for letter, sigma_i in zip(ALPHABET, perm, strict=False)
    ]

    answer = ALPHABET[perm.index(0)]  # one letter - answer
    unsure = ALPHABET[perm.index(1)]  # one letter - unsure answer

    if with_refusal:
        return shuffled_choices, answer, unsure
    return shuffled_choices, answer, None


def compute_metrics(grades: list[bool], is_refused: list[bool]) -> dict:
    """Calculate metrics for question answering evaluation.

    Accuracy = (num correct) / (num questions)
    precision = (num correct) / ((num questions) - (num unsure)).
    """
    if len(grades) != len(is_refused):
        raise ValueError("is_correct and is_refused must have the same length")

    n_total = len(grades)
    n_correct = sum(grades)
    n_unsure = sum(1 for x in is_refused if x)
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
