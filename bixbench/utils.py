import random
import string
import uuid
from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict


class EvalMode(StrEnum):
    mcq = auto()
    openanswer = auto()


class BaseModelWithID(BaseModel):
    def model_dump(self, **kwargs) -> dict:
        dump = super().model_dump(**kwargs)
        dump["id"] = str(dump["id"])
        return dump


class AgentInput(BaseModelWithID):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    id: uuid.UUID
    question: str
    target: str
    choices: list[str] | None = None


class LLMConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 1.0


def parse_response(
    text: str, tag: str = "answer", eval_mode: EvalMode = EvalMode.openanswer
) -> str:
    start = text.find(f"<{tag}>") + len(f"<{tag}>")
    end = text.find(f"</{tag}>")
    answer = text[start:end]
    if eval_mode == EvalMode.openanswer:
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
    return shuffled_choices, answer, "empty"
