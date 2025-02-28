import sys

import pytest

from bixbench.graders import compute_metrics, grade_mcq_answer
from bixbench.utils import EvalMode, parse_response, randomize_choices

sys.path.append("../")


@pytest.mark.parametrize(
    ("ideal", "distractors", "with_refusal", "expected_length"),
    [
        pytest.param(
            "Paris", ["London", "Berlin", "Madrid"], True, 5, id="with_refusal"
        ),
        pytest.param(
            "Paris", ["London", "Berlin", "Madrid"], False, 4, id="without_refusal"
        ),
    ],
)
def test_randomize_choices(
    ideal: str, distractors: list[str], with_refusal: bool, expected_length: int
):
    shuffled_choices, answer, unsure = randomize_choices(
        ideal, distractors, with_refusal
    )
    # 3 distractors + target +/- refusal option
    assert len(shuffled_choices) == expected_length
    if with_refusal:
        assert any(
            "Insufficient information to answer the question" in choice
            for choice in shuffled_choices
        )
    else:
        assert unsure == "empty"


@pytest.mark.parametrize(
    ("text", "eval_mode", "tag", "expected"),
    [
        pytest.param(
            "This is a mock response. Answer is: <answer>A</answer>.",
            EvalMode.mcq,
            "answer",
            "A",
            id="simple",
        ),
        pytest.param(
            "This is a mock response. Answer is: <answer> A </answer>.",
            EvalMode.mcq,
            "answer",
            "A",
            id="with_spaces",
        ),
        pytest.param(
            "This is a mock response. Answer is: <answer>a </answer>.",
            EvalMode.mcq,
            "answer",
            "A",
            id="lowercase",
        ),
        pytest.param(
            "This is a mock response. Answer is: <answer> Anything is okay </answer>",
            EvalMode.openanswer,
            "answer",
            " Anything is okay ",
            id="openanswer",
        ),
        pytest.param(
            "This is a mock response. Answer is: <response>this is the response</response>.",
            EvalMode.openanswer,
            "response",
            "this is the response",
            id="response tag",
        ),
    ],
)
def test_parse_response(text: str, eval_mode: EvalMode, tag: str, expected: str):
    assert parse_response(text=text, tag=tag, eval_mode=eval_mode) == expected


@pytest.mark.parametrize(
    ("target", "predicted", "unsure", "expected_grade", "expected_refusal"),
    [
        pytest.param(
            "A", "A", "B", 1, False,
            id="correct_and_sure"
        ),
        pytest.param(
            "A", "B", "B", 0, True,
            id="incorrect_and_unsure"
        ),
    ],
)
def test_grade_mcq_answer(target: str, predicted: str, unsure: str, expected_grade: int, expected_refusal: bool):

    grade, _, refusal = grade_mcq_answer(target, predicted, unsure)

    assert grade == expected_grade
    assert refusal == expected_refusal


@pytest.mark.parametrize(
    ("grades", "is_refused", "metrics"),
    [
        pytest.param(
            [1, 1, 1, 1], [False, False, False, False], {"accuracy": 1, "precision": 1, "coverage": 1, "n_total": 4, "n_correct": 4, "n_sure": 4},
            id="correct_and_sure"
        ),
        pytest.param(
            [1, 1, 1, 1], [True, True, True, True], {"accuracy": 1, "precision": 0, "coverage": 0, "n_total": 4, "n_correct": 4, "n_sure": 0},
            id="correct_and_unsure"
        ),
        pytest.param(
            [0, 0, 0, 0], [False, False, False, False], {"accuracy": 0, "precision": 0, "coverage": 1, "n_total": 4, "n_correct": 0, "n_sure": 4},
            id="incorrect_and_sure"
        ),
        pytest.param(
            [0, 0, 0, 0], [True, True, True, True], {"accuracy": 0, "precision": 0, "coverage": 0, "n_total": 4, "n_correct": 0, "n_sure": 0},
            id="incorrect_and_unsure"
        ),
    ],
)
def test_compute_metrics(grades: list[bool], is_refused: list[bool], metrics: dict):
    result = compute_metrics(grades, is_refused)
    assert result == metrics
