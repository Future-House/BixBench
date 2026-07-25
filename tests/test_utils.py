import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from bixbench.graders import GradeType, GradingFunction, MCQGrader, OpenEndedGrader
from bixbench.utils import (
    AnswerMode,
    compute_metrics,
    parse_response,
    randomize_choices,
)

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
        assert unsure is None


@pytest.mark.parametrize(
    ("text", "answer_mode", "tag", "expected"),
    [
        pytest.param(
            "This is a mock response. Answer is: <answer>A</answer>.",
            AnswerMode.mcq,
            "answer",
            "A",
            id="simple",
        ),
        pytest.param(
            "This is a mock response. Answer is: <answer> A </answer>.",
            AnswerMode.mcq,
            "answer",
            "A",
            id="with_spaces",
        ),
        pytest.param(
            "This is a mock response. Answer is: <answer>a </answer>.",
            AnswerMode.mcq,
            "answer",
            "A",
            id="lowercase",
        ),
        pytest.param(
            "This is a mock response. Answer is: <answer> Anything is okay </answer>",
            AnswerMode.openanswer,
            "answer",
            " Anything is okay ",
            id="openanswer",
        ),
        pytest.param(
            "This is a mock response. Answer is: <response>this is the response</response>.",
            AnswerMode.openanswer,
            "response",
            "this is the response",
            id="response tag",
        ),
    ],
)
def test_parse_response(text: str, answer_mode: AnswerMode, tag: str, expected: str):
    assert parse_response(text=text, tag=tag, answer_mode=answer_mode) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "predicted", "unsure", "expected_grade", "expected_refusal"),
    [
        pytest.param("A", "A", "B", 1, False, id="correct_and_sure"),
        pytest.param("A", "B", "B", 0, True, id="incorrect_and_unsure"),
    ],
)
async def test_grade_mcq_answer(
    target: str,
    predicted: str,
    unsure: str,
    expected_grade: int,
    expected_refusal: bool,
):
    mcq_grader = MCQGrader()
    grade_result = await mcq_grader.grade(
        target, predicted, unsure, evaluation_mode="str_verifier"
    )

    assert grade_result.grade == expected_grade
    assert grade_result.refusal == expected_refusal


@pytest.mark.parametrize(
    ("grader_response", "expected_grade_type"),
    [
        pytest.param("<grade>correct</grade>", GradeType.CORRECT, id="correct"),
        pytest.param("<grade>incorrect</grade>", GradeType.INCORRECT, id="incorrect"),
        pytest.param("<grade>refused</grade>", GradeType.REFUSED, id="refused"),
        # Both grading prompts advertise `<grade> correct </grade>` as the example
        # output, so padding and casing must not change the verdict.
        pytest.param(
            "<grade> correct </grade>", GradeType.CORRECT, id="correct_padded"
        ),
        pytest.param(
            "Reasoning...\n<grade>\n  REFUSED\n</grade>\n",
            GradeType.REFUSED,
            id="refused_uppercase_multiline",
        ),
        # A verdict we cannot read is a formatting failure by the grading model,
        # not an abstention by the answerer, so it stays incorrect.
        pytest.param(
            "The predicted answer is wrong.", GradeType.INCORRECT, id="missing_tag"
        ),
        pytest.param("<grade> refused", GradeType.INCORRECT, id="unclosed_tag"),
        pytest.param(
            "<grade> partially correct </grade>",
            GradeType.INCORRECT,
            id="unexpected_verdict",
        ),
    ],
)
def test_parse_grade_response(grader_response: str, expected_grade_type: GradeType):
    grade_type = GradingFunction()._parse_grade_response(grader_response)

    assert grade_type == expected_grade_type


def test_refusal_scores_zero_like_an_incorrect_answer():
    """Refusals score 0, so recording them cannot move accuracy or n_correct."""
    assert GradeType.CORRECT.numeric_grade == 1
    assert GradeType.INCORRECT.numeric_grade == 0
    assert GradeType.REFUSED.numeric_grade == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation_mode", ["llm_verifier", "range_verifier"])
@pytest.mark.parametrize(
    ("grader_response", "expected_grade", "expected_correct", "expected_refusal"),
    [
        pytest.param("<grade> correct </grade>", 1, True, False, id="correct"),
        pytest.param("<grade> incorrect </grade>", 0, False, False, id="incorrect"),
        pytest.param("<grade> refused </grade>", 0, False, True, id="refused"),
        pytest.param("no grade tag here", 0, False, False, id="missing_tag"),
    ],
)
async def test_open_ended_grading_records_refusal(
    evaluation_mode: str,
    grader_response: str,
    expected_grade: int,
    expected_correct: bool,
    expected_refusal: bool,
):
    """Both LLM grading paths must report a `refused` verdict as a refusal."""
    llm_client = AsyncMock()
    llm_client.call_single.return_value = MagicMock(text=grader_response)

    grade_result = await OpenEndedGrader(
        evaluation_mode=evaluation_mode, llm_client=llm_client
    ).grade(
        question="What is the capital of France?",
        target="Paris",
        predicted="I cannot answer this question.",
    )

    assert grade_result.grade == expected_grade
    assert grade_result.correct is expected_correct
    assert grade_result.refusal is expected_refusal


@pytest.mark.parametrize(
    ("grades", "is_refused", "metrics"),
    [
        pytest.param(
            [1, 1, 1, 1],
            [False, False, False, False],
            {
                "accuracy": 1,
                "precision": 1,
                "coverage": 1,
                "n_total": 4,
                "n_correct": 4,
                "n_sure": 4,
            },
            id="correct_and_sure",
        ),
        pytest.param(
            [1, 1, 1, 1],
            [True, True, True, True],
            {
                "accuracy": 1,
                "precision": 0,
                "coverage": 0,
                "n_total": 4,
                "n_correct": 4,
                "n_sure": 0,
            },
            id="correct_and_unsure",
        ),
        pytest.param(
            [0, 0, 0, 0],
            [False, False, False, False],
            {
                "accuracy": 0,
                "precision": 0,
                "coverage": 1,
                "n_total": 4,
                "n_correct": 0,
                "n_sure": 4,
            },
            id="incorrect_and_sure",
        ),
        pytest.param(
            [0, 0, 0, 0],
            [True, True, True, True],
            {
                "accuracy": 0,
                "precision": 0,
                "coverage": 0,
                "n_total": 4,
                "n_correct": 0,
                "n_sure": 0,
            },
            id="incorrect_and_unsure",
        ),
    ],
)
def test_compute_metrics(grades: list[bool], is_refused: list[bool], metrics: dict):
    result = compute_metrics(grades, is_refused)
    assert result == metrics
