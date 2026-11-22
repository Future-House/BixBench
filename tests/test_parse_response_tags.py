"""Tests for strict <tag>...</tag> extraction in bixbench.utils.parse_response."""

import pytest
from bixbench.utils import AnswerMode, parse_response


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Example Output: <answer> X </answer>\nFinal: <answer>b</answer>",
            "B",
        ),
        (
            "Reasoning with </answer> stray tag\n<answer> c </answer>",
            "C",
        ),
    ],
)
def test_parse_response_selects_last_complete_tag(text: str, expected: str) -> None:
    assert parse_response(text, answer_mode=AnswerMode.mcq) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Option A.",
        "I cannot answer this question.",
        "<answer>42.5",
        "42.5</answer>",
        "</answer> reasoning <answer>42.5",
    ],
)
def test_parse_response_raises_on_missing_or_unclosed_tags(text: str) -> None:
    with pytest.raises(ValueError, match="Missing complete <answer>"):
        parse_response(text, answer_mode=AnswerMode.openanswer)
