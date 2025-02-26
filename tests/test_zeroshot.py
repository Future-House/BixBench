import sys
from unittest.mock import MagicMock

import pytest

from bixbench.utils import AgentInput, EvalMode, parse_response, randomize_choices
from bixbench.zero_shot import ZeroshotBaseline

sys.path.append("../")


class TestZeroshotBaseline:
    @pytest.fixture
    def mock_litellm_response(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "text": "This is a mock response. Answer is: <answer>A</answer>."
        }
        return mock_response

    @pytest.fixture
    def mcq_input(self):
        return AgentInput(
            id="8204598f-b86a-4578-ab32-9d880c168718",
            question="What is the capital of France?",
            target="Paris",
            choices=["London", "Paris", "Berlin", "Madrid"],
        )

    @pytest.fixture
    def open_ended_input(self):
        return AgentInput(
            id="8204598f-b86a-4578-ab32-9d880c168718",
            question="Explain how photosynthesis works.",
            target="Plants convert sunlight into energy through photosynthesis...",
            choices=[],
        )

    def test_init(self):
        """Test initialization of ZeroshotBaseline with various parameters."""
        # Default initialization
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.mcq, with_refusal=True, model_name="gpt-4o"
        )
        assert baseline.eval_mode == EvalMode.mcq
        assert baseline.with_refusal is True
        assert baseline.llm_client.config["name"] == "gpt-4o"
        assert baseline.llm_client.config["temperature"] == 1.0

        # Custom temperature and additional kwargs
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.openanswer,
            with_refusal=False,
            model_name="claude-3-opus",
            temperature=0.7,
        )
        assert baseline.eval_mode == EvalMode.openanswer
        assert baseline.with_refusal is False
        assert baseline.llm_client.config["name"] == "claude-3-opus"
        assert baseline.llm_client.config["temperature"] == 0.7

    def test_get_prompt_template(self):
        """Test the correct prompt template is returned based on mode and refusal setting."""
        # MCQ with refusal
        baseline = ZeroshotBaseline(eval_mode=EvalMode.mcq, with_refusal=True)
        assert (
            baseline._get_prompt_template()
            == pytest.importorskip("bixbench.prompts").MCQ_PROMPT_TEMPLATE_WITH_REFUSAL
        )

        # MCQ without refusal
        baseline = ZeroshotBaseline(eval_mode=EvalMode.mcq, with_refusal=False)
        assert (
            baseline._get_prompt_template()
            == pytest.importorskip(
                "bixbench.prompts"
            ).MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL
        )

        # Open-ended
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.openanswer,
            with_refusal=True,  # This shouldn't matter for open-ended
        )
        assert (
            baseline._get_prompt_template()
            == pytest.importorskip("bixbench.prompts").OPEN_ENDED_PROMPT_TEMPLATE
        )

    def test_prep_query_open_ended(self, open_ended_input):
        """Test query preparation for open-ended mode."""
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.openanswer,
            with_refusal=False,  # Shouldn't matter for open-ended
        )
        baseline.input = open_ended_input

        query, target, unsure = baseline._prep_query()

        assert open_ended_input.question in query
        assert target == open_ended_input.target
        assert unsure == "empty"

    # TODO: add test for prep_query for mcq mode
# testing utils


@pytest.mark.parametrize(
    ("ideal", "distractors", "with_refusal", "expected_length"),
    [
        pytest.param("Paris", ["London", "Berlin",
                               "Madrid"], True, 5, id="with_refusal"),
        pytest.param("Paris", ["London", "Berlin",
                               "Madrid"], False, 4, id="without_refusal"),
    ]
)
def test_randomize_choices(ideal: str, distractors: list[str], with_refusal: bool, expected_length: int):

    shuffled_choices, answer, unsure = randomize_choices(
        ideal, distractors, with_refusal)
    print(shuffled_choices)
    # 3 distractors + target +/- refusal option
    assert len(shuffled_choices) == expected_length
    if with_refusal:
        assert any(
            "Insufficient information to answer the question" in choice for choice in shuffled_choices)
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
        )
    ],
)
def test_parse_response(text: str, eval_mode: EvalMode, tag: str, expected: str):
    assert parse_response(text=text, tag=tag, eval_mode=eval_mode) == expected
