import sys
from unittest.mock import MagicMock

import pytest

from bixbench.utils import AnswerMode, Query
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
        return Query(
            id="8204598f-b86a-4578-ab32-9d880c168718",
            question="What is the capital of France?",
            target="Paris",
            choices=["London", "Paris", "Berlin", "Madrid"],
        )

    @pytest.fixture
    def open_ended_input(self):
        return Query(
            id="8204598f-b86a-4578-ab32-9d880c168718",
            question="Explain how photosynthesis works.",
            target="Plants convert sunlight into energy through photosynthesis...",
            choices=[],
        )

    def test_init(self):
        """Test initialization of ZeroshotBaseline with various parameters."""
        # Default initialization
        baseline = ZeroshotBaseline(
            answer_mode=AnswerMode.mcq, with_refusal=True, model_name="gpt-4o"
        )
        assert baseline.answer_mode == AnswerMode.mcq
        assert baseline.with_refusal is True
        assert baseline.llm_client.config["name"] == "gpt-4o"
        assert baseline.llm_client.config["temperature"] == 1.0

        # Custom temperature and additional kwargs
        baseline = ZeroshotBaseline(
            answer_mode=AnswerMode.openanswer,
            with_refusal=False,
            model_name="claude-3-opus",
            temperature=0.7,
        )
        assert baseline.answer_mode == AnswerMode.openanswer
        assert baseline.with_refusal is False
        assert baseline.llm_client.config["name"] == "claude-3-opus"
        assert baseline.llm_client.config["temperature"] == 0.7

    def test_get_prompt_template(self):
        """Test the correct prompt template is returned based on mode and refusal setting."""
        # MCQ with refusal
        baseline = ZeroshotBaseline(answer_mode=AnswerMode.mcq, with_refusal=True)
        assert (
            baseline.prompt_template()
            == pytest.importorskip("bixbench.prompts").MCQ_PROMPT_TEMPLATE_WITH_REFUSAL
        )

        # MCQ without refusal
        baseline = ZeroshotBaseline(answer_mode=AnswerMode.mcq, with_refusal=False)
        assert (
            baseline.prompt_template()
            == pytest.importorskip(
                "bixbench.prompts"
            ).MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL
        )

        # Open-ended
        baseline = ZeroshotBaseline(
            answer_mode=AnswerMode.openanswer,
            with_refusal=True,  # This shouldn't matter for open-ended
        )
        assert (
            baseline.prompt_template()
            == pytest.importorskip("bixbench.prompts").OPEN_ENDED_PROMPT_TEMPLATE
        )


# TODO: ADD TESTS FOR GRADING, SETTING QUERY
