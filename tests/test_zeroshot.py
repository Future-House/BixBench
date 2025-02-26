from bixbench.utils import AgentInput, EvalMode
from bixbench.zero_shot import ZeroshotBaseline
import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.append("../")


class TestZeroshotBaseline:
    @pytest.fixture
    def mock_litellm_response(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "text": "This is a mock response. Answer is: <answer>A</answer>."}
        return mock_response

    @pytest.fixture
    def mcq_input(self):
        return AgentInput(
            id="8204598f-b86a-4578-ab32-9d880c168718",
            question="What is the capital of France?",
            target="Paris",
            choices=["London", "Paris", "Berlin", "Madrid"]
        )

    @pytest.fixture
    def open_ended_input(self):
        return AgentInput(
            id="8204598f-b86a-4578-ab32-9d880c168718",
            question="Explain how photosynthesis works.",
            target="Plants convert sunlight into energy through photosynthesis...",
            choices=[]
        )

    def test_init(self):
        """Test initialization of ZeroshotBaseline with various parameters"""
        # Default initialization
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.mcq,
            with_refusal=True,
            model_name="gpt-4o"
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
        """Test the correct prompt template is returned based on mode and refusal setting"""
        # MCQ with refusal
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.mcq,
            with_refusal=True
        )
        assert baseline._get_prompt_template() == pytest.importorskip(
            "bixbench.prompts").MCQ_PROMPT_TEMPLATE_WITH_REFUSAL

        # MCQ without refusal
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.mcq,
            with_refusal=False
        )
        assert baseline._get_prompt_template() == pytest.importorskip(
            "bixbench.prompts").MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL

        # Open-ended
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.openanswer,
            with_refusal=True  # This shouldn't matter for open-ended
        )
        assert baseline._get_prompt_template() == pytest.importorskip(
            "bixbench.prompts").OPEN_ENDED_PROMPT_TEMPLATE

    def test_prep_query_open_ended(self, open_ended_input):
        """Test query preparation for open-ended mode"""
        baseline = ZeroshotBaseline(
            eval_mode=EvalMode.openanswer,
            with_refusal=False  # Shouldn't matter for open-ended
        )
        baseline.input = open_ended_input

        query, target, unsure = baseline._prep_query()

        assert open_ended_input.question in query
        assert target == open_ended_input.target
        assert unsure == "empty"

    # todo: add test for prep_query for mcq mode
