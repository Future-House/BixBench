from llm_clients import LLMClient
from utils import randomize_choices, parse_response, EvalMode, AgentInput, LLMConfig
from prompts import (
    MCQ_PROMPT_TEMPLATE_WITH_REFUSAL,
    MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL,
    OPEN_ENDED_PROMPT_TEMPLATE,
)

class ZeroshotBaseline:
    def __init__(
        self,
        input: AgentInput,
        eval_mode: EvalMode,
        with_refusal: bool,
        llm_config: LLMConfig,
    ) -> None:
        self.input = input
        self.eval_mode = eval_mode
        self.with_refusal = with_refusal
        self.llm_config = llm_config

    @staticmethod
    def _get_prompt_template(self) -> str:
        """Get the appropriate prompt template based on evaluation mode and refusal setting."""
        if self.eval_mode == EvalMode.mcq:
            return (
                MCQ_PROMPT_TEMPLATE_WITH_REFUSAL
                if self.with_refusal
                else MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL
            )
        elif self.eval_mode == EvalMode.openended:
            return OPEN_ENDED_PROMPT_TEMPLATE

    def _prep_query(self) -> str:
        """Generate query based on evaluation mode and parameters."""

        template = self._get_prompt_template(self.eval_mode, self.with_refusal)

        if self.eval_mode == EvalMode.mcq:
            distractors, target, unsure = randomize_choices(
                self.input.target, self.input.choices, with_refusal=self.with_refusal
            )
            return template.format(
                question=self.input.question, options="\n".join(distractors)
            ), target, unsure

        if self.eval_mode == EvalMode.openended:
            return template.format(question=self.input.question), target, "empty"

    async def generate_zeroshot_answers(
        self,
    ) -> list[str]:
        """Generate baseline textual answers. Supports MCQ and open-ended questions.
        This version doesn't parse images.
        """

        llm_client = LLMClient(
            model_name=self.llm_config.model_name, temp=self.llm_config.temperature
        )

        query, target, unsure = self._prep_query()
        try:
            response = llm_client.get_response(query=query)
            predicted_answer = parse_response(response, eval_mode=self.eval_mode)
        except Exception as e:
            print(f"Failed to get response because of {e}")
            predicted_answer = "failed"

        return predicted_answer, target, unsure
