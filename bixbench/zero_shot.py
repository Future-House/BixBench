from typing import Dict, Any
from .utils import randomize_choices, parse_response, EvalMode, AgentInput
from .prompts import (
    MCQ_PROMPT_TEMPLATE_WITH_REFUSAL,
    MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL,
    OPEN_ENDED_PROMPT_TEMPLATE,
)
from lmi import LiteLLMModel
from aviary.core import Message


class ZeroshotBaseline:
    def __init__(
        self,
        eval_mode: EvalMode,
        with_refusal: bool,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.eval_mode = eval_mode
        self.with_refusal = with_refusal
        self.llm_client = LiteLLMModel(
            name=f"{model_name}",
            config={"name": model_name, "temperature": temperature, **kwargs},
        )

    def _get_prompt_template(self) -> str:
        """Get the appropriate prompt template based on evaluation mode and refusal setting."""
        if self.eval_mode == EvalMode.mcq:
            return (
                MCQ_PROMPT_TEMPLATE_WITH_REFUSAL
                if self.with_refusal
                else MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL
            )
        elif self.eval_mode == EvalMode.openanswer:
            return OPEN_ENDED_PROMPT_TEMPLATE

    def _prep_query(self) -> str:
        """Generate query based on evaluation mode and parameters."""

        template = self._get_prompt_template()

        if self.eval_mode == EvalMode.mcq:
            distractors, target, unsure = randomize_choices(
                self.input.target, self.input.choices, with_refusal=self.with_refusal
            )
            return (
                template.format(
                    question=self.input.question, options="\n".join(distractors)
                ),
                target,
                unsure,
            )

        if self.eval_mode == EvalMode.openanswer:
            return (
                template.format(question=self.input.question),
                self.input.target,
                "empty",
            )

    async def generate_zeroshot_answers(
        self,
        input: AgentInput,
    ) -> list[str]:
        """Generate baseline textual answers. Supports MCQ and open-ended questions.
        This version doesn't parse images.
        """
        self.input = input
        query, target, unsure = self._prep_query()
        try:
            messages = [Message(content=query)]
            completion = await self.llm_client.call_single(messages)
            response = completion.model_dump()["text"]
            predicted_answer = parse_response(response, eval_mode=self.eval_mode)
        except Exception as e:
            print(f"Failed to get response because of {e}")
            predicted_answer = "failed"

        return predicted_answer, target, unsure
