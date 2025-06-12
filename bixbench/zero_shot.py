from functools import cached_property
from typing import Any, Optional

from aviary.core import Message
from lmi import LiteLLMModel
from pydantic import BaseModel, Field, model_validator

from .prompts import (
    MCQ_PROMPT_TEMPLATE_WITH_REFUSAL,
    MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL,
    OPEN_ENDED_PROMPT_TEMPLATE,
)
from .utils import AnswerMode, Query, parse_response, randomize_choices


class ZeroshotBaseline(BaseModel):

    answer_mode: AnswerMode
    with_refusal: bool
    model_name: str = Field(default="gpt-4o")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    _llm_client: Optional[LiteLLMModel] = None
    _query: Optional[Query] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @model_validator(mode="after")
    def initialize_llm_client(self) -> "ZeroshotBaseline":
        """Initialize the LLM client after model creation."""
        config = {
            "name": self.model_name,
            "temperature": self.temperature,
            **self.extra_kwargs,
        }
        self._llm_client = LiteLLMModel(name=self.model_name, config=config)
        return self

    @property
    def llm_client(self) -> LiteLLMModel:
        """Access the LLM client."""
        if self._llm_client is None:
            raise RuntimeError("LLM client not initialized")
        return self._llm_client

    @property
    def query(self) -> Query:
        """Access the current query."""
        if self._query is None:
            raise RuntimeError("No query has been set")
        return self._query

    @query.setter
    def query(self, value: Query) -> None:
        """Set the current query."""
        self._query = value

    @cached_property
    def prompt_template(self) -> str:
        """Get the appropriate prompt template based on evaluation mode and refusal setting."""
        if self.answer_mode == AnswerMode.mcq:
            return (
                MCQ_PROMPT_TEMPLATE_WITH_REFUSAL
                if self.with_refusal
                else MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL
            )
        if self.answer_mode == AnswerMode.openanswer:
            return OPEN_ENDED_PROMPT_TEMPLATE
        raise ValueError(f"Unknown answer mode: {self.answer_mode}")

    def _prep_query(self) -> tuple[str, Any, Optional[Any]]:
        """Generate query based on evaluation mode and parameters."""
        template = self.prompt_template

        if self.answer_mode == AnswerMode.mcq:
            distractors, target, unsure = randomize_choices(
                self.query.target, self.query.choices, with_refusal=self.with_refusal
            )
            prompted_question = template.format(
                question=self.query.question, options="\n".join(distractors)
            )
            return prompted_question, target, unsure

        if self.answer_mode == AnswerMode.openanswer:
            prompted_question = template.format(question=self.query.question)
            return prompted_question, self.query.target, None

        raise ValueError(f"Unknown answer mode: {self.answer_mode}")

    async def generate_zeroshot_answers(self, query: Query) -> Query:
        """Generate baseline textual answers. Supports MCQ and open-answer questions."""
        self.query = query
        prompted_question, target, unsure = self._prep_query()
        messages = [Message(content=prompted_question)]
        completion = await self.llm_client.call_single(messages)
        response = completion.model_dump()["text"]
        try:
            predicted_answer = parse_response(response, answer_mode=self.answer_mode)
        except Exception:
            predicted_answer = "failed"

        self.query.predicted = predicted_answer
        self.query.target = target
        self.query.unsure = unsure
        return self.query
