import ast
import re
from enum import StrEnum
from typing import Any, Literal, Self

from aviary.core import Message
from lmi import LiteLLMModel
from pydantic import BaseModel, Field, field_validator, model_validator

from .prompts import OPEN_ENDED_GRADING_PROMPT, OPEN_ENDED_RANGE_GRADING_PROMPT
from .utils import AnswerMode


class GradeType(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    REFUSED = "refused"

    @property
    def numeric_grade(self) -> int:
        """Convert grade to numeric value."""
        return 1 if self == GradeType.CORRECT else 0

    @property
    def is_correct(self) -> bool:
        """Check if grade represents a correct answer."""
        return self == GradeType.CORRECT

    @property
    def is_incorrect(self) -> bool:
        """Check if grade represents a correct answer."""
        return self == GradeType.INCORRECT

    @property
    def is_refused(self) -> bool:
        """Check if grade represents a refusal."""
        return self == GradeType.REFUSED


class GradeResult(BaseModel):
    """Result of grading an answer."""

    grade: int = Field(ge=0, le=1, description="Numeric grade (0 or 1)")
    correct: bool = Field(description="Whether the answer is correct")
    refusal: bool = Field(description="Whether the answer is a refusal")
    grade_type: GradeType | None = Field(default=None, description="Type of grade")
    raw_response: str | None = Field(
        default=None, description="Raw LLM response for open-ended"
    )

    def to_dict(self) -> dict:
        return {"grade": self.grade, "correct": self.correct, "refusal": self.refusal}


class GradingFunction(BaseModel):
    """Base class for grading functions."""

    def _parse_grade_response(self, response: str) -> GradeType:
        """Parse the grade from LLM response."""
        match = re.search(r"<grade>\s*(.*?)\s*</grade>", response, re.DOTALL)
        grade = match[1].strip().lower() if match else None

        return GradeType.CORRECT if grade == "correct" else GradeType.INCORRECT

    async def _grade_str_verifier(
        self,
        target,
        predicted,
        unsure=None,
        question=None,
        partial_match=False,
        llm_match=False,
        llm_client=None,
        grading_prompt_template=OPEN_ENDED_GRADING_PROMPT,
    ) -> GradeResult:
        # normalize the target and predicted answers
        # EXACT match grading
        cleaned_target = re.sub(r"[^a-zA-Z0-9]", "", target).lower()
        cleaned_predicted = re.sub(r"[^a-zA-Z0-9]", "", predicted).lower()

        correct = cleaned_predicted == cleaned_target
        refusal = predicted == unsure if unsure else False
        if not correct and partial_match:
            # PARTIAL match grading
            correct = cleaned_predicted in cleaned_target
            if not correct and llm_match:
                # LLM verifier grading
                return await self._grade_llm_verifier(
                    question=question,
                    target=target,
                    predicted=predicted,
                    llm_client=llm_client,
                    grading_prompt_template=grading_prompt_template,
                )

        if correct:
            grade_type = GradeType.CORRECT
        elif refusal:
            grade_type = GradeType.REFUSED
        else:
            grade_type = GradeType.INCORRECT

        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,  # this is only for mcq grading
            grade_type=grade_type,
        )

    async def _grade_llm_verifier(
        self,
        question: str,
        target: str,
        predicted: str,
        llm_client: LiteLLMModel,
        grading_prompt_template=OPEN_ENDED_GRADING_PROMPT,
    ) -> GradeResult:
        grading_query = grading_prompt_template.format(
            question=question, target=target, predicted=predicted
        )
        completion = await llm_client.call_single([Message(content=grading_query)])
        response = completion.text or ""
        grade_type = self._parse_grade_response(response)

        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,
            grade_type=grade_type,
            raw_response=response,
        )

    async def _grade_range_llm_verifier(
        self,
        question,
        target: str,
        predicted: str,
        llm_client: LiteLLMModel,
        grading_prompt_template=OPEN_ENDED_RANGE_GRADING_PROMPT,
    ) -> GradeResult:
        grading_query = grading_prompt_template.format(
            question=question, target=target, predicted=predicted
        )
        completion = await llm_client.call_single([Message(content=grading_query)])
        response = completion.text or ""
        grade_type = self._parse_grade_response(response)

        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,
            grade_type=grade_type,
        )

    def _grade_range_verifier(
        self,
        target: str,
        predicted: str,
    ) -> GradeResult:
        lower, upper = ast.literal_eval(target)
        correct = lower <= float(predicted) <= upper
        if correct:
            grade_type = GradeType.CORRECT

        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,
            grade_type=grade_type,
        )


class MCQGrader(BaseModel):
    """Grader for multiple choice questions."""

    case_sensitive: bool = Field(
        default=False, description="Whether grading is case sensitive"
    )

    class Config:
        arbitrary_types_allowed = True

    async def grade(
        self,
        target: str,
        predicted: str,
        unsure: str | None = None,
        evaluation_mode: Literal[
            "str_verifier", "range_verifier", "llm_verifier"
        ] = "str_verifier",
    ) -> GradeResult:
        grading_func = GradingFunction()
        if evaluation_mode == "str_verifier":
            return await grading_func._grade_str_verifier(
                target=target,
                predicted=predicted,
                unsure=unsure,
                partial_match=False,
                llm_match=False,
            )
        if evaluation_mode == "range_verifier":
            return grading_func._grade_range_verifier(
                target=target, predicted=predicted
            )

        raise ValueError(f"Unknown eval_mode: {evaluation_mode}")


class OpenEndedGrader(BaseModel):
    """Grader for open-ended questions."""

    evaluation_mode: Literal["llm_verifier", "str_verifier", "range_verifier"] = Field(
        default="llm_verifier", description="Evaluation mode for open-ended answers"
    )
    llm_client: Any = Field(description="LLM client for grading")
    grading_prompt_template: str = Field(
        default=OPEN_ENDED_GRADING_PROMPT, description="Template for grading prompt"
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator("evaluation_mode")
    @classmethod
    def validate_eval_mode(cls, v: str) -> str:
        """Validate evaluation_mode is one of the allowed values."""
        allowed_modes = {"llm_verifier", "str_verifier", "range_verifier"}
        if v not in allowed_modes:
            raise ValueError(f"evaluation_mode must be one of {allowed_modes}")
        return v

    @model_validator(mode="after")
    def validate_llm_client(self) -> Self:
        """Ensure llm_client is provided when using llm_verifier mode."""
        if self.evaluation_mode == "llm_verifier" and not self.llm_client:
            raise ValueError("llm_client is required when using llm_verifier mode")
        return self

    async def grade(
        self,
        question: str,
        target: str,
        predicted: str,
        partial_match: bool = True,
        llm_match: bool = True,
    ) -> GradeResult:
        """Grade an open-ended answer."""
        grading_func = GradingFunction()
        if self.evaluation_mode == "str_verifier":
            return await grading_func._grade_str_verifier(
                target=target,
                predicted=predicted,
                question=question,
                partial_match=partial_match,
                llm_match=llm_match,
                llm_client=self.llm_client,
                grading_prompt_template=self.grading_prompt_template,
            )

        if self.evaluation_mode == "range_verifier":
            return await grading_func._grade_range_llm_verifier(
                question=question,
                target=target,
                predicted=predicted,
                llm_client=self.llm_client,
                grading_prompt_template=self.grading_prompt_template,
            )

        if self.evaluation_mode == "llm_verifier":
            return await grading_func._grade_llm_verifier(
                question=question,
                target=target,
                predicted=predicted,
                llm_client=self.llm_client,
                grading_prompt_template=self.grading_prompt_template,
            )
        raise ValueError(f"Unknown eval_mode: {self.evaluation_mode}")


class GradeAnswer(BaseModel):
    """Unified grader that handles both MCQ and open-ended questions."""

    answer_mode: AnswerMode
    llm_client: Any = None

    class Config:
        arbitrary_types_allowed = True

    async def grade(
        self,
        target: str,
        predicted: str,
        question: str | None = None,
        unsure: str | None = None,
        evaluation_mode: Literal[
            "llm_verifier", "str_verifier", "range_verifier"
        ] = "str_verifier",
        partial_match: bool = False,
        llm_match: bool = False,
    ) -> tuple[int, bool, bool]:
        print(f"Grading: {question}, {target}, {predicted}")
        if self.answer_mode == AnswerMode.mcq:
            mcq_grader = MCQGrader()
            result = await mcq_grader.grade(
                target=target,
                predicted=predicted,
                unsure=unsure,
                evaluation_mode=evaluation_mode,
            )
            return result.grade, result.correct, result.refusal

        if self.answer_mode == AnswerMode.openanswer:
            assert question is not None
            open_ended_grader = OpenEndedGrader(
                evaluation_mode=evaluation_mode, llm_client=self.llm_client
            )
            result = await open_ended_grader.grade(
                question=question,
                target=target,
                predicted=predicted,
                partial_match=partial_match,
                llm_match=llm_match,
            )
            print("===============")
            print(f"Result: {result}")
            print("===============")
            return result.grade, result.correct, result.refusal

        raise ValueError(f"Unknown answer mode: {self.answer_mode}")
