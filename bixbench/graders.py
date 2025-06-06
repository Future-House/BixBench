from typing import Tuple, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import re
import ast

from aviary.core import Message
from .prompts import OPEN_ENDED_GRADING_PROMPT
from .utils import AnswerMode


class GradeType(str, Enum):
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
    def is_refused(self) -> bool:
        """Check if grade represents a refusal."""
        return self == GradeType.REFUSED


class GradeResult(BaseModel):
    """Result of grading an answer."""

    grade: int = Field(ge=0, le=1, description="Numeric grade (0 or 1)")
    correct: bool = Field(description="Whether the answer is correct")
    refusal: bool = Field(description="Whether the answer is a refusal")
    grade_type: Optional[GradeType] = Field(default=None, description="Type of grade")
    raw_response: Optional[str] = Field(
        default=None, description="Raw LLM response for open-ended"
    )

    def to_dict(self) -> dict:
        return {"grade": self.grade, "correct": self.correct, "refusal": self.refusal}


class MCQGrader(BaseModel):
    """Grader for multiple choice questions."""

    case_sensitive: bool = Field(
        default=False, description="Whether grading is case sensitive"
    )

    def grade(
        self, target: str, predicted: str, unsure: Optional[str] = None
    ) -> GradeResult:

        if not self.case_sensitive:
            predicted = predicted.upper().strip()
            target = target.upper().strip()
            unsure = unsure.upper().strip() if unsure else None

        correct = predicted == target
        refusal = predicted == unsure if unsure else False

        # Determine grade type
        if correct:
            grade_type = GradeType.CORRECT
        elif refusal:
            grade_type = GradeType.REFUSED
        else:
            grade_type = GradeType.INCORRECT

        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,
            grade_type=grade_type,
        )


class OpenEndedGrader(BaseModel):
    """Grader for open-ended questions."""

    evaluation_mode: Literal["llm_verifier", "str_verifier", "range_verifier"] = Field(
        default="llm", description="Evaluation mode for open-ended answers"
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
    def validate_llm_client(self) -> "OpenEndedGrader":
        """Ensure llm_client is provided when using llm_verifier mode."""
        if self.evaluation_mode == "llm_verifier" and not self.llm_client:
            raise ValueError("llm_client is required when using llm_verifier mode")
        return self

    async def grade(self, question: str, target: str, predicted: str) -> GradeResult:
        """
        Grade an answer based on the configured evaluation mode.

        Args:
            question: The question that was asked
            target: The expected answer
            predicted: The predicted answer

        Returns:
            GradeResult with grading information
        """
        if self.evaluation_mode == "str_verifier":
            return self._grade_str_verifier(question, target, predicted)
        elif self.evaluation_mode == "range_verifier":
            return self._grade_range_verifier(target, predicted)
        elif self.evaluation_mode == "llm_verifier":
            return await self._grade_llm_verifier(question, target, predicted)
        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

    def _grade_str_verifier(self, question, target, predicted) -> GradeResult:
        # normalize the target and predicted answers
        # EXACT match grading
        cleaned_target = re.sub(r"[^a-zA-Z0-9]", "", target).lower()
        cleaned_predicted = re.sub(r"[^a-zA-Z0-9]", "", predicted).lower()

        correct = cleaned_predicted == cleaned_target
        if not correct:
            # PARTIAL match grading
            correct = cleaned_predicted in cleaned_target
            if not correct:
                return self._grade_llm_verifier(question, target, predicted)

        grade_type = GradeType.CORRECT if correct else GradeType.INCORRECT
        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,  # this is only for mcq grading
            grade_type=grade_type,
        )

    async def _grade_llm_verifier(
        self,
        question,
        target,
        predicted,
    ) -> GradeResult:
        grading_query = self.grading_prompt_template.format(
            question=question, target=target, predicted=predicted
        )
        completion = await self.llm_client.call_single([Message(content=grading_query)])
        response = completion.model_dump()["text"]
        grade_type = self._parse_grade_response(response)

        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,
            grade_type=grade_type,
            raw_response=response,
        )

    def _grade_range_verifier(target, predicted) -> GradeResult:
        lower, upper = ast.literal_eval(target)
        correct = lower <= float(predicted) <= upper
        if correct:
            grade_type = GradeType.CORRECT
        else:
            grade_type = GradeType.INCORRECT
        return GradeResult(
            grade=grade_type.numeric_grade,
            correct=grade_type.is_correct,
            refusal=grade_type.is_refused,
            grade_type=grade_type,
        )

    def _parse_grade_response(self, response: str) -> GradeType:
        """Parse the grade from LLM response."""
        match = re.search(r"<grade>\s*(.*?)\s*</grade>", response, re.DOTALL)

        if not match:
            return GradeType.INCORRECT

        grade_text = match.group(1).strip().lower()

        try:
            return GradeType(grade_text)
        except ValueError:
            return GradeType.INCORRECT


class GradeAnswer(BaseModel):
    """Unified grader that handles both MCQ and open-ended questions."""

    answer_mode: AnswerMode
    question: str
    target: str
    predicted: str
    unsure: Optional[str] = None
    evaluation_mode: Optional[str] = None
    llm_client: Any = None

    class Config:
        arbitrary_types_allowed = True

    async def grade(self) -> Tuple[int, bool, bool]:
        if self.answer_mode == AnswerMode.mcq:
            mcq_grader = MCQGrader()
            result = mcq_grader.grade(self.target, self.predicted, self.unsure)
            return result.grade, result.correct, result.refusal

        elif self.answer_mode == AnswerMode.openanswer:
            open_ended_grader = OpenEndedGrader(
                evaluation_mode=self.evaluation_mode, llm_client=self.llm_client
            )
            result = await open_ended_grader.grade(
                self.question, self.target, self.predicted
            )
            return result.grade, result.correct, result.refusal

        else:
            raise ValueError(f"Unknown answer mode: {self.answer_mode}")
