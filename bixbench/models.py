from pathlib import Path
from typing import Any, Literal, Optional

from aviary.utils import EvalAnswerMode
from fhda import prompts
from fhda.utils import NBLanguage
from ldp.agent import AgentConfig
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentSettings(BaseModel):
    agent_type: str
    agent_kwargs: dict[str, Any]

    def construct_agent_config(self) -> AgentConfig:
        return AgentConfig(
            agent_type=self.agent_type,
            agent_kwargs=self.agent_kwargs,
        )


class RolloutSettings(BaseModel):
    max_steps: int
    batch_size: int
    rollout_type: str = "vanilla"

    @classmethod
    @field_validator("max_steps")
    def validate_max_steps(cls, v):
        if v <= 0:
            raise ValueError("max_steps must be greater than 0")
        return v

    @classmethod
    @field_validator("batch_size")
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be greater than 0")
        return v

    @classmethod
    @field_validator("rollout_type")
    def validate_rollout_type(cls, v):
        valid_types = ["vanilla", "custom", "aviary"]
        if v not in valid_types:
            raise ValueError(f"rollout_type must be one of {valid_types}")
        return v


class NotebookSettings(BaseModel):
    name: str
    language: NBLanguage

    @classmethod
    @field_validator("language")
    def validate_language(cls, v):
        if isinstance(v, NBLanguage):
            return v
        try:
            return NBLanguage[v.upper()]
        except KeyError as err:
            raise ValueError(
                f"Invalid language: {v}. Must be convertible to NBLanguage enum."
            ) from err


class PromptTemplates(BaseModel):
    mcq: str
    open: str
    hypothesis: str


class CapsuleSettings(BaseModel):
    mode: Literal["open", "mcq", "hypothesis"]
    include_refusal_option: bool
    system_prompt: str
    prompt_templates: PromptTemplates
    eval_mode: str | None = None
    avoid_images: bool

    @classmethod
    @field_validator("eval_mode")
    def validate_eval_mode(cls, v):
        if v is None or (isinstance(v, str) and v.lower() in {"none", "null", ""}):
            return None
        try:
            return EvalAnswerMode[v]
        except KeyError as err:
            raise ValueError(
                f"Invalid eval_mode: {v}. Must be convertible to EvalAnswerMode enum."
            ) from err


class PathSettings(BaseModel):
    workspace_dir: str
    trajectories_dir: str
    data_folder: str
    hf_repo_id: str

    def get_absolute_paths(self):
        return {
            "local_workspace_dir": Path(self.workspace_dir).absolute(),
            "local_trajectories_dir": Path(self.trajectories_dir).absolute(),
            "local_data_folder": Path(self.data_folder).absolute(),
        }


class PostProcessingSettings(BaseModel):
    total_questions: int
    total_iterations: int


class BixbenchConfig(BaseModel):
    run_name: str
    agent: AgentSettings
    rollout: RolloutSettings
    notebook: NotebookSettings
    capsule: CapsuleSettings
    paths: PathSettings
    postprocessing: Optional[PostProcessingSettings] = None

    # Computed fields that come from processing the raw config
    agent_config: Optional[AgentConfig] = None
    system_prompt: Optional[str] = None
    base_prompt: Optional[str] = None
    local_workspace_dir: Optional[Path] = None
    local_trajectories_dir: Optional[Path] = None
    local_data_folder: Optional[Path] = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def set_derived_fields(self):
        # Ensure eval_mode is properly set to None if it's "None"
        if isinstance(
            self.capsule.eval_mode, str
        ) and self.capsule.eval_mode.lower() in {"none", "null", ""}:
            self.capsule.eval_mode = None

        # Create AgentConfig
        self.agent_config = self.agent.construct_agent_config()

        # Get system prompt and base prompt
        self.system_prompt = getattr(prompts, self.capsule.system_prompt)
        capsule_mode = self.capsule.mode
        self.base_prompt = getattr(
            prompts, self.capsule.prompt_templates.model_dump()[capsule_mode]
        )
        if self.capsule.avoid_images:
            self.base_prompt += "\n" + prompts.AVOID_IMAGES

        # Set absolute path values
        path_dict = self.paths.get_absolute_paths()
        self.local_workspace_dir = path_dict["local_workspace_dir"] / self.run_name
        self.local_trajectories_dir = (
            path_dict["local_trajectories_dir"] / self.run_name
        )
        # We can share the data folder across runs
        self.local_data_folder = path_dict["local_data_folder"]

        return self


class PaperReplicationConfig(BaseModel):
    run: bool = False
    from_trajectories: bool = True


class MajorityVoteConfig(BaseModel):
    run: bool = False
    k_value: int = 10
    groups: dict[str, list[str]] = Field(default_factory=dict)


class RunComparisonConfig(BaseModel):
    run: bool = True
    # This is used to account for environment failures that don't always show up in the data
    total_questions_per_run: int = 296
    run_name_groups: list[list[str]] = Field(default_factory=list)
    group_titles: list[str] = Field(default_factory=list)
    color_groups: list[str] = Field(default_factory=list)
    use_zero_shot_baselines: bool = False
    random_baselines: list[Optional[float]] = Field(default_factory=list)
    baseline_name_mappings: dict[str, str] = Field(default_factory=dict)


class PostprocessingConfig(BaseModel):
    data_path: str = "data/trajectories/"
    results_dir: str = "bixbench_results"
    checkpointing: bool = False

    replicate_paper_results: PaperReplicationConfig = Field(
        default_factory=PaperReplicationConfig
    )
    majority_vote: MajorityVoteConfig = Field(default_factory=MajorityVoteConfig)
    run_comparison: RunComparisonConfig = Field(default_factory=RunComparisonConfig)
