from pathlib import Path
from typing import Any, Literal

import fhda.config as cfg
from aviary.utils import EvalAnswerMode
from fhda import prompts
from fhda.utils import NBLanguage
from ldp.agent import AgentConfig
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


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
    skip_existing_trajectories: bool = True

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
    use_docker: bool = True

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

    @model_validator(mode="after")
    def validate_use_docker(self):
        if self.use_docker:
            cfg.USE_DOCKER = True
        return self


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
    postprocessing: PostProcessingSettings | None = None

    # Computed fields that come from processing the raw config
    system_prompt: str | None = None
    dataset_split: str = "train"

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def local_workspace_dir(self) -> Path:
        return Path(self.paths.workspace_dir).absolute()

    @computed_field
    @property
    def local_trajectories_dir(self) -> Path:
        return Path(self.paths.trajectories_dir).absolute() / self.run_name

    @computed_field
    @property
    def local_data_folder(self) -> Path:
        return Path(self.paths.data_folder).absolute()

    @computed_field
    @property
    def base_prompt(self) -> str:
        prompt = getattr(
            prompts, self.capsule.prompt_templates.model_dump()[self.capsule.mode]
        )
        if self.capsule.avoid_images:
            prompt += "\n" + prompts.AVOID_IMAGES
        return prompt

    @computed_field
    @property
    def agent_config(self) -> AgentConfig:
        return self.agent.construct_agent_config()

    @model_validator(mode="after")
    def set_derived_fields(self):
        # Ensure eval_mode is properly set to None if it's "None"
        if isinstance(
            self.capsule.eval_mode, str
        ) and self.capsule.eval_mode.lower() in {"none", "null", ""}:
            self.capsule.eval_mode = None

        # Get system prompt and base prompt
        self.system_prompt = getattr(prompts, self.capsule.system_prompt)

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
    total_questions_per_run: int | None = None
    run_name_groups: list[list[str]] = Field(default_factory=list)
    group_titles: list[str] = Field(default_factory=list)
    color_groups: list[str] = Field(default_factory=list)
    use_zero_shot_baselines: bool = False
    random_baselines: list[float | None] = Field(default_factory=list)
    baseline_name_mappings: dict[str, str] = Field(default_factory=dict)


class PostprocessingConfig(BaseModel):
    data_path: str = "data/trajectories/"
    results_dir: str = "bixbench_results"
    eval_df_filename: str = "eval_df.csv"
    debug: bool = False

    replicate_paper_results: PaperReplicationConfig = Field(
        default_factory=PaperReplicationConfig
    )
    majority_vote: MajorityVoteConfig = Field(default_factory=MajorityVoteConfig)
    run_comparison: RunComparisonConfig = Field(default_factory=RunComparisonConfig)

    @computed_field
    @property
    def eval_df_path(self) -> Path:
        return Path(self.results_dir) / self.eval_df_filename
