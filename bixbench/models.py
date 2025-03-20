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
    
    # Optional support for new folder structure
    use_new_structure: bool = False
    run_id: str = None

    def get_absolute_paths(self):
        paths = {
            "local_workspace_dir": Path(self.workspace_dir).absolute(),
            "local_trajectories_dir": Path(self.trajectories_dir).absolute(),
            "local_data_folder": Path(self.data_folder).absolute(),
        }
        
        # If using new folder structure with a run_id, adjust paths
        if self.use_new_structure and self.run_id:
            # For backward compatibility, check if paths already include run_id
            # This is a heuristic - if trajectories_dir already has the run_id, we keep as is
            if self.run_id not in str(paths["local_trajectories_dir"]):
                paths["local_trajectories_dir"] = Path(f"runs/{self.run_id}/data/trajectories").absolute()
                
        return paths


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
    mini_mode: bool = False  # Flag for mini mode (runs only 10 examples)
    max_problems: int = None  # Limit number of problems for mini mode
    use_new_structure: bool = False  # Flag for using new folder structure

    # Computed fields that come from processing the raw config
    agent_config: Optional[AgentConfig] = None
    system_prompt: Optional[str] = None
    base_prompt: Optional[str] = None
    local_workspace_dir: Optional[Path] = None
    local_trajectories_dir: Optional[Path] = None
    local_data_folder: Optional[Path] = None
    run_id: Optional[str] = None  # Extracted run ID

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

        # Extract run_id from trajectories_dir path if it's in the new structure format
        trajectory_path = Path(self.paths.trajectories_dir)
        
        # Check if this looks like a path in the new structure (runs/run_id/data/trajectories)
        if "runs" in str(trajectory_path) and self.use_new_structure:
            parts = str(trajectory_path).split("/")
            for i, part in enumerate(parts):
                if part == "runs" and i + 1 < len(parts):
                    self.run_id = parts[i + 1]
                    break
        
        # If we couldn't extract a run_id but we want to use the new structure,
        # try to extract it from the model name
        if not self.run_id and self.use_new_structure:
            # Check if the model name contains a timestamp pattern which is typically part of run_id
            import re
            match = re.search(r'([a-zA-Z0-9_-]+_\d{8}_\d{6})', self.run_name)
            if match:
                self.run_id = match.group(1)
            else:
                # Fallback to using the run_name as run_id
                self.run_id = self.run_name
        
        # Update paths object with run_id and new structure flag
        if self.use_new_structure and self.run_id:
            self.paths.use_new_structure = True
            self.paths.run_id = self.run_id

        # Set absolute path values
        path_dict = self.paths.get_absolute_paths()
        
        # If using new structure, handle paths differently
        if self.use_new_structure and self.run_id:
            # workspace_dir remains standard for compatibility
            self.local_workspace_dir = path_dict["local_workspace_dir"] / self.run_name
            # trajectories_dir is already adjusted in PathSettings.get_absolute_paths
            self.local_trajectories_dir = path_dict["local_trajectories_dir"]
            # data_folder stays the same
            self.local_data_folder = path_dict["local_data_folder"]
        else:
            # Use original path handling for backward compatibility
            self.local_workspace_dir = path_dict["local_workspace_dir"] / self.run_name
            self.local_trajectories_dir = path_dict["local_trajectories_dir"] / self.run_name
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
    debug: bool = False
    mini_mode: bool = False  # Flag for mini mode (processes only 10 questions)
    use_new_structure: bool = False  # Flag for using new folder structure
    run_id: Optional[str] = None  # Used with new structure to identify the run

    replicate_paper_results: PaperReplicationConfig = Field(
        default_factory=PaperReplicationConfig
    )
    majority_vote: MajorityVoteConfig = Field(default_factory=MajorityVoteConfig)
    run_comparison: RunComparisonConfig = Field(default_factory=RunComparisonConfig)
    
    @model_validator(mode="after")
    def adjust_paths_for_new_structure(self):
        """Adjust paths if using the new folder structure"""
        if self.use_new_structure and self.run_id:
            # Check if paths already include the new structure
            if not self.data_path.startswith(f"runs/{self.run_id}/"):
                self.data_path = f"runs/{self.run_id}/data/trajectories"
            
            if not self.results_dir.startswith(f"runs/{self.run_id}/"):
                self.results_dir = f"runs/{self.run_id}/results"
        
        return self
