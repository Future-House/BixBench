import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import datasets
import yaml
from aviary.core import MultipleChoiceQuestion
from fhda.data_analysis_env import DataAnalysisEnv
from fhda.utils import collect_notebook_stats
from huggingface_hub import hf_hub_download
from ldp.agent import Agent
from ldp.alg.rollout import RolloutManager
from ldp.data_structures import Trajectory, Transition
from tqdm.auto import tqdm

from bixbench.models import BixbenchConfig
from bixbench.utils import as_completed_with_concurrency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Specifically silence certain loggers
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fhda.notebook_env").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "run_configuration" / "generate_trajectories.yaml"
)


class TrajectoryGenerator:
    """
    Generator for creating and storing agent trajectories on data analysis tasks.

    This class handles the full pipeline of loading benchmark capsules, setting up
    environments, running agents through these environments, and storing the resulting
    trajectories.
    """

    def __init__(
        self, config_path=DEFAULT_CONFIG_PATH, replica_id: int | None = None
    ) -> None:
        """
        Initialize the TrajectoryGenerator with config and create necessary directories.

        Args:
            config_path: Path to the configuration file
            replica_id: Replica ID
        """
        self.config = self.load_config(config_path)
        self.replica_id = replica_id
        # Create directories
        self.config.local_workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config.local_trajectories_dir.mkdir(parents=True, exist_ok=True)
        self.config.local_data_folder.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path) -> BixbenchConfig:
        """
        Load and process configuration from the provided path.

        Args:
            config_path: Path to the configuration file

        Returns:
            BixbenchConfig: Processed configuration object with validation
        """
        config_path = Path(config_path)
        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Create and validate the config using Pydantic
        return BixbenchConfig(**config_dict)

    async def process_capsule_data(self, zip_filename: str) -> None:
        """
        Process capsule data by downloading and extracting necessary files.

        Args:
            zip_filename: Name of the zip file to process
        """
        zip_path = self.config.local_data_folder / zip_filename
        extract_dir = self.config.local_data_folder / zip_filename.replace(".zip", "")
        # Check if capsule folder exists and is non-empty
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.debug(
                "Capsule folder %s already exists and is non-empty", extract_dir.name
            )
            return

        await asyncio.to_thread(
            hf_hub_download,
            repo_id=self.config.paths.hf_repo_id,
            filename=zip_filename,
            local_dir=self.config.local_data_folder,
            repo_type="dataset",
        )
        await asyncio.to_thread(self._extract_and_process_files, zip_path, extract_dir)

    async def process_question(self, question: dict[str, Any]) -> None:
        """
        Process a single benchmark question by downloading and extracting necessary files.

        Args:
            question: Dictionary containing question information
        """
        zip_filename: str = question["data_folder"]
        extract_dir = self.config.local_data_folder / zip_filename.replace(".zip", "")

        # Check if capsule folder exists and is non-empty
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.debug(
                "Question folder %s already exists and is non-empty", extract_dir.name
            )
            question["local_data_folder"] = extract_dir
            return

        # Download and process if not already present
        await self.process_capsule_data(zip_filename)

        question["local_data_folder"] = extract_dir

    async def load_bixbench(self) -> list[dict[str, Any]]:
        """
        Load BixBench dataset and process all questions.

        Returns:
            List[Dict[str, Any]]: List of processed benchmark questions
        """
        bixbench = datasets.load_dataset(self.config.paths.hf_repo_id, split=self.config.dataset_split).to_list()  # type: ignore[attr-defined]

        # Process all capsule data concurrently
        zip_filenames = {question["data_folder"] for question in bixbench}
        tasks = [self.process_capsule_data(zip_fname) for zip_fname in zip_filenames]
        await asyncio.gather(*tasks)

        # Prepare all questions concurrently
        tasks = [self.process_question(question) for question in bixbench]
        await asyncio.gather(*tasks)

        return bixbench

    def _extract_and_process_files(self, zip_path: Path, extract_dir: Path) -> None:
        """
        Extract and process zip files for a capsule.

        Args:
            zip_path: Path to the zip file
            extract_dir: Directory to extract files to
        """
        # Extract the zip file
        shutil.unpack_archive(zip_path, extract_dir)

        # Get the Data folder path
        data_folder = next(
            (p for p in extract_dir.rglob("*") if p.is_dir() and "Data" in p.name), None
        )
        if data_folder is None:
            raise FileNotFoundError(
                "Could not find a directory containing 'Data' in its name"
            )

        # Move contents of Data folder to parent directory
        for item in data_folder.iterdir():
            shutil.move(str(item), str(extract_dir / item.name))

        # Remove the Data folder
        shutil.rmtree(data_folder)

        # Safely remove Notebook folder if it exists
        try:
            notebook_folder = next(
                (
                    p
                    for p in extract_dir.rglob("*")
                    if p.is_dir() and "Notebook" in p.name
                ),
                None,
            )
            if notebook_folder is not None:
                shutil.rmtree(notebook_folder)
        except StopIteration:
            # No Notebook folder found, that's okay
            logger.debug("No Notebook folder found")

        # Remove any .ipynb files in the extract directory
        for ipynb_file in extract_dir.glob("*.ipynb"):
            ipynb_file.unlink()

        # Remove the zip file
        try:
            zip_path.unlink()
        except FileNotFoundError:
            logger.debug("Zip file not found")

    async def store_trajectory(
        self, trajectory: Trajectory, env: DataAnalysisEnv
    ) -> None:
        """
        Store trajectory and environment information to disk.

        Args:
            trajectory: The trajectory to store
            env: The environment that generated the trajectory
        """
        metadata = (
            {k: v for k, v in env.metadata.items() if k != "local_data_folder"}
            if env.metadata
            else {}
        )
        mcqs = metadata.pop("mcqs", [])
        mcq_options = mcqs[0].options if mcqs else []
        mcq_question = mcqs[0].question if mcqs else ""
        refusal_option = mcqs[0].unsure_answer_letter if mcqs else None

        extract = {
            "problem_id": env.problem_id,
            "agent_answer": env.state.answer,
            "ideal_answer": env.answer,
            "problem": env.problem,
            "mcq_options": mcq_options,
            "mcq_question": mcq_question,
            "notebook_stats": collect_notebook_stats(env.state.nb),
            "num_actions": len(env.state.actions),
            "question_format": self.config.capsule.mode,
            "refusal_option": self.config.capsule.include_refusal_option,
            "model": self.config.agent.agent_kwargs["llm_model"]["name"],
            # Local data folder is not serializable
            "metadata": metadata,
            "refusal_options": refusal_option,
            "nb": env.state.nb,
            "run_name": self.config.run_name,
        }

        # Store trajectory metadata
        filename = self.get_trajectory_path(env.problem_id)
        with filename.open("w") as f:
            json.dump(extract, f, indent=4)
        # Store trajectory
        await trajectory.to_jsonl(
            self.config.local_trajectories_dir
            / str(filename).replace(".json", ".jsonl")
        )

    def get_trajectory_path(self, problem_id: str) -> Path:
        """Get the path to the trajectory for a given problem ID.

        Args:
            problem_id: The problem ID

        Returns:
            Path: The path to the trajectory
        """
        if self.replica_id is not None:
            return (
                self.config.local_trajectories_dir
                / f"{problem_id}_replica_{self.replica_id}.json"
            )
        return self.config.local_trajectories_dir / f"{problem_id}.json"

    def environment_factory(self, question: dict[str, Any]) -> DataAnalysisEnv:
        """
        Create a DataAnalysisEnv instance from a question.

        Args:
            question: Dictionary containing question information

        Returns:
            DataAnalysisEnv: Initialized environment
        """
        processed_question = load_mcq(
            question, open_question=True, question_id=question["question_id"]
        )
        question["mcqs"] = [processed_question]

        language = self.config.notebook.language
        problem = self.config.base_prompt.format(
            question=question["question"], language=language
        )
        answer = question["ideal"]
        question_id = question["question_id"]
        if self.replica_id is not None:
            question_id = f"{question_id}_replica_{self.replica_id}"
        work_dir = (
            self.config.local_workspace_dir
            / self.config.run_name
            / question["capsule_uuid"]
            / question_id
        ).absolute()
        work_dir.mkdir(parents=True, exist_ok=True)
        local_capsule_data_path = self.config.local_data_folder / question[
            "data_folder"
        ].replace(".zip", "")

        # Copy all files from data folder to work directory
        for item in local_capsule_data_path.iterdir():
            if item.is_file():
                shutil.copy2(item, work_dir)
            elif item.is_dir():
                shutil.copytree(item, work_dir / item.name, dirs_exist_ok=True)
        nb_path = work_dir / self.config.notebook.name

        # Add some extra metadata from config
        question["avoid_images"] = self.config.capsule.avoid_images
        question["include_refusal_option"] = self.config.capsule.include_refusal_option

        env_args = {
            "problem_id": question["question_id"],
            "problem": problem,
            "eval_mode": self.config.capsule.eval_mode,
            "nb_path": nb_path,
            "work_dir": work_dir,
            "language": self.config.notebook.language,
            "system_prompt": self.config.system_prompt,
            "metadata": question,
            "answer": answer,
            "use_tmp_work_dir": False,
        }

        return DataAnalysisEnv(**env_args)

    async def custom_rollout(
        self, agent: Agent, environment: DataAnalysisEnv
    ) -> tuple[Trajectory, DataAnalysisEnv]:
        """
        Custom implementation of rollout logic.

        Args:
            agent: The agent to use for rollout
            environment: The environment to run the agent in

        Returns:
            Trajectory: The generated trajectory

        Raises:
            NotImplementedError: This method needs to be implemented by subclasses
        """
        raise NotImplementedError("Custom rollout not implemented")

    async def vanilla_rollout(
        self, agent: Agent, environment: DataAnalysisEnv
    ) -> tuple[Trajectory, DataAnalysisEnv]:
        """
        Standard implementation of rollout logic.

        Args:
            agent: The agent to use for rollout
            environment: The environment to run the agent in

        Returns:
            Tuple[Trajectory, DataAnalysisEnv]: The generated trajectory and updated environment
        """
        obs, tools = await environment.reset()
        agent_state = await agent.init_state(tools)
        trajectory = Trajectory()

        for timestep in range(self.config.rollout.max_steps):
            action, next_agent_state, value = await agent.get_asv(agent_state, obs)
            next_obs, reward, done, trunc = await environment.step(action.value)
            # Create the transition object
            transition = Transition(
                timestep=timestep,
                agent_state=agent_state,
                next_agent_state=next_agent_state,
                observation=obs,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                truncated=trunc,
                value=value,
            )
            # Update steps by creating a new list with the additional transition
            trajectory.steps = [*trajectory.steps, transition]
            if done or trunc:
                break

            agent_state = next_agent_state
            obs = next_obs

        return trajectory, environment

    async def batch_rollout(
        self, list_of_environments: list[DataAnalysisEnv]
    ) -> list[tuple[Trajectory, DataAnalysisEnv]]:
        """
        Run rollouts for a batch of environments.

        Args:
            list_of_environments: List of environments to run rollouts in

        Returns:
            List[Union[Trajectory, Tuple[Trajectory, DataAnalysisEnv]]]: List of trajectories or
                trajectory-environment pairs depending on rollout type
        """
        if self.config.rollout.rollout_type == "aviary":
            agent = self.config.agent_config.construct_agent()
            rollout = RolloutManager(agent=agent)
            trajectories = await rollout.sample_trajectories(
                environments=list_of_environments,
                max_steps=self.config.rollout.max_steps,
            )
            return list(zip(trajectories, list_of_environments, strict=True))

        agent = self.config.agent_config.construct_agent()
        rollout_manager = getattr(self, f"{self.config.rollout.rollout_type}_rollout")

        return await asyncio.gather(
            *[
                rollout_manager(agent, environment)
                for environment in list_of_environments
            ]
        )

    def filter_completed(self, bixbench: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out completed trajectories."""
        return [
            question
            for question in bixbench
            if not self.get_trajectory_path(question["question_id"]).exists()
        ]

    async def rollout_with_saving(
        self, environment: DataAnalysisEnv
    ) -> list[tuple[Trajectory, DataAnalysisEnv]]:
        trajectory, env = await self.vanilla_rollout(
            self.config.agent_config.construct_agent(), environment
        )
        await self.store_trajectory(trajectory, env)
        return trajectory, env

    async def run(self) -> None:
        """Run the full trajectory generation pipeline."""
        logger.info("Loading BixBench dataset...")
        bixbench = await self.load_bixbench()

        if self.config.rollout.skip_existing_trajectories:
            bixbench = self.filter_completed(bixbench)

        # Process environments in batches with tqdm progress bar
        with tqdm(total=len(bixbench), desc="Processing benchmark tasks") as pbar:
            for i in range(0, len(bixbench), self.config.rollout.batch_size):
                bsz = min(self.config.rollout.batch_size, len(bixbench) - i)
                batch = bixbench[i : i + (4 * bsz)]
                environments = (
                    self.environment_factory(question) for question in batch
                )
                rollouts = (self.rollout_with_saving(env) for env in environments)
                try:
                    async for _ in as_completed_with_concurrency(
                        rollouts, bsz, timeout=3600
                    ):
                        pbar.update(1)
                except TimeoutError:
                    logger.warning("Timeout occurred while rolling out environments")
                    continue
        logger.info("Completed trajectory generation")


def load_mcq(
    mcq: dict, open_question: bool = False, question_id: str | None = None
) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question=mcq["question"],
        options=[
            mcq["ideal"],
            *mcq["distractors"],
        ],
        ideal_answer=mcq["ideal"],
        shuffle_seed=MultipleChoiceQuestion.SEED_USING_QUESTION,
        prompt_without_options=open_question,
        question_id=question_id or "Q",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectories for BixBench tasks"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--replica_id",
        type=int,
        default=None,
        help="Replica ID",
    )
    args = parser.parse_args()

    generator = TrajectoryGenerator(args.config_file, args.replica_id)
    asyncio.run(generator.run())
