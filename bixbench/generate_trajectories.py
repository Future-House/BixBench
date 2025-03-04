import argparse
import ast
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import datasets
import litellm
import yaml
from fhda.data_analysis_env import DataAnalysisEnv
from fhda.utils import collect_notebook_stats, load_mcq
from huggingface_hub import hf_hub_download
from ldp.agent import Agent
from ldp.alg.rollout import RolloutManager
from ldp.data_structures import Trajectory, Transition
from tqdm.auto import tqdm

from bixbench.models import BixbenchConfig

litellm.verbose = False
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

    def __init__(self, config_path=DEFAULT_CONFIG_PATH) -> None:
        """
        Initialize the TrajectoryGenerator with config and create necessary directories.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self.load_config(config_path)
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

    async def process_capsule(self, capsule: dict[str, Any]) -> None:
        """
        Process a single benchmark capsule by downloading and extracting necessary files.

        Args:
            capsule: Dictionary containing capsule information
        """
        zip_filename = capsule["data_folder"]
        extract_dir = self.config.local_data_folder / zip_filename.replace(".zip", "")
        zip_path = self.config.local_data_folder / zip_filename

        # Check if capsule folder exists and is non-empty
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.debug(
                "Capsule folder %s already exists and is non-empty", extract_dir.name
            )
            capsule["local_data_folder"] = extract_dir
            return

        # Download and process if not already present
        await asyncio.to_thread(
            hf_hub_download,
            repo_id=self.config.paths.hf_repo_id,
            filename=zip_filename,
            local_dir=self.config.local_data_folder,
            repo_type="dataset",
        )

        await asyncio.to_thread(self._extract_and_process_files, zip_path, extract_dir)
        capsule["local_data_folder"] = extract_dir

    async def load_bixbench(self) -> list[dict[str, Any]]:
        """
        Load BixBench dataset and process all capsules.

        Returns:
            List[Dict[str, Any]]: List of processed benchmark capsules
        """
        bixbench = datasets.load_dataset(
            self.config.paths.hf_repo_id, split="train"
        ).to_list()

        # Process all capsules concurrently
        tasks = [self.process_capsule(capsule) for capsule in bixbench]
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
        data_folder = next(p for p in extract_dir.iterdir() if "Data" in p.name)

        # Move contents of Data folder to parent directory
        for item in data_folder.iterdir():
            shutil.move(str(item), str(extract_dir / item.name))

        # Remove the Data folder
        shutil.rmtree(data_folder)

        # Safely remove Notebook folder if it exists
        try:
            notebook_folder = next(
                p
                for p in extract_dir.iterdir()
                if "Notebook" in p.name and p.is_dir()  # Only match directories
            )
            shutil.rmtree(notebook_folder)
        except StopIteration:
            # No Notebook folder found, that's okay
            pass

        # Remove any .ipynb files in the extract directory
        for ipynb_file in extract_dir.glob("*.ipynb"):
            ipynb_file.unlink()

        # Remove the zip file
        zip_path.unlink()

    async def store_trajectory(
        self, trajectory: Trajectory, env: DataAnalysisEnv
    ) -> None:
        """
        Store trajectory and environment information to disk.

        Args:
            trajectory: The trajectory to store
            env: The environment that generated the trajectory
        """
        extract = {
            "problem_id": env.problem_id,
            "agent_answer": env.state.answer,
            "ideal_answer": env.answer,
            "problem": env.problem,
            "mcq_options": [q.options for q in env.mcqs] if env.mcqs else [],
            "mcq_question": [q.question for q in env.mcqs] if env.mcqs else [],
            "notebook_stats": collect_notebook_stats(env.state.nb),
            "num_actions": len(env.state.actions),
            "question_format": self.config.capsule.mode,
            "refusal_option": self.config.capsule.include_refusal_option,
            "model": self.config.agent.agent_kwargs["llm_model"]["name"],
            # Local data folder is not serializable
            "metadata": {
                k: v for k, v in env.metadata.items() if k != "local_data_folder"
            },
            "refusal_options": {
                q.question_id: q.unsure_answer_letter for q in (env.mcqs or [])
            },
            "nb": env.state.nb,
            "run_name": self.config.run_name,
        }

        # Download run metadata
        with (self.config.local_trajectories_dir / f"{env.problem_id}.json").open(
            "w"
        ) as f:
            json.dump(
                extract,
                f,
                indent=4,
            )
        # Download run trajectory
        await trajectory.to_jsonl(
            self.config.local_trajectories_dir / f"{env.problem_id}.jsonl"
        )

    def environment_factory(self, capsule: dict[str, Any]) -> DataAnalysisEnv:
        """
        Create a DataAnalysisEnv instance from a capsule.

        Args:
            capsule: Dictionary containing capsule information

        Returns:
            DataAnalysisEnv: Initialized environment
        """
        raw_questions = ast.literal_eval(capsule["questions"])
        processed_questions = [
            load_mcq(i, open_question=True, question_id=i["id"]) for i in raw_questions
        ]
        problem = self.config.base_prompt.format(
            questions="\n-------\n".join([
                i.question_prompt for i in processed_questions
            ])
        )
        answer = {i.question_id: i.ideal_answer for i in processed_questions}
        work_dir = (self.config.local_workspace_dir / capsule["uuid"]).absolute()
        work_dir.mkdir(parents=True, exist_ok=True)
        local_capsule_data_path = self.config.local_data_folder / capsule[
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
        capsule["avoid_images"] = self.config.capsule.avoid_images
        capsule["include_refusal_option"] = self.config.capsule.include_refusal_option

        env_args = {
            "problem_id": capsule["short_id"],
            "problem": problem,
            "eval_mode": self.config.capsule.eval_mode,
            "nb_path": nb_path,
            "work_dir": work_dir,
            "language": self.config.notebook.language,
            "system_prompt": self.config.system_prompt,
            "metadata": capsule,
            "answer": answer,
            "mcqs": processed_questions,
            "use_tmp_work_dir": False,
        }

        return DataAnalysisEnv(**env_args)

    async def custom_rollout(
        self, agent: Agent, environment: DataAnalysisEnv
    ) -> Trajectory:
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
    ) -> list[Trajectory | tuple[Trajectory, DataAnalysisEnv]]:
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
            return [
                (trajectory, environment)
                for trajectory, environment in zip(
                    trajectories, list_of_environments, strict=True
                )
            ]

        agent = self.config.agent_config.construct_agent()
        rollout_manager = getattr(self, f"{self.config.rollout.rollout_type}_rollout")

        return await asyncio.gather(*[
            rollout_manager(agent, environment) for environment in list_of_environments
        ])

    async def run(self) -> None:
        """Run the full trajectory generation pipeline."""
        logger.info("Loading BixBench dataset...")
        bixbench = await self.load_bixbench()

        # Process environments in batches with tqdm progress bar
        total_batches = (
            len(bixbench) + self.config.rollout.batch_size - 1
        ) // self.config.rollout.batch_size

        with tqdm(total=len(bixbench), desc="Processing benchmark tasks") as pbar:
            for i in range(0, len(bixbench), self.config.rollout.batch_size):
                batch = bixbench[i : i + self.config.rollout.batch_size]
                environments = [self.environment_factory(capsule) for capsule in batch]
                results = await self.batch_rollout(environments)
                for trajectory, env in results:
                    await self.store_trajectory(trajectory, env)

                # Update progress bar
                pbar.update(len(batch))
                pbar.set_postfix({
                    "batch": f"{i // self.config.rollout.batch_size + 1}/{total_batches}"
                })


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
    args = parser.parse_args()

    generator = TrajectoryGenerator(args.config)
    asyncio.run(generator.run())
