import ast
import asyncio
import json
import logging
import shutil
from pathlib import Path
import yaml

import datasets
from huggingface_hub import hf_hub_download
from ldp.agent import AgentConfig
from ldp.alg.rollout import RolloutManager
from ldp.data_structures import Trajectory

from fhda import prompts
from fhda.data_analysis_env import DataAnalysisEnv
from fhda.utils import NBLanguage, load_mcq, collect_notebook_stats
from aviary.utils import EvalAnswerMode

logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Convert config values to appropriate types/enums
    agent_config = AgentConfig(
        agent_type=config["agent"]["agent_type"],
        agent_kwargs=config["agent"]["agent_kwargs"],
    )

    # Get system prompt and base prompt from config
    system_prompt = getattr(prompts, config["capsule"]["system_prompt"])
    capsule_mode = config["capsule"]["mode"]
    base_prompt = getattr(prompts, config["capsule"]["prompt_templates"][capsule_mode])

    return {
        "agent_config": agent_config,
        "max_rollout_steps": config["rollout"]["max_steps"],
        "notebook_name": config["notebook"]["name"],
        "language": NBLanguage[config["notebook"]["language"].upper()],
        "capsule_mode": capsule_mode,
        "include_refusal_option": config["capsule"]["include_refusal_option"],
        "system_prompt": system_prompt,
        "base_prompt": base_prompt,
        "eval_mode": EvalAnswerMode[config["capsule"]["eval_mode"]],
        "local_workspace_dir": Path(config["paths"]["workspace_dir"]).absolute(),
        "local_traces_dir": Path(config["paths"]["traces_dir"]).absolute(),
        "local_data_folder": Path(config["paths"]["data_folder"]).absolute(),
        "hf_repo_id": config["paths"]["hf_repo_id"],
    }


# Load config at module level
config = load_config()

# Create directories
config["local_workspace_dir"].mkdir(parents=True, exist_ok=True)
config["local_traces_dir"].mkdir(parents=True, exist_ok=True)
config["local_data_folder"].mkdir(parents=True, exist_ok=True)


def load_bixbench() -> datasets.Dataset:
    bixbench = datasets.load_dataset(config["hf_repo_id"], split="train").to_list()[4:]
    # Save all datasets locally
    # Create local directory if it doesn't exist
    if (
        config["local_data_folder"].exists()
        and len(list(config["local_data_folder"].iterdir())) == 53
    ):
        logger.info("Local data folder already exists with 53 items, skipping download")
        return bixbench

    # Download and extract all datasets
    for capsule in bixbench:
        zip_filename = capsule["data_folder"]

        # Local paths
        zip_path = config["local_data_folder"] / zip_filename
        extract_dir = config["local_data_folder"] / zip_filename.replace(".zip", "")

        # Download the zip file
        hf_hub_download(
            repo_id=config["hf_repo_id"],
            filename=zip_filename,
            local_dir=config["local_data_folder"],
            repo_type="dataset",
        )

        # Extract the zip file
        shutil.unpack_archive(zip_path, extract_dir)

        # Get the Data folder path
        data_folder = next(p for p in extract_dir.iterdir() if "Data" in p.name)

        # Move contents of Data folder to parent directory
        for item in data_folder.iterdir():
            shutil.move(str(item), str(extract_dir / item.name))

        # Remove the Data folder and Notebook folder
        shutil.rmtree(data_folder)
        notebook_folder = next(p for p in extract_dir.iterdir() if "Notebook" in p.name)
        shutil.rmtree(notebook_folder)
        # Remove any .ipynb files in the extract directory
        for ipynb_file in extract_dir.glob("*.ipynb"):
            ipynb_file.unlink()
        # Remove the zip file
        zip_path.unlink()
        capsule["local_data_folder"] = extract_dir
        break
    return bixbench


async def store_trajectory(trajectory: Trajectory, env: DataAnalysisEnv) -> None:
    extract = {
        "problem_id": env.problem_id,
        "agent_answer": env.state.answer,
        "ideal_answer": env.answer,
        "problem": env.problem,
        "mcq_options": [q.options for q in env.mcqs] if env.mcqs else [],
        "mcq_question": [q.question for q in env.mcqs] if env.mcqs else [],
        "notebook_stats": collect_notebook_stats(env.state.nb),
        "num_actions": len(env.state.actions),
        # Local data folder is not serializable
        "metadata": {k: v for k, v in env.metadata.items() if k != "local_data_folder"},
        "refusal_options": {
            q.question_id: q.unsure_answer_letter for q in (env.mcqs or [])
        },
        "nb": env.state.nb,
    }

    # Download run metadata
    with (config["local_traces_dir"] / f"{env.problem_id}.json").open("w") as f:
        json.dump(
            extract,
            f,
            indent=4,
        )
    # Download run trajectory
    await trajectory.to_jsonl(config["local_traces_dir"] / f"{env.problem_id}.jsonl")


def environment_factory(capsule: dict) -> DataAnalysisEnv:
    raw_questions = ast.literal_eval(capsule["questions"])
    processed_questions = [
        load_mcq(i, open_question=True, question_id=i["id"]) for i in raw_questions
    ]
    problem = config["base_prompt"].format(
        questions="\n-------\n".join([i.question_prompt for i in processed_questions])
    )
    answer = {i.question_id: i.ideal_answer for i in processed_questions}
    work_dir = (config["local_workspace_dir"] / capsule["uuid"]).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)
    local_capsule_data_path = config["local_data_folder"] / capsule[
        "data_folder"
    ].replace(".zip", "")
    # Copy all files from data folder to work directory
    for item in local_capsule_data_path.iterdir():
        if item.is_file():
            shutil.copy2(item, work_dir)
        elif item.is_dir():
            shutil.copytree(item, work_dir / item.name)
    nb_path = work_dir / config["notebook_name"]

    env_args = {
        "problem_id": capsule["short_id"],
        "problem": problem,
        "eval_mode": config["eval_mode"],
        "nb_path": nb_path,
        "work_dir": work_dir,
        "language": config["language"],
        "system_prompt": config["system_prompt"],
        "metadata": capsule,
        "answer": answer,
        "mcqs": processed_questions,
        "use_tmp_work_dir": False,
    }

    return DataAnalysisEnv(**env_args)


async def main() -> None:
    bixbench = load_bixbench()
    # Construct agent and rollout manager
    agent = config["agent_config"].construct_agent()
    rollout = RolloutManager(agent=agent)

    # Construct batch of environments
    all_trajectories: list = []
    for capsule in bixbench:
        env = environment_factory(capsule)
        trajectories = await rollout.sample_trajectories(
            environments=[env], max_steps=config["max_rollout_steps"]
        )
        await store_trajectory(trajectories[0], env)
        all_trajectories.append(trajectories)
        break


if __name__ == "__main__":
    asyncio.run(main())
