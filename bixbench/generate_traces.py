import argparse
import ast
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

import datasets
from huggingface_hub import hf_hub_download
from ldp.agent import AgentConfig
from ldp.alg.rollout import RolloutManager
from ldp.data_structures import Trajectory

import fhda
import fhda.prompts as prompts
from aviary.utils import EvalAnswerMode
from fhda.data_analysis_env import DataAnalysisEnv
from fhda.utils import NBLanguage, load_mcq, collect_notebook_stats

logger = logging.getLogger(__name__)
# CONFIG
agent_config = AgentConfig(
    agent_type="ReActAgent",
    agent_kwargs={
        "model": "gpt-4o",
        "temperature": 0.0,
    },
)

max_rollout_steps = 25
callbacks = []
notebook_name = "notebook.ipynb"
language = NBLanguage.PYTHON
capsule_mode = "mcq"
include_refusal_option = True
system_prompt = prompts.CAPSULE_SYSTEM_PROMPT_OPEN
if capsule_mode == "mcq":
    base_prompt = prompts.MCQ_PROMPT_TEMPLATE
elif capsule_mode == "open":
    base_prompt = prompts.OPEN_PROMPT_TEMPLATE
elif capsule_mode == "hypothesis":
    base_prompt = prompts.HYPOTHESIS_PROMPT_TEMPLATE
eval_mode = EvalAnswerMode.LLM
local_workspace_dir = Path("data/workspace").absolute()
local_workspace_dir.mkdir(parents=True, exist_ok=True)
local_traces_dir = Path("data/traces").absolute()
local_traces_dir.mkdir(parents=True, exist_ok=True)
local_data_folder = Path("data/capsules").absolute()
local_data_folder.mkdir(parents=True, exist_ok=True)
hf_repo_id = "futurehouse/bixbench-internal"


def load_bixbench() -> datasets.Dataset:
    bixbench = datasets.load_dataset(hf_repo_id, split="train").to_list()[2:]
    # Save all datasets locally
    # Create local directory if it doesn't exist
    if local_data_folder.exists() and len(list(local_data_folder.iterdir())) == 53:
        logger.info("Local data folder already exists with 53 items, skipping download")
        return bixbench

    # Download and extract all datasets
    for capsule in bixbench:
        zip_filename = capsule["data_folder"]

        # Local paths
        zip_path = local_data_folder / zip_filename
        extract_dir = local_data_folder / zip_filename.replace(".zip", "")

        # Download the zip file
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=zip_filename,
            local_dir=local_data_folder,
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
        "question_rewards": env.question_rewards,
        "notebook_stats": collect_notebook_stats(env.state.nb),
        "actions": env.state.actions,
        "metadata": env.metadata,
        "refusal_options": {
            q.question_id: q.unsure_answer_letter for q in (env.mcqs or [])
        },
        "nb": env.state.nb
    }

    # Download run metadata
    with (local_traces_dir / f"{env.problem_id}.json").open("w") as f:
        json.dump(
            extract,
            f,
            indent=4,
        )
    # Download run trajectory
    await trajectory.to_jsonl(local_traces_dir / f"{env.problem_id}.jsonl")


def environment_factory(capsule: dict) -> DataAnalysisEnv:
    raw_questions = ast.literal_eval(capsule["questions"])
    processed_questions = [
        load_mcq(i, open_question=True, question_id=i["id"]) for i in raw_questions
    ]
    problem = base_prompt.format(
        questions="\n-------\n".join([i.question_prompt for i in processed_questions])
    )
    answer = {i.question_id: i.ideal_answer for i in processed_questions}
    work_dir = (local_workspace_dir / capsule["uuid"]).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)
    local_capsule_data_path = local_data_folder / capsule["data_folder"].replace(
        ".zip", ""
    )
    # Copy all files from data folder to work directory
    for item in local_capsule_data_path.iterdir():
        if item.is_file():
            shutil.copy2(item, work_dir)
        elif item.is_dir():
            shutil.copytree(item, work_dir / item.name)
    nb_path = work_dir / notebook_name

    env_args = {
        "problem_id": capsule["short_id"],
        "problem": problem,
        "eval_mode": eval_mode,
        "nb_path": nb_path,
        "work_dir": work_dir,
        "language": language,
        "system_prompt": system_prompt,
        "metadata": capsule,
        "answer": answer,
        "mcqs": processed_questions,
        "use_tmp_work_dir": False,
    }

    return DataAnalysisEnv(**env_args)


async def main() -> None:
    bixbench = load_bixbench()
    # Construct agent and rollout manager
    agent = agent_config.construct_agent()
    rollout = RolloutManager(agent=agent, callbacks=callbacks)

    # Construct batch of environments
    all_trajectories: list = []
    for capsule in bixbench:
        env = environment_factory(capsule)
        trajectories = await rollout.sample_trajectories(
            environments=[env], max_steps=max_rollout_steps
        )

        all_trajectories.append(trajectories)
        break

if __name__ == "__main__":
    asyncio.run(main())
