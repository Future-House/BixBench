import fhda
import datasets
from fhda.data_analysis_env import DataAnalysisEnv
from fhda.utils import NBLanguage, load_mcq
import fhda.prompts as prompts
from huggingface_hub import hf_hub_download
import argparse
from ldp.agent import AgentConfig
from ldp.alg.rollout import RolloutManager
import os
import asyncio
import ast
from pathlib import Path
import shutil
import json
from tempfile import mkdtemp
import logging
from aviary.utils import EvalAnswerMode

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
local_data_folder = "data/capsules/"
system_prompt = prompts.CAPSULE_SYSTEM_PROMPT_OPEN
if capsule_mode == "mcq":
    base_prompt = prompts.MCQ_PROMPT_TEMPLATE
elif capsule_mode == "open":
    base_prompt = prompts.OPEN_PROMPT_TEMPLATE
elif capsule_mode == "hypothesis":
    base_prompt = prompts.HYPOTHESIS_PROMPT_TEMPLATE
eval_mode = EvalAnswerMode.LLM
local_output_path = "data/traces/v1"
hf_repo_id = "futurehouse/bixbench-internal"


def load_bixbench():
    bixbench = datasets.load_dataset(hf_repo_id, split="train")
    # Save all datasets locally
    # Create local directory if it doesn't exist
    local_dir = Path(local_data_folder)
    if local_dir.exists() and len(list(local_dir.iterdir())) == 53:
        logger.info("Local data folder already exists with 53 items, skipping download")
        return bixbench
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract all datasets
    for capsule in bixbench:
        zip_filename = capsule["data_folder"]

        # Local paths
        zip_path = local_dir / zip_filename
        extract_dir = local_dir / zip_filename.replace(".zip", "")

        # Download the zip file
        print(zip_filename)
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=zip_filename,
            local_dir=local_dir,
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


def string_to_list(example):
    example["questions"] = ast.literal_eval(example["questions"])
    example["categories"] = ast.literal_eval(example["categories"])
    return example


def environment_factory(capsule: dict) -> DataAnalysisEnv:
    raw_questions = ast.literal_eval(capsule["questions"])
    processed_questions = [
        load_mcq(i, open_question=True, question_id=i["id"]) for i in raw_questions
    ]
    problem = base_prompt.format(
        questions="\n-------\n".join([i.question_prompt for i in processed_questions])
    )
    answer = {i.question_id: i.ideal_answer for i in processed_questions}
    work_dir = Path(local_output_path).absolute() / capsule["short_id"]
    work_dir.mkdir(parents=True, exist_ok=True)
    local_data_folder_path = Path(local_data_folder) / capsule["data_folder"].replace(
        ".zip", ""
    )
    # Copy all files from data folder to work directory
    for item in local_data_folder_path.iterdir():
        if item.is_file():
            shutil.copy2(item, work_dir)
        elif item.is_dir():
            shutil.copytree(item, work_dir / item.name)
    nb_path = work_dir / notebook_name

    capsule_args = {
        "problem_id": capsule["short_id"],
        "problem": problem,
        "eval_mode": "test",
        "nb_path": nb_path,
        "work_dir": work_dir,
        "language": language,
        "system_prompt": system_prompt,
        "metadata": capsule,
        "answer": answer,
        "mcqs": processed_questions,
        "use_tmp_work_dir": False,
    }

    return DataAnalysisEnv(**capsule_args)


def main():
    bixbench = load_bixbench()
    # Construct agent and rollout manager
    agent = agent_config.construct_agent()
    rollout = RolloutManager(agent=agent, callbacks=callbacks)

    # Construct batch of environments
    all_trajectories = []
    for capsule in bixbench:
        env = environment_factory(capsule)
        trajectories = await rollout.sample_trajectories(
            environments=[env], max_steps=max_rollout_steps
        )
        # print(trajectories)

        all_trajectories.append(trajectories)
        break
    # Save trajectories to file
    # with open('trajectories.json', 'w') as f:
    #     json.dump(all_trajectories, f)


if __name__ == "__main__":
    asyncio.run(main())
