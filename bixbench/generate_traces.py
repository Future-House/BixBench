import fhda
import datasets
from fhda.data_analysis_env import DataAnalysisEnv
from fhda.utils import NBLanguage, load_mcq
import fhda.prompts as prompts

import argparse
from ldp.agent import AgentConfig
from ldp.alg.rollout import RolloutManager
import os
import asyncio
import ast
from pathlib import Path
import shutil
from tempfile import mkdtemp

# CONFIG
agent_config = AgentConfig(
    agent_type="ReActAgent",
    agent_kwargs={
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
)

max_rollout_steps = 5
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
eval_mode = "skip"
local_output_path = "data/traces/v1"
def load_bixbench():
    bixbench = datasets.load_dataset("futurehouse/bixbench-internal", split="train")
    return bixbench

def string_to_list(example):
    example['questions'] = ast.literal_eval(example['questions'])
    example['categories'] = ast.literal_eval(example['categories'])
    return example

def environment_factory(capsule: dict) -> DataAnalysisEnv:
    raw_questions = ast.literal_eval(capsule["questions"])
    processed_questions = [
        load_mcq(i, open_question=True, question_id=i["id"]) for i in raw_questions
    ]
    problem = base_prompt.format(
        questions="\n-------\n".join(
            [i.question_prompt for i in processed_questions]
        )
    )
    answer = {i.question_id: i.ideal_answer for i in processed_questions}
    work_dir = Path(local_output_path) / capsule["short_id"]
    work_dir.mkdir(parents=True, exist_ok=True)
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
    }
    print(capsule_args)
    print(os.environ)
    
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
        trajectories = asyncio.run(rollout.sample_trajectories(environments=[env], max_steps=max_rollout_steps))
        # print(trajectories)
        all_trajectories.append(trajectories)
        break
        
    



if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Generate traces for BixBench dataset')
    # parser.add_argument('--output-dir', type=str, default='data/traces',
    #                    help='Directory to save traces')
    # parser.add_argument('--agent-config', type=str, default='gpt-4',
    #                    help='Path to the agent config yaml file') 
    # parser.add_argument('--temperature', type=float, default=1.0,
    #                    help='Temperature for model sampling')
    # parser.add_argument('--max-tokens', type=int, default=2048,
    #                    help='Maximum tokens for model response')

    # args = parser.parse_args()
    # bixbench = load_bixbench()
    # generate_traces(bixbench)
    # Check for api keys based on model config
    # Check for docker if using docker
    main()
