import argparse
import ast
import argparse
import ast
import asyncio
import logging
import os
import sys
from itertools import islice
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from bixbench import AgentInput, ZeroshotBaseline

logger = logging.getLogger(__name__)

dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)


HF_TOKEN = os.getenv("HF_TOKEN")
HF_URL = "futurehouse/BixBench-internal"

if HF_TOKEN is None:
    logger.warning(
        "HF_TOKEN environment variable not found. Please set this in your environment "
        "or create a .env file with this value to enable Hugging Face model access."
    )
else:
    try:
        login(token=HF_TOKEN)
    except Exception:
        logger.exception("Failed to login with HF_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BixBench Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--eval-mode",
        choices=["mcq", "openanswer"],
        default="mcq",
        help="Evaluation mode",
    )
    parser.add_argument("--model", default="gpt-4o", help="Model name to use")

    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Model temperature"
    )

    parser.add_argument(
        "--with-refusal",
        action="store_true",
        default=False,
        help="Add refusal option for MCQs",
    )

    # used for testing purposes
    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="Number of examples to evaluate. Default is -1 for all examples",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results. Default is 'results'",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Output file name (optional)"
    )
    return parser.parse_args()


def string_to_list(example):
    example["questions"] = ast.literal_eval(example["questions"])
    example["categories"] = ast.literal_eval(example["categories"])
    return example


async def evaluate(
    baseline_agent: ZeroshotBaseline,
    num_examples=-1,
    output_dir="results",
    output_file=None,
) -> None:
) -> None:
    # Load dataset
    dataset = load_dataset(HF_URL)
    # map string to list
    dataset = dataset.map(string_to_list)
    results = []
    num_examples = num_examples if num_examples > 0 else len(dataset["train"])
    for i, row in enumerate(islice(dataset["train"], num_examples)):
        for q_dict in row["questions"]:
            agent_input = AgentInput(
            agent_input = AgentInput(
                id=dataset["train"][i]["uuid"],
                question=q_dict["question"],
                target=q_dict["ideal_answer"],
                choices=[q_dict[f"distractor_{i}"] for i in range(1, 4)],
            )
            (
                answer,
                target,
                unsure_answer,
            ) = await baseline_agent.generate_zeroshot_answers(agent_input)
            ) = await baseline_agent.generate_zeroshot_answers(agent_input)
            results.append(
                {
                    "uuid": dataset["train"][i]["uuid"],
                    "question": agent_input.question,
                    "question": agent_input.question,
                    "predicted": answer,
                    "target": target,
                    "unsure": unsure_answer,
                }
            )

    # make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_file)
    pd.DataFrame(results).to_csv(output_path, index=False)
    pd.DataFrame(results).to_csv(output_path, index=False)


async def main():
    try:
        logger.info("Starting evaluation...")
        args = parse_args()
        output_file = (
            args.output_file
            if args.output_file is not None
            else f"results_{args.eval_mode}_{args.with_refusal}_{args.model}_{args.temperature}.csv"
        )
        baseline_agent = ZeroshotBaseline(
            eval_mode=args.eval_mode,
            with_refusal=args.with_refusal,
            model_name=args.model,
            temperature=args.temperature,
        )
        await evaluate(baseline_agent, args.num_examples, args.output_dir, output_file)
        logger.info(
            f"Evaluation completed and results saved to {args.output_dir} folder"
        )
    except Exception as e:
        logger.info(f"Error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
