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

from bixbench import Query, ZeroshotBaseline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

HF_URL = "futurehouse/BixBench"


def _hf_login():
    HF_TOKEN = os.getenv("HF_TOKEN")
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
        description="BixBench Zero-ShotEvaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--answer-mode",
        choices=["mcq", "openanswer"],
        default="mcq",
        help="Evaluation mode",
    )
    parser.add_argument("--model", default="gpt-4o", help="Model name to use")

    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Model temperature"
    )

    parser.add_argument(
        "--local-csv",
        type=str,
        default=None,
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
    dataset: pd.DataFrame,
    zeroshot_agent: ZeroshotBaseline,
    output_dir: str = "results",
    output_file: str | None = None,
):

    results = []
    for i, row in dataset.iterrows():
        for q_dict in row["questions"]:
            query = await zeroshot_agent.generate_zeroshot_answers(
                Query(
                    id=row["uuid"],
                    question=q_dict["question"],
                    target=q_dict["ideal_answer"],
                    choices=[q_dict[f"distractor_{j}"] for j in range(1, 4)],
                    evaluation_mode=q_dict.get("eval_method", None),
                )
            )

            results.append(
                {
                    "uuid": query.id,
                    "question": query.question,
                    "predicted": query.predicted,
                    "target": query.target,
                    "unsure": query.unsure,
                    "evaluation_mode": query.evaluation_mode,
                }
            )

    # make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_file)
    pd.DataFrame(results).to_csv(output_path, index=False)


async def main():
    logger.info("Starting evaluation...")
    args = parse_args()
    output_file = (
        args.output_file
        if args.output_file is not None
        else f"results_{args.answer_mode}_{args.with_refusal}_{args.model}_{args.temperature}.csv"
    )
    if args.local_csv is None:
        _hf_login()
        dataset = load_dataset(HF_URL)["train"].to_pandas()
        dataset["questions"] = dataset["questions"].apply(eval)
    else:
        dataset = pd.read_csv(args.local_csv, converters={"questions": eval})
    if args.num_examples > 0:
        dataset = dataset.head(args.num_examples)

    zeroshot_agent = ZeroshotBaseline(
        answer_mode=args.answer_mode,
        with_refusal=args.with_refusal,
        model_name=args.model,
        temperature=args.temperature,
    )
    await evaluate(dataset, zeroshot_agent, args.output_dir, output_file)
    logger.info(f"Evaluation completed and results saved to {args.output_dir} folder")


if __name__ == "__main__":
    asyncio.run(main())
