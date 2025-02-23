import pandas as pd
from dotenv import load_dotenv

load_dotenv()
import argparse
import sys
import asyncio
import os
from typing import Dict, Any
from lmi import LiteLLMModel
import json
from pathlib import Path
from bixbench import (
    grade_mcq_answer,
    grade_open_ended_answer,
    compute_metrics,
    EvalMode,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grade answers from a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file", required=True, help="Input CSV file with answers to grade"
    )
    parser.add_argument(
        "--eval-mode",
        choices=["mcq", "openanswer"],
        required=True,
        help="Evaluation mode",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="Model name for open-ended grading"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Model temperature"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save results"
    )
    parser.add_argument("--output-file", default=None, help="Output JSON filename")
    return parser.parse_args()


async def grade_answers(
    input_file: str,
    eval_mode: EvalMode,
    model_name: str = "gpt-4o",
    temperature: float = 1.0,
    **kwargs: Dict[str, Any],
):
    """Grade answers based on evaluation mode."""
    df = pd.read_csv(input_file)
    try:
        if eval_mode == EvalMode.openanswer:

            llm_client = LiteLLMModel(
                name=f"{model_name}",
                config={"name": model_name, "temperature": temperature, **kwargs},
            )
            df["grade"], df["correct"], df["sure"] = zip(
                *[
                    await grade_open_ended_answer(
                        row["question"], row["target"], row["predicted"], llm_client
                    )
                    for _, row in df.iterrows()
                ]
            )
        else:
            df["grade"], df["correct"], df["sure"] = zip(
                *[
                    grade_mcq_answer(row["target"], row["predicted"], row["unsure"])
                    for _, row in df.iterrows()
                ]
            )

        # save df as pd
        df.to_csv(input_file, index=False)

        return compute_metrics(df["grade"].to_list(), df["sure"].to_list())

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def main():
    try:
        args = parse_args()
        metrics = await grade_answers(
            args.input_file,
            args.eval_mode,
            args.model,
            args.temperature,
        )

        # make dir if doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if args.output_file is None:
            output_file = Path(args.input_file).stem + "_graded.json"
        else:
            output_file = args.output_file

        output_path = Path(args.output_dir) / output_file

        print(metrics)
        print(f"Saving results to {output_path}")
        with open(os.path.join(output_path), "w") as f:
            json.dump(metrics, f, indent=4)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
