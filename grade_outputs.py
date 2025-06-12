import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from lmi import LiteLLMModel

from bixbench import (
    AnswerMode,
    GradeAnswer,
    compute_metrics,
)

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grade answers from a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file", required=True, help="Input CSV file with answers to grade"
    )
    parser.add_argument(
        "--answer-mode",
        choices=["mcq", "openanswer"],
        required=True,
        help="Answer mode",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="Model name for open-answer grading"
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
    input_file: str | Path,
    answer_mode: AnswerMode,
    model_name: str = "gpt-4o",
    temperature: float = 1.0,
    **kwargs: dict[str, Any],
):
    """Grade answers based on evaluation mode."""
    query_df = pd.read_csv(input_file)

    if answer_mode == AnswerMode.openanswer:
        llm_client = LiteLLMModel(
            name=f"{model_name}",
            config={"name": model_name, "temperature": temperature, **kwargs},
        )
        grader = GradeAnswer(
            answer_mode=answer_mode,
            llm_client=llm_client,
        )

        results = [
            await grader.grade(
                question=row["question"],
                target=str(row["target"]),
                predicted=str(row["predicted"]),
                unsure=None,
                evaluation_mode=row.get("evaluation_mode", "llm_verifier"),
                partial_match=True,
                llm_match=True,
            )
            for _, row in query_df.iterrows()
        ]

        query_df["grade"], query_df["correct"], query_df["sure"] = zip(
            *results, strict=True
        )
    elif answer_mode == AnswerMode.mcq:
        grader = GradeAnswer(answer_mode=answer_mode)
        results = [
            await grader.grade(
                target=row["target"],
                predicted=row["predicted"],
                unsure=row["unsure"],
                evaluation_mode="str_verifier",
            )
            for _, row in query_df.iterrows()
        ]

        query_df["grade"], query_df["correct"], query_df["sure"] = zip(
            *results, strict=True
        )

    else:
        raise ValueError(f"Unknown answer mode: {answer_mode}")

    # save query_df as pd
    query_df.to_csv(input_file, index=False)
    return compute_metrics(query_df["grade"].to_list(), query_df["sure"].to_list())


async def main():
    try:
        args = parse_args()
        metrics = await grade_answers(
            args.input_file,
            args.answer_mode,
            args.model,
            args.temperature,
        )

        # make dir if doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_file = (
            Path(args.input_file).stem + "_graded.json"
            if args.output_file is None
            else args.output_file
        )

        output_path = Path(args.output_dir) / output_file

        print(metrics)
        print(f"Saving results to {output_path}")
        with open(os.path.join(output_path), "w") as f:
            json.dump(metrics, f, indent=4)

    except Exception as e:
        print(f"Error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
