import argparse
import ast
import asyncio
import json
import operator
import os

import config as cfg
import nbformat
import pandas as pd
from fhda.utils import view_notebook

from bixbench import plotting_utils
from bixbench import postprocessing_utils as utils

pd.options.mode.chained_assignment = None
# If true, save and load intermediate results to avoid re-running the same steps


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file and process specific columns.

    Args:
        path (str): Path to the CSV file containing raw data

    Returns:
        pd.DataFrame: Processed DataFrame with converted column types
    """
    raw_data = pd.read_csv(path)
    mapping = {
        "agent_answer": utils.load_answer,
        "ideal_answer": utils.load_answer,
        "mcq_options": ast.literal_eval,
        "mcq_question": ast.literal_eval,
        "nb": lambda x: nbformat.reads(json.dumps(ast.literal_eval(x)), as_version=4),
        "avoid_images": bool,
        "actions": int,
        "refusal_option": bool,
    }
    for col, func in mapping.items():
        if col in raw_data.columns:
            raw_data[col] = raw_data[col].apply(func)

    # Convert json notebook to markdown for postprocessing
    if "nb" in raw_data.columns and "nb_md" not in raw_data.columns:
        df_md = pd.DataFrame(
            raw_data["nb"].apply(lambda x: view_notebook(x.cells, "python")).tolist(),
            columns=["md_notebook", "md_images"],
        )
        raw_data[["md_notebook", "md_images"]] = df_md
    return raw_data


async def process_trajectories(
    df: pd.DataFrame, checkpointing: bool = True
) -> pd.DataFrame:
    """
    Create a gradable dataframe from a raw dataframe of trajectories.

    This function processes the raw data, runs evaluation loops, and saves
    the results to CSV files for further analysis.

    Args:
        df (pd.DataFrame): Raw data containing model trajectories
        checkpointing (bool): Whether to save intermediate results to CSV files

    Returns:
        pd.DataFrame: Processed evaluation dataframe
    """
    eval_df = utils.create_eval_df(df)
    eval_df = await utils.run_eval_loop(eval_df)

    # Create correct column for open ended questions
    eval_df.loc[eval_df.question_format == "open", "correct"] = eval_df.loc[
        eval_df.question_format == "open", "llm_answer"
    ].apply(lambda x: x == "1")
    # Extract XML from LLM MCQ answers
    eval_df.loc[eval_df.question_format == "mcq", "llm_answer"] = eval_df.loc[
        eval_df.question_format == "mcq", "llm_answer"
    ].apply(utils.xml_extract)
    # Compare LLM answers to ideal answers
    eval_df.loc[eval_df.question_format == "mcq", "correct"] = (
        eval_df.loc[eval_df.question_format == "mcq", "llm_answer"]
        == eval_df.loc[eval_df.question_format == "mcq", "correct_letter"]
    )
    if checkpointing:
        eval_df.to_csv("bixbench_results/eval_df.csv", index=False)
    return eval_df


async def run_majority_vote(eval_df: pd.DataFrame, k_value: int = 10) -> None:
    """
    Implement majority voting evaluation across different model configurations.

    DISCLAIMER: This function is highly tailored to the BixBench paper requirements.
    It is not designed to be used as a general function for comparing model performance.

    This function reads evaluation data, performs majority voting analysis for
    multiple choice questions, and produces visualization comparing different model
    configurations with and without specific features.
    """
    # Only run majority vote on mcq questions
    maj_vote_df = eval_df[eval_df.question_format == "mcq"].copy()

    if maj_vote_df.empty:
        print("No MCQ questions found, skipping majority vote")
        return

    # Store results for all runs
    run_results = {}

    for run_name in maj_vote_df.run_name.unique():
        grouped_df = maj_vote_df[maj_vote_df.run_name == run_name].copy()
        grouped_df = grouped_df.groupby("uuid").agg(list)
        grouped_df["correct_letter"] = grouped_df["correct_letter"].apply(
            operator.itemgetter(0)
        )
        grouped_df = grouped_df.dropna()
        k_values, means, stds = utils.run_majority_voting(
            grouped_df, range(1, k_value), k_value
        )
        run_results[run_name] = (k_values, means, stds)

    for group_name, group_runs in cfg.MAJORITY_VOTE_GROUPS.items():
        random_baselines = [0.2]
        random_baselines_labels = ["Random Guess with Refusal Option"]
        if any("without_refusal" in run_name for run_name in group_runs):
            random_baselines.append(0.25)
            random_baselines_labels.append("Random Guess without Refusal Option")

        plotting_utils.majority_vote_accuracy_by_k(
            {run_name: run_results[run_name] for run_name in group_runs},
            name=group_name,
            random_baselines=random_baselines,
            random_baselines_labels=random_baselines_labels,
        )


async def compare_runs(eval_df: pd.DataFrame) -> None:
    """
    Compare performance between different model architectures.

    DISCLAIMER: This function is highly tailored to the BixBench paper requirements.
    It is not designed to be used as a general function for comparing model performance.

    This function analyzes and visualizes the performance differences between
    GPT-4o and Claude models across different question formats.
    """
    # Filter eval_df to only include run_names configured in config.py
    eval_df = eval_df[eval_df["run_name"].isin(utils.flatten_list(cfg.RUN_NAME_GROUPS))]

    # Calculate means and confidence intervals
    results = utils.calculate_results(eval_df, total_questions=2960)

    # Plot results
    plotting_utils.plot_model_comparison(
        results, cfg.BASELINES, cfg.RUN_NAME_GROUPS, cfg.COLOR_GROUPS
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process BixBench evaluation data")
    parser.add_argument(
        "--data_path",
        type=str,
        default="bixbench_results/raw_trajectory_data.csv",
        help="Path to the raw trajectory data CSV file",
    )
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        default=True,
        help="Whether to save and load intermediate results",
    )
    args = parser.parse_args()

    # # Load raw trajectory data
    # os.makedirs("bixbench_results", exist_ok=True)
    # data = load_raw_data(args.data_path)

    # # Process trajectories and save eval df
    # eval_df = asyncio.run(process_trajectories(data, checkpointing=args.checkpointing))

    if args.checkpointing:
        eval_df = pd.read_csv("bixbench_results/eval_df.csv")
        eval_df["correct"] = eval_df["correct"].astype(bool)

    # Run majority vote
    asyncio.run(run_majority_vote(eval_df, k_value=10))

    # Compare runs
    # asyncio.run(compare_runs(eval_df))
