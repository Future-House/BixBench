import argparse
import ast
import asyncio
import json
import operator

import nbformat
import pandas as pd
from fhda.utils import view_notebook

from bixbench import plotting_utils

# Import these from local directory - adjust path if needed
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

    eval_df.to_csv("bixbench_results/eval_loop_results.csv", index=False)
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
        grouped_df["llm_answer"] = grouped_df["llm_answer"].fillna("X")
        grouped_df = grouped_df.groupby("uuid").agg(list)
        grouped_df["correct_letter"] = grouped_df["correct_letter"].apply(
            operator.itemgetter(0)
        )
        grouped_df = grouped_df.dropna()
        k_values, means, stds = utils.run_majority_voting(
            grouped_df, range(1, k_value), k_value
        )
        run_results[run_name] = (k_values, means, stds)

    r1 = {
        "claude_mcq_image_with_refusal": "Claude with vision",
        "claude_mcq_no_image_with_refusal": "Claude without vision",
        "4o_mcq_image_with_refusal": "GPT-4o with vision",
        "4o_mcq_no_image_with_refusal": "GPT-4o without vision",
    }

    r2 = {
        "claude_mcq_image_without_refusal": "Claude without Refusal Option",
        "4o_mcq_image_without_refusal": "GPT-4o without Refusal Option",
        "claude_mcq_image_with_refusal": "Claude with Refusal Option",
        "4o_mcq_image_with_refusal": "GPT-4o with Refusal Option",
    }

    # Plot with vision and without vision
    plotting_utils.majority_vote_accuracy_by_k(
        {value: run_results[key] for key, value in r1.items()}, name="image_comparison"
    )

    # Plot with and without refusal option
    plotting_utils.majority_vote_accuracy_by_k(
        {value: run_results[key] for key, value in r2.items()},
        name="refusal_option_comparison",
    )


async def compare_capsule_mode(eval_df: pd.DataFrame) -> None:
    """
    Compare performance between different model architectures.

    DISCLAIMER: This function is highly tailored to the BixBench paper requirements.
    It is not designed to be used as a general function for comparing model performance.

    This function analyzes and visualizes the performance differences between
    GPT-4o and Claude models across different question formats.
    """
    # Define model names for clarity
    model1, model2 = "gpt-4o", "claude-3-5-sonnet"

    # Prepare data
    eval_df["format"] = eval_df["run_name"].apply(
        lambda x: (
            "open"
            if "open" in x
            else ("mcq_with_refusal" if "with_refusal" in x else "mcq_without_refusal")
        )
    )
    eval_df["model"] = eval_df["run_name"].apply(
        lambda x: model1 if "4o" in x else model2
    )
    eval_df = eval_df[~eval_df.run_name.str.contains("no_image")]

    # Calculate means and confidence intervals
    results = calculate_results(eval_df)

    # Plot results
    plotting_utils.plot_model_comparison(results, model1, model2)


def calculate_results(df: pd.DataFrame) -> list[dict]:
    """
    Calculate means and confidence intervals for each model and format.

    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results

    Returns:
        list: List of dictionaries containing statistical results for each model and format
    """
    results = []
    for model in df["model"].unique():
        for fmt in ["open", "mcq_with_refusal", "mcq_without_refusal"]:
            mask = (df["model"] == model) & (df["format"] == fmt)
            scores = df[mask]["correct"]
            if len(scores) > 0:
                mean = scores.mean()
                n = len(scores)
                ci_low, ci_high = utils.wilson_ci(mean, n)
                results.append({
                    "model": model,
                    "format": fmt,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })
    return results


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

    # Load raw trajectory data
    # os.makedirs("bixbench_results", exist_ok=True)
    # data = load_raw_data(args.data_path)

    # # Process trajectories and save eval df
    # eval_df = asyncio.run(process_trajectories(data, checkpointing=args.checkpointing))

    if args.checkpointing:
        eval_df = pd.read_csv("bixbench_results/eval_df.csv")
        eval_df["correct"] = eval_df["correct"].astype(bool)

    # Run majority vote
    asyncio.run(run_majority_vote(eval_df, k_value=10))
    # Compare capsule mode
    asyncio.run(compare_capsule_mode(eval_df))
