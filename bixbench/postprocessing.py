import argparse
import ast
import asyncio
import contextlib
import json
import operator
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fhda.utils import view_notebook

from bixbench import plotting_utils
from bixbench import postprocessing_utils as utils
from bixbench.models import (
    MajorityVoteConfig,
    PostprocessingConfig,
    RunComparisonConfig,
)

pd.options.mode.chained_assignment = None
# If true, save and load intermediate results to avoid re-running the same steps


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file or directory of json files and process specific columns.

    If the path is a directory, all json files in the directory will be loaded into a dataframe.

    Args:
        path (str): Path to the CSV file or directory containing raw data

    Returns:
        pd.DataFrame: Processed DataFrame with converted column types
    """
    raw_data = (
        utils.load_dataframe_from_json_directory(path)
        if Path(path).is_dir()
        else pd.read_csv(path)
    )

    mapping = {
        "agent_answer": utils.load_answer,
        "ideal_answer": utils.load_answer,
        "mcq_options": ast.literal_eval,
        "mcq_question": ast.literal_eval,
        "nb": utils.load_notebook,
        "avoid_images": bool,
        "actions": int,
        "refusal_option": bool,
    }
    for col, func in mapping.items():
        if col in raw_data.columns:
            with contextlib.suppress(ValueError):
                raw_data[col] = raw_data[col].apply(func)

    # Convert json notebook to markdown for postprocessing
    if "nb" in raw_data.columns and "nb_md" not in raw_data.columns:
        df_md = pd.DataFrame(
            raw_data["nb"].apply(lambda x: view_notebook(x.cells, "python")).tolist(),
            columns=["md_notebook", "md_images"],
        )
        raw_data[["md_notebook", "md_images"]] = df_md
    return raw_data


async def process_trajectories(df: pd.DataFrame, max_concurrent: int = 5) -> pd.DataFrame:
    """
    Create a gradable dataframe from a raw dataframe of trajectories.

    This function processes the raw data, runs evaluation loops, and saves
    the results to CSV files for further analysis.

    Args:
        df (pd.DataFrame): Raw data containing model trajectories
        max_concurrent (int): Maximum number of concurrent API calls to make

    Returns:
        pd.DataFrame: Processed evaluation dataframe with graded responses
    """
    eval_df = utils.create_eval_df(df)
    eval_df = await utils.run_eval_loop(eval_df, max_concurrent=max_concurrent)

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

    return eval_df


async def run_majority_vote(
    eval_df: pd.DataFrame, config: MajorityVoteConfig, results_dir: str
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """
    Implement majority voting evaluation across different model configurations.

    This function analyzes evaluation data, performs majority voting analysis for
    multiple choice questions, and produces visualizations comparing different model
    configurations with various features.

    Args:
        eval_df (pd.DataFrame): DataFrame containing evaluation results
        config (MajorityVoteConfig): Configuration for majority voting
        results_dir (str): Directory to save results

    Returns:
        Dict[str, Tuple[List[int], List[float], List[float]]]: Dictionary mapping run names to
            tuples of (k_values, mean accuracies, standard deviations)
    """
    # Only run majority vote on mcq questions
    maj_vote_df = eval_df[eval_df.question_format == "mcq"].copy()

    if maj_vote_df.empty:
        print("No MCQ questions found, skipping majority vote")
        return {}

    # Get configuration values
    k_value = config.k_value
    mv_groups = config.groups

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

    # Plot results for each group if specified in config
    for group_name, group_runs in mv_groups.items():
        # Filter run_results to only include runs specified in the group
        filtered_results = {
            run_name: run_results[run_name]
            for run_name in group_runs
            if run_name in run_results
        }

        # Determine random baselines
        random_baselines = [0.2]  # Default with refusal option
        random_baselines_labels = ["Random Guess with Refusal Option"]

        if any("without_refusal" in run_name for run_name in group_runs):
            random_baselines.append(0.25)
            random_baselines_labels.append("Random Guess without Refusal Option")

        plotting_utils.majority_vote_accuracy_by_k(
            filtered_results,
            name=group_name,
            random_baselines=random_baselines,
            random_baselines_labels=random_baselines_labels,
            results_dir=results_dir,
            save_to_root=False,  # Changed to False to prevent overwriting in root directory
        )

    return run_results


async def compare_runs(
    eval_df: pd.DataFrame,
    config: RunComparisonConfig,
    results_dir: str,
    replicate_paper_results: bool,
) -> dict[str, dict[str, Any]]:
    """
    Compare performance between different model architectures.

    This function analyzes and visualizes the performance differences between
    various model configurations across different question formats.

    Args:
        eval_df (pd.DataFrame): DataFrame containing evaluation results
        config (RunComparisonConfig): Configuration for run comparison
        results_dir (str): Directory to save results
        replicate_paper_results (bool): Whether to use the same plots from the paper

    Returns:
        Dict[str, Dict[str, Any]]: Performance results for different model configurations
    """
    run_name_groups = config.run_name_groups
    group_titles = config.group_titles
    color_groups = config.color_groups
    total_questions_per_run = config.total_questions_per_run

    # Get baselines configuration
    baselines = {}
    if config.use_zero_shot_baselines and os.path.exists(
        f"{results_dir}/zero_shot_baselines.json"
    ):
        with open(f"{results_dir}/zero_shot_baselines.json", encoding="utf-8") as f:
            baselines_data = json.load(f)

        baseline_mappings = config.baseline_name_mappings
        baselines = {
            baseline_mappings.get(k, k): v["accuracy"]
            for k, v in baselines_data.items()
        }

    # Get random baselines
    random_baselines = config.random_baselines

    # Filter eval_df to only include run_names configured
    flat_run_names = utils.flatten_list(run_name_groups)
    eval_df = eval_df[eval_df["run_name"].isin(flat_run_names)]

    # Calculate means and confidence intervals
    results = utils.calculate_results(
        eval_df, total_questions_per_run=total_questions_per_run
    )
    # Print results in a more readable format
    for run_name, metrics in results.items():
        mean_percent = metrics['mean'] * 100
        ci_low_percent = metrics['ci_low'] * 100
        ci_high_percent = metrics['ci_high'] * 100
        print(f"{run_name}: {mean_percent:.2f}% (CI: {ci_low_percent:.2f}%-{ci_high_percent:.2f}%)")

    if replicate_paper_results:
        # Plot results using the detailed paper plotting
        plotting_utils.plot_model_comparison(
            results,
            baselines,
            run_name_groups,
            color_groups,
            group_titles=group_titles,
            random_baselines=random_baselines,
            results_dir=results_dir,
            save_to_root=False,  # Changed to False to prevent overwriting in root directory
        )
    else:
        # Use simplified plotting
        plotting_utils.plot_simplified_comparison(
            results,
            run_name_groups,
            group_titles=group_titles,
            has_mcq=any("mcq" in run for run in flat_run_names),
            results_dir=results_dir,
            save_to_root=False,  # Changed to False to prevent overwriting in root directory
        )

    return results


async def load_or_process_data(config, max_concurrent: int = 5) -> pd.DataFrame:
    """
    Load data from files or process trajectories based on configuration.

    Args:
        config: Configuration object with data paths and processing options
        max_concurrent (int): Maximum number of concurrent API calls to make

    Returns:
        pd.DataFrame: Processed evaluation dataframe
    """
    results_dir = config.results_dir
    data_path = config.data_path
    replicate_paper_results = config.replicate_paper_results

    # Case 1: Replicating paper results from trajectories
    if replicate_paper_results.run and replicate_paper_results.from_trajectories:
        trajectory_path = f"{results_dir}/raw_trajectory_data.csv"
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(
                f"raw_trajectory_data.csv not found in {results_dir}, "
                "please follow the readme to download the raw trajectory data"
            )
        data = load_raw_data(trajectory_path)
        return await process_trajectories(data, max_concurrent=max_concurrent)

    # Case 2: Replicating paper results from pre-computed eval_df
    if replicate_paper_results.run and (not replicate_paper_results.from_trajectories):
        eval_df_path = f"{results_dir}/eval_df.csv"
        if not os.path.exists(eval_df_path):
            raise FileNotFoundError(
                f"eval_df.csv not found in {results_dir}, "
                "please follow the readme to download the eval_df.csv"
            )
        eval_df = pd.read_csv(eval_df_path)
        eval_df["correct"] = eval_df["correct"].astype(bool)
        return eval_df

    # Case 3: Running new analysis from raw data
    data = load_raw_data(data_path)
    return await process_trajectories(data, max_concurrent=max_concurrent)


async def main(config_path: str, max_concurrent: int = 5):
    """
    Main function to run BixBench postprocessing based on YAML configuration.

    Args:
        config_path (str): Path to the YAML configuration file
        max_concurrent (int): Maximum number of concurrent API calls to make
    """
    # Load configuration
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Parse configuration with Pydantic
    config = PostprocessingConfig.model_validate(config_dict)

    # Set up results directory
    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Load or process data based on configuration
    eval_df = await load_or_process_data(config, max_concurrent=max_concurrent)

    # Save intermediary processed data for debugging
    if config.debug | (config.replicate_paper_results.from_trajectories):
        eval_df.to_csv(f"{results_dir}/eval_df_new.csv", index=False)

    # Run majority vote if configured
    if config.majority_vote.run:
        await run_majority_vote(eval_df, config.majority_vote, results_dir)

    # Run comparison if configured
    if config.run_comparison.run:
        await compare_runs(
            eval_df,
            config.run_comparison,
            results_dir,
            config.replicate_paper_results.run,
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process BixBench evaluation data")
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=5, 
        help="Maximum number of concurrent API calls to make (default: 5)"
    )
    parser.add_argument(
        "--use-new-structure", action="store_true",
        help="Use the new folder structure (runs/run_id/...)"
    )
    parser.add_argument(
        "--run-id", type=str,
        help="Run ID for the new folder structure (only used with --use-new-structure)"
    )

    args = parser.parse_args()
    
    # If using new structure, we need to modify the config after loading
    async def main_with_new_structure():
        # Load configuration
        with open(args.config_file, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # Update config for new structure if requested
        if args.use_new_structure:
            config_dict["use_new_structure"] = True
            if args.run_id:
                config_dict["run_id"] = args.run_id
        
        # Parse configuration with Pydantic
        config = PostprocessingConfig.model_validate(config_dict)
        
        # Set up results directory
        results_dir = config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Load or process data based on configuration
        eval_df = await load_or_process_data(config, max_concurrent=args.max_concurrent)
        
        # Save intermediary processed data for debugging
        if config.debug | (config.replicate_paper_results.from_trajectories):
            eval_df.to_csv(f"{results_dir}/eval_df_new.csv", index=False)
        
        # Run majority vote if configured
        if config.majority_vote.run:
            await run_majority_vote(eval_df, config.majority_vote, results_dir)
        
        # Run comparison if configured
        if config.run_comparison.run:
            await compare_runs(
                eval_df,
                config.run_comparison,
                results_dir,
                config.replicate_paper_results.run,
            )
    
    # Run with support for new structure
    if args.use_new_structure:
        asyncio.run(main_with_new_structure())
    else:
        # Run main function with the config file and max_concurrent parameter
        asyncio.run(main(args.config_file, args.max_concurrent))
