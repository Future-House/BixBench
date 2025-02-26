import asyncio
import ast
import pandas as pd
import nbformat
import json

from fhda.utils import view_notebook
import postprocessing_utils as utils
import plotting_utils

pd.options.mode.chained_assignment = None

def load_raw_data(path: str):
    """
    Load raw data from a CSV file and process specific columns.
    
    Args:
        path (str): Path to the CSV file containing raw data
        
    Returns:
        pd.DataFrame: Processed DataFrame with converted column types
    """
    print("Loading raw data from", path)
    df = pd.read_csv(path)
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
        if col in df.columns:
            df[col] = df[col].apply(func)

    # Convert json notebook to markdown for postprocessing
    if "nb" in df.columns and not "nb_md" in df.columns:
        df_md = pd.DataFrame(
            df["nb"].apply(lambda x: view_notebook(x.cells, "python")).tolist(),
            columns=["md_notebook", "md_images"],
        )
        df[["md_notebook", "md_images"]] = df_md
    print("Loaded raw data from", path)
    return df


async def process_trajectories(df: pd.DataFrame):
    """
    Create a gradable dataframe from a raw dataframe of trajectories.
    
    This function processes the raw data, runs evaluation loops, and saves
    the results to CSV files for further analysis.
    
    Args:
        df (pd.DataFrame): Raw data containing model trajectories
    """
    print("Creating eval df")
    eval_df = utils.create_eval_df(df)
    print("Running eval loop")
    eval_df = await utils.run_eval_loop(eval_df)
    print("Running MCQ eval loop")

    eval_df.to_csv("bixbench_results/eval_loop_results.csv", index=False)
    # Create correct column for open ended questions
    eval_df.loc[eval_df.question_format == "open", "correct"] = eval_df.loc[
        eval_df.question_format == "open", "llm_answer"
    ].apply(lambda x: True if x == "1" else False)
    # Extract XML from LLM MCQ answers
    eval_df.loc[eval_df.question_format == "mcq", "llm_answer"] = eval_df.loc[
        eval_df.question_format == "mcq", "llm_answer"
    ].apply(utils.xml_extract)
    # Compare LLM answers to ideal answers
    eval_df.loc[eval_df.question_format == "mcq", "correct"] = (
        eval_df.loc[eval_df.question_format == "mcq", "llm_answer"]
        == eval_df.loc[eval_df.question_format == "mcq", "correct_letter"]
    )
    print("Grouping by run name")
    print(eval_df.groupby("run_name").correct.mean())
    eval_df.to_csv("bixbench_results/all_eval_df.csv", index=False)


async def run_majority_vote():
    """
    Implement majority voting evaluation across different model configurations.
    
    This function reads evaluation data, performs majority voting analysis for
    multiple choice questions, and produces visualization comparing different model
    configurations with and without specific features.
    """
    eval_df = pd.read_csv("bixbench_results/all_eval_df.csv")

    # Config
    k_value = 3

    maj_vote_df = eval_df[eval_df.question_format == "mcq"].copy()

    # Store results for all runs
    run_results = {}

    for run_name in maj_vote_df.run_name.unique():
        print("RUN NAME", run_name)
        grouped_df = maj_vote_df[maj_vote_df.run_name == run_name].copy()
        grouped_df["llm_answer"] = grouped_df["llm_answer"].fillna("X")
        grouped_df = grouped_df.groupby("uuid").agg(list)
        grouped_df["correct_letter"] = grouped_df["correct_letter"].apply(
            lambda x: x[0]
        )
        grouped_df = grouped_df.dropna()
        k_values, means, stds = utils.run_majority_voting(
            grouped_df, range(1, k_value), k_value
        )
        run_results[run_name] = (k_values, means, stds)
    print(run_results)
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


async def compare_capsule_mode():
    """
    Compare performance between different model architectures.
    
    This function analyzes and visualizes the performance differences between
    GPT-4o and Claude models across different question formats.
    """
    # Define model names for clarity
    model1, model2 = "gpt-4o", "claude-3-5-sonnet"

    # Prepare data
    tmp = pd.read_csv("bixbench_results/all_eval_df.csv")
    tmp["correct"] = tmp["correct"].astype(bool)

    tmp["format"] = tmp["run_name"].apply(
        lambda x: (
            "open"
            if "open" in x
            else ("mcq_with_refusal" if "with_refusal" in x else "mcq_without_refusal")
        )
    )
    tmp["model"] = tmp["run_name"].apply(lambda x: model1 if "4o" in x else model2)
    tmp = tmp[~tmp.run_name.str.contains("no_image")]

    # Calculate means and confidence intervals
    results = calculate_results(tmp)
    print(results)

    # Plot results
    plotting_utils.plot_model_comparison(results, model1, model2)


def calculate_results(df):
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
                results.append(
                    {
                        "model": model,
                        "format": fmt,
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return results


async def compare_capsule_mode_with_refusal():
    """
    Compare models with refusal mode enabled.
    
    This function loads evaluation data, processes it to compare how different models
    perform when the refusal option is available, and visualizes the results.
    """
    # Define model names for clarity
    model1, model2 = "gpt-4o", "claude-3-5-sonnet"

    # Prepare data
    tmp = pd.read_csv("bixbench_results/all_eval_df.csv")
    tmp["correct"] = tmp["correct"].astype(bool)

    # Filter to include only runs with refusal option enabled
    tmp = tmp[tmp.run_name.str.contains("with_refusal")]
    
    tmp["model"] = tmp["run_name"].apply(lambda x: model1 if "4o" in x else model2)
    tmp["vision"] = tmp["run_name"].apply(lambda x: "With Vision" if "image" in x and "no_image" not in x else "Without Vision")
    
    # Calculate means and confidence intervals
    results = calculate_results_for_refusal(tmp)
    print(results)

    # Plot results
    plotting_utils.plot_refusal_comparison(results, model1, model2)


def calculate_results_for_refusal(df):
    """
    Calculate means and confidence intervals for refusal mode comparison.
    
    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results
        
    Returns:
        list: List of dictionaries containing statistical results for each model and vision mode
    """
    results = []
    for model in df["model"].unique():
        for vision in ["With Vision", "Without Vision"]:
            mask = (df["model"] == model) & (df["vision"] == vision)
            scores = df[mask]["correct"]
            if len(scores) > 0:
                mean = scores.mean()
                n = len(scores)
                ci_low, ci_high = utils.wilson_ci(mean, n)
                results.append(
                    {
                        "model": model,
                        "vision": vision,
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return results


if __name__ == "__main__":
    # Load raw trajectory data
    data = load_raw_data("bixbench_results/raw_trajectory_data.csv")
    asyncio.run(process_trajectories(data))
    asyncio.run(run_majority_vote())
    asyncio.run(compare_capsule_mode())
    asyncio.run(compare_capsule_mode_with_refusal())