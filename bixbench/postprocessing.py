# Standard library imports
import asyncio
import ast

# Third-party imports
import pandas as pd

# Local imports
import postprocessing_utils as utils
import plotting_utils

pd.options.mode.chained_assignment = None


async def grade_outputs():
    df = pd.read_csv("bixbench_results/raw_data.csv")
    df["md_images"] = df["md_images"].apply(ast.literal_eval)

    # Create eval df
    eval_df = utils.create_eval_df(df)

    insufficient_map = {
        "claude_open_no_image": "claude_mcq_no_image_with_insufficient",
        "4o_open_no_image": "4o_mcq_no_image_with_insufficient",
        "claude_open_image": "claude_mcq_image_with_insufficient",
        "4o_open_image": "4o_mcq_image_with_insufficient",
    }

    no_insufficient_map = {
        "claude_open_no_image": "claude_mcq_no_image_no_insufficient",
        "4o_open_no_image": "4o_mcq_no_image_no_insufficient",
        "claude_open_image": "claude_mcq_image_no_insufficient",
        "4o_open_image": "4o_mcq_image_no_insufficient",
    }

    open_eval_df = utils.update_eval_df(eval_df, insufficient=False, run_map=None)
    insufficient_eval_df = utils.update_eval_df(
        eval_df, insufficient=True, run_map=insufficient_map
    )
    no_insufficient_eval_df = utils.update_eval_df(
        eval_df, insufficient=False, run_map=no_insufficient_map
    )

    print(eval_df.run_name.unique())
    print(insufficient_eval_df.run_name.unique())
    print(no_insufficient_eval_df.run_name.unique())

    # OPEN ENDED
    batch = open_eval_df["prompt"].tolist()
    results = await utils.process_batch(batch, "gpt-4o", max_concurrent=100)
    open_eval_df["correct"] = results
    open_eval_df["correct"] = open_eval_df.correct.apply(lambda x: 1 if x == "1" else 0)
    open_eval_df.to_csv("bixbench_results/open_eval_df.csv", index=False)
    print(open_eval_df.groupby("run_name").correct.mean())

    # MCQ
    insufficient_eval_df = await utils.run_mcq_eval_loop(insufficient_eval_df)
    no_insufficient_eval_df = await utils.run_mcq_eval_loop(no_insufficient_eval_df)

    insufficient_eval_df["agent_mcq_answer"] = insufficient_eval_df[
        "agent_mcq_answer"
    ].apply(utils.xml_extract)
    insufficient_eval_df["correct"] = (
        insufficient_eval_df["agent_mcq_answer"]
        == insufficient_eval_df["correct_letter"]
    )
    no_insufficient_eval_df["agent_mcq_answer"] = no_insufficient_eval_df[
        "agent_mcq_answer"
    ].apply(utils.xml_extract)
    no_insufficient_eval_df["correct"] = (
        no_insufficient_eval_df["agent_mcq_answer"]
        == no_insufficient_eval_df["correct_letter"]
    )
    insufficient_eval_df.to_csv(
        "bixbench_results/insufficient_eval_df.csv", index=False
    )
    no_insufficient_eval_df.to_csv(
        "bixbench_results/no_insufficient_eval_df.csv", index=False
    )


async def run_majority_vote():
    insufficient_eval_df = pd.read_csv("bixbench_results/insufficient_eval_df.csv")
    no_insufficient_eval_df = pd.read_csv(
        "bixbench_results/no_insufficient_eval_df.csv"
    )

    # Majority vote by run_name
    maj_vote_df = pd.concat(
        [insufficient_eval_df.copy(), no_insufficient_eval_df.copy()]
    )
    maj_vote_df = maj_vote_df[maj_vote_df.run_name.str.contains("mcq")]

    # Store results for all runs
    run_results = {}

    for run_name in maj_vote_df.run_name.unique():
        print("RUN NAME", run_name)
        grouped_df = maj_vote_df[maj_vote_df.run_name == run_name].copy()
        grouped_df["agent_mcq_answer"] = grouped_df["agent_mcq_answer"].fillna("X")
        grouped_df = grouped_df.groupby("uuid").agg(list)
        grouped_df["correct_letter"] = grouped_df["correct_letter"].apply(
            lambda x: x[0]
        )
        grouped_df = grouped_df.dropna()
        k_values, means, stds = utils.run_majority_voting(grouped_df, range(1, 10), 10)
        run_results[run_name] = (k_values, means, stds)

    r1 = {
        "claude_mcq_image_with_insufficient": "Claude with vision",
        "claude_mcq_no_image_with_insufficient": "Claude without vision",
        "4o_mcq_image_with_insufficient": "GPT-4o with vision",
        "4o_mcq_no_image_with_insufficient": "GPT-4o without vision",
    }

    r2 = {
        "claude_mcq_image_no_insufficient": "Claude without Insufficient Option",
        "4o_mcq_image_no_insufficient": "GPT-4o without Insufficient Option",
        "claude_mcq_image_with_insufficient": "Claude with Insufficient Option",
        "4o_mcq_image_with_insufficient": "GPT-4o with Insufficient Option",
    }

    plotting_utils.majority_vote_accuracy_by_k(
        {value: run_results[key] for key, value in r1.items()}, name="image_comparison"
    )
    plotting_utils.majority_vote_accuracy_by_k(
        {value: run_results[key] for key, value in r2.items()},
        name="insufficient_option_comparison",
    )


async def compare_capsule_mode():
    # Load data
    dfs = {
        "insufficient": pd.read_csv("bixbench_results/insufficient_eval_df.csv"),
        "no_insufficient": pd.read_csv("bixbench_results/no_insufficient_eval_df.csv"),
        "open": pd.read_csv("bixbench_results/open_eval_df.csv"),
    }

    # Define model names for clarity
    model1, model2 = "gpt-4o", "claude-3-5-sonnet"

    # Prepare data
    tmp = pd.concat(
        [dfs["insufficient"].copy(), dfs["no_insufficient"].copy(), dfs["open"].copy()]
    )
    tmp["format"] = tmp["run_name"].apply(
        lambda x: (
            "open"
            if "open" in x
            else (
                "mcq_with_insufficient"
                if "with_insufficient" in x
                else "mcq_without_insufficient"
            )
        )
    )
    tmp["model"] = tmp["run_name"].apply(lambda x: model1 if "4o" in x else model2)
    tmp = tmp[~tmp.run_name.str.contains("no_image")]

    # Calculate means and confidence intervals
    results = calculate_results(tmp)

    # Plot results
    plotting_utils.plot_model_comparison(results, model1, model2)


def calculate_results(df):
    """Calculate means and confidence intervals for each model and format."""
    results = []
    for model in df["model"].unique():
        for fmt in ["open", "mcq_with_insufficient", "mcq_without_insufficient"]:
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


asyncio.run(grade_outputs())
asyncio.run(run_majority_vote())
asyncio.run(compare_capsule_mode())
