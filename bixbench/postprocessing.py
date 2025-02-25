import pandas as pd
import postprocessing_utils as utils
import asyncio
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

pd.options.mode.chained_assignment = None


async def main():
    df = pd.read_csv("raw_data.csv")
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
    open_eval_df.to_csv("open_eval_df1.csv", index=False)
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
    insufficient_eval_df.to_csv("insufficient_eval_df1.csv", index=False)
    no_insufficient_eval_df.to_csv("no_insufficient_eval_df1.csv", index=False)


async def main2():
    insufficient_eval_df = pd.read_csv("insufficient_eval_df1.csv")
    no_insufficient_eval_df = pd.read_csv("no_insufficient_eval_df1.csv")

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
        k_values, means, stds = utils.run_majority_voting(grouped_df, range(1, 3), 3)
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

    utils.majority_vote_accuracy_by_k(
        {r1[k]: run_results[k] for k in r1}, name="image_comparison"
    )
    utils.majority_vote_accuracy_by_k(
        {r2[k]: run_results[k] for k in r2}, name="insufficient_option_comparison"
    )


async def main3():
    # Load data
    dfs = {
        "insufficient": pd.read_csv("insufficient_eval_df1.csv"),
        "no_insufficient": pd.read_csv("no_insufficient_eval_df1.csv"),
        "open": pd.read_csv("open_eval_df1.csv"),
    }

    # Define model names for clarity
    model1, model2 = "gpt-4o", "claude-3-5-sonnet"

    # Prepare data
    tmp = pd.concat(
        [dfs["insufficient"].copy(), dfs["no_insufficient"].copy(), dfs["open"].copy()]
    )
    tmp["format"] = tmp["run_name"].apply(
        lambda x: "open"
        if "open" in x
        else (
            "mcq_with_insufficient"
            if "with_insufficient" in x
            else "mcq_without_insufficient"
        )
    )
    tmp["model"] = tmp["run_name"].apply(lambda x: model1 if "4o" in x else model2)
    tmp = tmp[~tmp.run_name.str.contains("no_image")]

    # Calculate means and confidence intervals
    results = calculate_results(tmp)

    # Plot results
    plot_model_comparison(results, model1, model2)


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


def plot_model_comparison(results, model1, model2):
    """Create a bar chart comparing model performance across different formats."""
    # Setup
    plt.figure(figsize=(12, 6))
    barWidth = 0.35
    formats = ["open", "mcq_with_insufficient", "mcq_without_insufficient"]
    x = np.arange(len(formats))
    colors = {model1: "orange", model2: "#b3d9f2"}

    # Define baselines
    baselines = {
        "claude-3-5-sonnet-latest-grader-openended": 0.11486486486486487,
        "gpt-4o-grader-openended": 0.09121621621621621,
        "claude-3-5-sonnet-latest-grader-mcq-refusal-True": 0.13851351351351351,
        "gpt-4o-grader-mcq-refusal-True": 0.10810810810810811,
        "claude-3-5-sonnet-latest-grader-mcq-refusal-False": 0.33783783783783783,
        "gpt-4o-grader-mcq-refusal-False": 0.32094594594594594,
        "random w/ refusal": 0.2,
        "random w/o refusal": 0.25,
    }

    # Draw baseline lines
    draw_baselines(x, baselines, barWidth)

    # Draw model performance bars
    draw_model_bars(x, results, barWidth, formats, colors, model1, model2)

    # Customize plot appearance
    plt.ylabel("Accuracy")
    plt.title("Model Performance by Question Format with Wilson CI @95%")
    plt.xticks(x + barWidth / 2, ["Open-ended", "MCQ w/ refusal", "MCQ w/o refusal"])

    # Create legend with proper order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0]  # This will put Claude first, then GPT-4
    handles = [handles[idx] for idx in order]
    labels = [labels[idx] for idx in order]
    plt.legend(handles, labels)

    # Add grid and display
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def draw_baselines(x, baselines, barWidth):
    """Draw baseline lines on the plot."""
    baseline_color = "grey"
    random_color = "grey"
    line_width = 2
    extension = 0.05
    half_bar = barWidth / 2
    baseline_bar = "-"

    # Define baseline positions
    baseline_positions = [
        # Format: (baseline_key, x_position, width_offset)
        ("claude-3-5-sonnet-latest-grader-openended", x[0], 0),
        ("gpt-4o-grader-openended", x[0], barWidth),
        ("claude-3-5-sonnet-latest-grader-mcq-refusal-True", x[1], 0),
        ("gpt-4o-grader-mcq-refusal-True", x[1], barWidth),
        ("claude-3-5-sonnet-latest-grader-mcq-refusal-False", x[2], 0),
        ("gpt-4o-grader-mcq-refusal-False", x[2], barWidth),
    ]

    # Draw model baselines
    for baseline_key, x_pos, width_offset in baseline_positions:
        plt.hlines(
            y=baselines[baseline_key],
            xmin=x_pos - extension - half_bar + width_offset,
            xmax=x_pos + barWidth + extension - half_bar + width_offset,
            color=baseline_color,
            linestyle=baseline_bar,
            linewidth=line_width,
            label="baseline"
            if baseline_key == "claude-3-5-sonnet-latest-grader-openended"
            else "",
        )

    # Draw random baselines
    plt.hlines(
        y=baselines["random w/ refusal"],
        xmin=x[1] - extension - half_bar,
        xmax=x[1] + 2 * barWidth + extension - half_bar,
        color=random_color,
        linestyle="--",
        linewidth=line_width,
        label="random",
    )

    plt.hlines(
        y=baselines["random w/o refusal"],
        xmin=x[2] - extension - half_bar,
        xmax=x[2] + 2 * barWidth + extension - half_bar,
        color=random_color,
        linestyle="--",
        linewidth=line_width,
    )


def draw_model_bars(x, results, barWidth, formats, colors, model1, model2):
    """Draw performance bars for each model."""
    for i, model in enumerate([model1, model2]):
        model_results = [r for r in results if r["model"] == model]
        means = [
            next((r["mean"] for r in model_results if r["format"] == fmt), 0)
            for fmt in formats
        ]
        ci_lows = [
            next((r["ci_low"] for r in model_results if r["format"] == fmt), 0)
            for fmt in formats
        ]
        ci_highs = [
            next((r["ci_high"] for r in model_results if r["format"] == fmt), 0)
            for fmt in formats
        ]

        yerr = np.array(
            [
                [m - l for m, l in zip(means, ci_lows)],
                [h - m for m, h in zip(means, ci_highs)],
            ]
        )

        plt.bar(
            x + i * barWidth,
            means,
            barWidth,
            label=model,
            color=colors[model],
            alpha=0.5 if model == model1 else 1,
            yerr=yerr,
            capsize=5,
        )


# asyncio.run(main())
asyncio.run(main3())
