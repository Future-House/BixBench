import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from plot_style import set_fh_mpl_style

set_fh_mpl_style()


def majority_vote_accuracy_by_k(
    run_results: dict, style: str = "dark", name=""
) -> None:
    plt.figure(figsize=(15, 6))

    for run_name, (k_values, means, stds) in run_results.items():
        print(k_values, means, stds)
        if k_values is None:
            continue
        plt.plot(k_values, means, "o-", label=run_name)
        plt.fill_between(
            k_values,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
        )

    plt.xlabel("Number of Votes (k)")
    plt.ylabel("Accuracy")
    plt.xlim(1, 9)
    plt.title("Majority Voting Accuracy vs Number of Votes")
    plt.xticks(k_values)
    if name == "image_comparison":
        plt.axhline(y=0.2, color="red", linestyle=":", label="Random Guess")
    else:
        plt.axhline(
            y=0.2,
            color="red",
            linestyle=":",
            label="With Insufficient Option Random Guess",
        )
        plt.axhline(
            y=0.25,
            color="green",
            linestyle=":",
            label="Without Insufficient Option Random Guess",
        )
    plt.legend()  # bbox_to_anchor=(1.05, 0), loc='lower left')
    plt.grid(True, alpha=0.3)
    #todo: avoid hardcoding out paths or make this an optional parameter 
    plt.savefig(f"bixbench_results/majority_vote_accuracy_{name}.png")
    plt.show()


def plot_model_comparison(results, model1, model2):
    """Create a bar chart comparing model performance across different formats."""
    # Setup
    plt.figure(figsize=(12, 6))
    barWidth = 0.35
    formats = ["open", "mcq_with_insufficient", "mcq_without_insufficient"]
    x = np.arange(len(formats))
    colors = {model1: "orange", model2: "#b3d9f2"}

    # Load baselines from JSON file
    #todo: avoid hardcoding out paths or make this an optional parameter 
    with open("bixbench_results/zero_shot_baselines.json", "r") as f:
        baselines = json.load(f)
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
    # fix: name is not declared
    plt.savefig(f"bixbench_results/model_comparison_{name}.png")
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
            label=(
                "baseline"
                if baseline_key == "claude-3-5-sonnet-latest-grader-openended"
                else ""
            ),
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
