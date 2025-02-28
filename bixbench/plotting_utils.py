# DISCLAIMER: This file is highly tailored to the BixBench paper requirements.
# It is not designed to be used as a general function for plotting model performance.

import json

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
from plot_style import set_fh_mpl_style

set_fh_mpl_style()


def majority_vote_accuracy_by_k(
    run_results: dict,
    name="",
    random_baselines: list[float] | None = None,
    random_baselines_labels: list[str] | None = None,
) -> None:
    if random_baselines_labels is None:
        random_baselines_labels = ["With Refusal Option Random Guess", "Without Refusal Option Random Guess"]
    if random_baselines is None:
        random_baselines = [0.2, 0.25]
    plt.figure(figsize=(15, 6))

    for run_name, (k_values, means, stds) in run_results.items():
        print(k_values, means, stds)
        if k_values is None:
            continue
        plt.plot(k_values, means, "o-", label=run_name)
        plt.fill_between(
            k_values,
            [m - s for m, s in zip(means, stds, strict=True)],
            [m + s for m, s in zip(means, stds, strict=True)],
            alpha=0.2,
        )

    plt.xlabel("Number of Votes (k)")
    plt.ylabel("Accuracy")
    plt.xlim(1, 9)
    plt.title("Majority Voting Accuracy vs Number of Votes")
    plt.xticks(k_values)

    for i, (baseline, label) in enumerate(
        zip(random_baselines, random_baselines_labels, strict=True)
    ):
        plt.axhline(
            y=baseline,
            color="red" if i == 0 else "green",
            linestyle=":",
            label=label,
        )
    plt.legend()  # bbox_to_anchor=(1.05, 0), loc='lower left')
    plt.grid(alpha=0.3, visible=True)
    plt.savefig(f"{cfg.RESULTS_DIR}/majority_vote_accuracy_{name}.png")
    plt.show()


def plot_model_comparison(results, model1, model2):
    """Create a bar chart comparing model performance across different formats."""
    # Setup
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    formats = ["open", "mcq_with_refusal", "mcq_without_refusal"]
    x = np.arange(len(formats))

    # Update colors to match notebook
    color_cycle = ["#1BBC9B", "#FF8C00", "#FF69B4", "#ce8aed", "#80cedb", "#FFFFFF"]
    colors = {model1: color_cycle[0], model2: color_cycle[1]}

    # Load baselines from JSON file
    with open(f"{cfg.RESULTS_DIR}/zero_shot_baselines.json", encoding="utf-8") as f:
        baselines = json.load(f)
    baselines = {k: v["accuracy"] for k, v in baselines.items()}
    baselines["random w/ refusal"] = 0.2
    baselines["random w/o refusal"] = 0.25
    # Draw baseline lines
    draw_baselines(x, baselines, bar_width)

    # Draw model performance bars
    draw_model_bars(x, results, bar_width, formats, colors, model1, model2)

    # Customize plot appearance
    plt.ylabel("Accuracy", fontsize=18)
    plt.yticks(np.arange(0, 0.5, 0.1), fontsize=18)
    plt.title("Model Performance by Question Format with Wilson CI @95%")
    plt.xticks(
        x + bar_width / 2,
        ["Open-answer", "MCQ w/ refusal", "MCQ w/o refusal"],
        fontsize=18,
    )

    # Create legend with proper order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0]  # This will put Claude first, then GPT-4
    handles = [handles[idx] for idx in order]
    labels = [labels[idx] for idx in order]
    plt.legend(handles, labels)

    # Add grid and display
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{cfg.RESULTS_DIR}/bixbench_results_format_comparison.png")
    plt.show()


def draw_baselines(x, baselines, bar_width):
    """Draw baseline lines on the plot."""
    baseline_color = "grey"
    random_color = "grey"
    line_width = 2
    extension = 0.05
    half_bar = bar_width / 2
    baseline_bar = "-"

    # Define baseline positions
    baseline_positions = [
        # Format: (baseline_key, x_position, width_offset)
        ("gpt-4o-grader-openended", x[0], 0),
        ("claude-3-5-sonnet-latest-grader-openended", x[0], bar_width),
        ("gpt-4o-grader-mcq-refusal-True", x[1], 0),
        ("claude-3-5-sonnet-latest-grader-mcq-refusal-True", x[1], bar_width),
        ("gpt-4o-grader-mcq-refusal-False", x[2], 0),
        ("claude-3-5-sonnet-latest-grader-mcq-refusal-False", x[2], bar_width),
    ]

    # Draw model baselines
    for baseline_key, x_pos, width_offset in baseline_positions:
        plt.hlines(
            y=baselines[baseline_key],
            xmin=x_pos - extension - half_bar + width_offset,
            xmax=x_pos + bar_width + extension - half_bar + width_offset,
            color=baseline_color,
            linestyle=baseline_bar,
            linewidth=line_width,
            label="baseline" if baseline_key == "gpt-4o-grader-openended" else "",
        )

    # Draw random baselines
    plt.hlines(
        y=baselines["random w/ refusal"],
        xmin=x[1] - extension - half_bar,
        xmax=x[1] + 2 * bar_width + extension - half_bar,
        color=random_color,
        linestyle="--",
        linewidth=line_width,
        label="random",
    )

    plt.hlines(
        y=baselines["random w/o refusal"],
        xmin=x[2] - extension - half_bar,
        xmax=x[2] + 2 * bar_width + extension - half_bar,
        color=random_color,
        linestyle="--",
        linewidth=line_width,
    )


def draw_model_bars(x, results, bar_width, formats, colors, model1, model2):
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

        yerr = np.array([
            [m - low for m, low in zip(means, ci_lows, strict=True)],
            [h - m for m, h in zip(means, ci_highs, strict=True)],
        ])

        plt.bar(
            x + i * bar_width,
            means,
            bar_width,
            label=model,
            color=colors[model],
            alpha=0.5 if model == model1 else 1,
            yerr=yerr,
            capsize=5,
        )
