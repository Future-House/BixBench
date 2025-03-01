# DISCLAIMER: This file is highly tailored to the BixBench paper requirements.
# It is not designed to be used as a general function for plotting model performance.

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
from plot_style import set_fh_mpl_style

from bixbench import postprocessing_utils as utils

set_fh_mpl_style()


def majority_vote_accuracy_by_k(
    run_results: dict,
    name="",
    random_baselines: list[float] | None = None,
    random_baselines_labels: list[str] | None = None,
) -> None:
    if random_baselines_labels is None:
        random_baselines_labels = [
            "With Refusal Option Random Guess",
            "Without Refusal Option Random Guess",
        ]
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


def plot_model_comparison(results, baselines, run_groups, color_groups):
    """Create a bar chart comparing model performance across different formats."""
    # Setup
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(len(run_groups))
    bar_width = 0.35
    color_map = {group: cfg.COLOR_CYCLE[i] for i, group in enumerate(color_groups)}

    # Draw model performance bars
    draw_model_bars(x_axis, results, run_groups, bar_width, color_map)

    # Draw baseline lines
    draw_baselines(x_axis, baselines, run_groups, bar_width)


    # Customize plot appearance
    plt.ylabel("Accuracy", fontsize=18)
    plt.yticks(np.arange(0, 0.5, 0.1), fontsize=18)
    plt.title("Model Performance by Group with Wilson CI @95%")
    plt.xticks(
        x_axis + bar_width / 2,
        cfg.GROUP_TITLES,
        fontsize=18,
    )

    plt.legend()
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{cfg.RESULTS_DIR}/bixbench_results_comparison.png")
    plt.show()


def draw_baselines(x_axis, baselines, run_groups, bar_width):
    """Draw baseline lines on the plot."""
    baseline_color = "grey"
    random_color = "grey"
    line_width = 2
    extension = 0.05
    half_bar = bar_width / 2
    baseline_bar = "-"
    flattened_run_groups = utils.flatten_list(run_groups)
    # Define baseline positions
    baseline_positions = [
        (baselines[run_name], x_axis[i // 2], 0 if i % 2 == 0 else bar_width)
        for i, run_name in enumerate(flattened_run_groups)
    ]

    # Draw model baselines
    for c, (baseline, x_pos, width_offset) in enumerate(baseline_positions):
        plt.hlines(
            y=baseline,
            xmin=x_pos - extension - half_bar + width_offset,
            xmax=x_pos + bar_width + extension - half_bar + width_offset,
            color=baseline_color,
            linestyle=baseline_bar,
            linewidth=line_width,
            label="baseline" if c == 0 else None,
        )

    # Draw random guess baselines
    random_label_used = False
    for c, baseline in enumerate(cfg.RANDOM_BASELINES):
        if baseline is None:
            continue
        plt.hlines(
            y=baseline,
            xmin=x_axis[c] - extension - half_bar,
            xmax=x_axis[c] + 2 * bar_width + extension - half_bar,
            color=random_color,
            linestyle="--",
            linewidth=line_width,
            label="random" if not random_label_used else None,
        )
        random_label_used = True


def draw_model_bars(x_axis, results, run_groups, bar_width, color_map):
    """Draw performance bars for each model."""
    for group_idx, group in enumerate(run_groups):
        for j, run_name in enumerate(group):
            mean = results[run_name]["mean"]
            ci_low = results[run_name]["ci_low"]
            ci_high = results[run_name]["ci_high"]
            yerr = np.array([
                [mean - ci_low],
                [ci_high - mean],
            ])
            label, color = next(
                [group, color]
                for group, color in color_map.items()
                if group in run_name
            )
            xpos = x_axis[group_idx] + j * bar_width
            print(run_name, color, mean, yerr, label)
            plt.bar(
                xpos,
                mean,
                bar_width,
                label=label if group_idx == 0 else None,
                color=color,
                yerr=yerr,
                capsize=5,
            )
