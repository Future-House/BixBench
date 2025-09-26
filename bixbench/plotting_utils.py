# DISCLAIMER: This file is highly tailored to the BixBench paper requirements.
# It is not designed to be used as a general function for plotting model performance.


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from plot_style import set_fh_mpl_style

from bixbench import postprocessing_utils as utils

set_fh_mpl_style()

# There is stochasticity in the majority vote accuracy plot, so we set a seed for reproducibility
np.random.default_rng(42)

COLOR_CYCLE = ["#1BBC9B", "#FF8C00", "#FF69B4", "#ce8aed", "#80cedb", "#FFFFFF"]


def majority_vote_accuracy_by_k(
    run_results: dict[str, tuple[list[int], list[float], list[float]]],
    name: str = "",
    random_baselines: list[float] | None = None,
    random_baselines_labels: list[str] | None = None,
    results_dir: str = "bixbench_results",
) -> None:
    """
    Plot the accuracy of majority voting as a function of the number of votes (k).

    Args:
        run_results: Dictionary mapping run names to tuples of (k_values, means, stds)
        name: Name suffix for the saved plot file
        random_baselines: List of accuracy values for random baseline models
        random_baselines_labels: Labels for the random baseline models
        results_dir: Directory to save results

    Returns:
        None: Saves the plot to disk and displays it
    """
    if random_baselines_labels is None:
        random_baselines_labels = [
            "With Refusal Option Random Guess",
            "Without Refusal Option Random Guess",
        ]
    if random_baselines is None:
        random_baselines = [0.2, 0.25]
    plt.figure(figsize=(12, 6))

    for run_name, (k_values, means, stds) in run_results.items():
        if k_values is None:
            continue
        plt.plot(k_values, means, "o-", label=run_name)
        plt.fill_between(
            k_values,
            [m - s for m, s in zip(means, stds, strict=True)],
            [m + s for m, s in zip(means, stds, strict=True)],
            alpha=0.2,
        )

    plt.xlabel("Number of Votes (k)", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xlim(1, max(k_values))
    plt.ylim(0.1, 0.35)
    plt.yticks(
        np.arange(0.1, 0.36, 0.05),
        [f"{x:.2f}" for x in np.arange(0.1, 0.36, 0.05)],
        fontsize=18,
    )
    plt.title("Majority Voting Accuracy", fontsize=18)
    plt.xticks(k_values, fontsize=18)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    for i, (baseline, label) in enumerate(zip(random_baselines, random_baselines_labels, strict=True)):
        plt.axhline(
            y=baseline,
            color="red" if i == 0 else "green",
            linestyle=":",
            label=label,
        )
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3, visible=True)
    plt.savefig(f"{results_dir}/majority_vote_accuracy_{name}.png")
    plt.show()


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    baselines: dict[str, float],
    run_groups: list[list[str]],
    color_groups: list[str],
    group_titles: list[str] | None = None,
    random_baselines: list[float] | None = None,
    results_dir: str = "bixbench_results",
) -> None:
    """
    Create a bar chart comparing model performance across different formats.

    Args:
        results: Dictionary mapping run names to performance metrics (mean, ci_low, ci_high)
        baselines: Dictionary mapping run names to baseline performance values
        run_groups: List of lists, where each inner list contains run names in a group
        color_groups: List of group names for color mapping
        group_titles: Optional list of titles for each group
        random_baselines: Optional list of random baseline values for each group
        results_dir: Directory to save results

    Returns:
        None: Saves the plot to disk and displays it
    """
    # Setup
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(len(run_groups))
    bar_width = 0.35
    color_map = {group: COLOR_CYCLE[i] for i, group in enumerate(color_groups)}

    # Use default group titles if not provided
    if group_titles is None or len(group_titles) != len(run_groups):
        group_titles = [f"Group {i + 1}" for i in range(len(run_groups))]

    # Draw model performance bars
    draw_model_bars(x_axis, results, run_groups, bar_width, color_map)

    # Draw baseline lines
    draw_baselines(x_axis, baselines, run_groups, bar_width, random_baselines)

    # Customize plot appearance
    plt.ylabel("Accuracy", fontsize=18)
    plt.yticks(np.arange(0, 0.5, 0.1), fontsize=18)
    plt.title("Model Performance by Group with Wilson CI @95%")
    plt.xticks(
        x_axis + bar_width / 2,
        group_titles,
        fontsize=18,
    )

    plt.legend()
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/bixbench_results_comparison.png")
    plt.show()


def draw_baselines(
    x_axis: np.ndarray,
    baselines: dict[str, float],
    run_groups: list[list[str]],
    bar_width: float,
    random_baselines: list[float] | None = None,
) -> None:
    """
    Draw baseline lines on the plot for performance comparison.

    Args:
        x_axis: Array of x-coordinates for the group positions
        baselines: Dictionary mapping run names to baseline performance values
        run_groups: List of lists, where each inner list contains run names in a group
        bar_width: Width of the bars in the plot
        random_baselines: Optional list of random baseline values for each group

    Returns:
        None
    """
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
            label="baseline" if c == 0 else "",
        )

    # Draw random guess baselines
    random_label_used = False
    for c, baseline in enumerate(random_baselines or []):
        if baseline is None:
            continue
        plt.hlines(
            y=baseline,
            xmin=x_axis[c] - extension - half_bar,
            xmax=x_axis[c] + 2 * bar_width + extension - half_bar,
            color=random_color,
            linestyle="--",
            linewidth=line_width,
            label="random" if not random_label_used else "",
        )
        random_label_used = True


def draw_model_bars(
    x_axis: np.ndarray,
    results: dict[str, dict[str, float]],
    run_groups: list[list[str]],
    bar_width: float,
    color_map: dict[str, str],
) -> None:
    """
    Draw performance bars for each model on the plot.

    Args:
        x_axis: Array of x-coordinates for the group positions
        results: Dictionary mapping run names to performance metrics (mean, ci_low, ci_high)
        run_groups: List of lists, where each inner list contains run names in a group
        bar_width: Width of the bars in the plot
        color_map: Dictionary mapping group names to colors

    Returns:
        None
    """
    for group_idx, group in enumerate(run_groups):
        for j, run_name in enumerate(group):
            mean = results[run_name]["mean"]
            ci_low = results[run_name]["ci_low"]
            ci_high = results[run_name]["ci_high"]
            yerr = np.array([
                [mean - ci_low],
                [ci_high - mean],
            ])
            label, color = next([group, color] for group, color in color_map.items() if group in run_name)
            xpos = x_axis[group_idx] + j * bar_width
            plt.bar(
                xpos,
                mean,
                bar_width,
                label=label if group_idx == 0 else None,
                color=color,
                yerr=yerr,
                capsize=5,
            )


def plot_simplified_comparison(
    results: dict[str, dict[str, float]],
    run_groups: list[list[str]],
    group_titles: list[str] | None = None,
    has_mcq: bool = False,
    results_dir: str = "bixbench_results",
) -> None:
    """
    Create a simplified bar chart comparing model performance.

    Args:
        results: Dictionary mapping run names to performance metrics (mean, ci_low, ci_high)
        run_groups: List of lists, where each inner list contains run names in a group
        group_titles: Optional titles for each group
        has_mcq: Whether the results include MCQ questions (to show random baselines)
        results_dir: Directory to save results

    Returns:
        None: Saves the plot to disk and displays it
    """
    # Setup
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(len(run_groups))
    width = 0.8 / max(len(group) for group in run_groups)

    # Use group titles if provided, otherwise generate generic ones
    if group_titles is None or len(group_titles) != len(run_groups):
        group_titles = [f"Group {i + 1}" for i in range(len(run_groups))]

    # Prepare colors from a default color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Draw bars for each run
    for i, group in enumerate(run_groups):
        for j, run_name in enumerate(group):
            if run_name not in results:
                continue

            result = results[run_name]
            mean = result["mean"]
            ci_low = result["ci_low"]
            ci_high = result["ci_high"]
            yerr = np.array([[mean - ci_low], [ci_high - mean]])

            # Calculate bar position
            pos = x_axis[i] + (j - len(group) / 2 + 0.5) * width

            # Get color
            color_idx = j % len(color_cycle)
            color = color_cycle[color_idx]

            plt.bar(
                pos,
                mean,
                width,
                label=run_name if i == 0 else None,
                color=color,
                yerr=yerr,
                capsize=5,
            )

    # Add random baselines for MCQ questions
    if has_mcq:
        plt.axhline(y=0.2, color="red", linestyle="--", label="Random (w/ refusal)")
        plt.axhline(y=0.25, color="green", linestyle="--", label="Random (w/o refusal)")

    # Customize plot appearance
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Model Performance Comparison")
    plt.xticks(x_axis, group_titles, fontsize=12)
    plt.legend(loc="best")
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.7)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(f"{results_dir}/model_comparison.png")
    plt.show()
