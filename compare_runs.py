#!/usr/bin/env python
"""
Script to visually compare results from different BixBench evaluation runs.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_runs():
    """Find all run directories in bixbench_results."""
    base_dir = Path("bixbench_results")
    run_dirs = [d for d in base_dir.glob("*") if d.is_dir() and d.name not in ["multi_model", "multi_model_mcq"]]
    return sorted(run_dirs)


def load_eval_data(run_dir, eval_type="multi_model"):
    """Load evaluation data from a specific run directory."""
    eval_path = run_dir / eval_type / "eval_df_new.csv"
    if not eval_path.exists():
        print(f"Warning: No evaluation data found at {eval_path}")
        return None
    
    return pd.read_csv(eval_path)


def compare_accuracy(run_dirs, eval_type="multi_model", output_dir=None):
    """Compare accuracy across different runs."""
    # Load data from all specified runs
    run_data = {}
    for run_dir in run_dirs:
        eval_df = load_eval_data(run_dir, eval_type)
        if eval_df is not None:
            # Calculate accuracy per model
            results = {}
            for model in eval_df["run_name"].unique():
                model_df = eval_df[eval_df["run_name"] == model]
                correct = model_df["correct"].sum()
                total = len(model_df)
                results[model] = correct / total
            
            run_data[run_dir.name] = results
    
    if not run_data:
        print("No data found for comparison!")
        return
    
    # Create a DataFrame for plotting
    df_models = set()
    for results in run_data.values():
        df_models.update(results.keys())
    
    df_models = sorted(df_models)
    df_data = []
    
    for run_name, results in run_data.items():
        row = {"run": run_name}
        for model in df_models:
            row[model] = results.get(model, 0)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    df.set_index("run").plot(kind="bar", figsize=(12, 6))
    plt.title(f"Model Accuracy Comparison ({eval_type})")
    plt.ylabel("Accuracy")
    plt.xlabel("Run ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/accuracy_comparison_{eval_type}.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare BixBench evaluation runs")
    parser.add_argument(
        "--runs", 
        nargs="+", 
        help="Run IDs to compare (if not specified, all runs will be compared)"
    )
    parser.add_argument(
        "--type", 
        choices=["multi_model", "multi_model_mcq", "both"], 
        default="both",
        help="Type of evaluation to compare"
    )
    parser.add_argument(
        "--output", 
        help="Directory to save comparison plots"
    )
    
    args = parser.parse_args()
    
    # Find available runs
    all_runs = find_runs()
    if not all_runs:
        print("No run directories found in bixbench_results/")
        return
    
    print(f"Found {len(all_runs)} evaluation runs:")
    for run in all_runs:
        print(f" - {run.name}")
    
    # Filter runs if specified
    if args.runs:
        selected_runs = [r for r in all_runs if r.name in args.runs]
        if not selected_runs:
            print("No matching runs found for the specified IDs!")
            return
    else:
        selected_runs = all_runs
    
    # Run comparisons
    if args.type in ["multi_model", "both"]:
        print("\nComparing open-ended results...")
        compare_accuracy(selected_runs, "multi_model", args.output)
    
    if args.type in ["multi_model_mcq", "both"]:
        print("\nComparing MCQ results...")
        compare_accuracy(selected_runs, "multi_model_mcq", args.output)


if __name__ == "__main__":
    main()