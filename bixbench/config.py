import json

RESULTS_DIR = "bixbench_results"


# POSTPROCESSING CONFIG
K_VALUE = 10

RUN_NAME_GROUPS = [
    ["4o_open_image", "claude_open_image"],
    ["4o_mcq_image_with_refusal", "claude_mcq_image_with_refusal"],
    ["4o_mcq_image_without_refusal", "claude_mcq_image_without_refusal"],
]

MAJORITY_VOTE_GROUPS = {
    "image_comparison": [
        "claude_mcq_image_with_refusal",
        "claude_mcq_no_image_with_refusal",
        "4o_mcq_image_with_refusal",
        "4o_mcq_no_image_with_refusal",
    ],
    "refusal_option_comparison": [
        "claude_mcq_image_without_refusal",
        "4o_mcq_image_without_refusal",
        "claude_mcq_image_with_refusal",
        "4o_mcq_image_with_refusal",
    ],
}

GROUP_TITLES = ["Open-answer", "MCQ w/ refusal", "MCQ w/o refusal"]
COLOR_GROUPS = ["4o", "claude"]

# Load baselines from JSON file
with open(f"{RESULTS_DIR}/zero_shot_baselines.json", encoding="utf-8") as f:
    BASELINES = json.load(f)
BASELINES = {k: v["accuracy"] for k, v in BASELINES.items()}


BASELINE_NAME_MAPPINGS = {
    "gpt-4o-grader-openended": "4o_open_image",
    "claude-3-5-sonnet-latest-grader-openended": "claude_open_image",
    "gpt-4o-grader-mcq-refusal-True": "4o_mcq_image_with_refusal",
    "claude-3-5-sonnet-latest-grader-mcq-refusal-True": "claude_mcq_image_with_refusal",
    "gpt-4o-grader-mcq-refusal-False": "4o_mcq_image_without_refusal",
    "claude-3-5-sonnet-latest-grader-mcq-refusal-False": "claude_mcq_image_without_refusal",
}

BASELINES = {BASELINE_NAME_MAPPINGS[k]: v for k, v in BASELINES.items()}
# This needs to be the same length as RUN_NAME_GROUPS
RANDOM_BASELINES = [None, 0.2, 0.25]

COLOR_CYCLE = ["#1BBC9B", "#FF8C00", "#FF69B4", "#ce8aed", "#80cedb", "#FFFFFF"]
