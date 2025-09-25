import ast
import asyncio
import base64
import json
import random
import re
from asyncio import Semaphore
from pathlib import Path
from typing import Any

import litellm
import nbformat
import numpy as np
import pandas as pd
from tqdm import tqdm

from bixbench import prompts


def load_dataframe_from_json_directory(path: str) -> pd.DataFrame:
    """Load a dataframe from a json directory."""
    all_data = []
    for file in list(Path(path).glob("**/*.json")):
        replica = re.search(r"replica_(\d+)", file.name)[1]
        replica = int(replica) if replica is not None else 0
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
            data["replica"] = replica
            all_data.append(data)
    return pd.DataFrame(all_data)


def flatten_list(nested_list: list[list[Any]]) -> list[Any]:
    """Flatten a nested list of items into a single list.

    Args:
        nested_list: A list containing sublists to flatten

    Returns:
        A flattened list containing all items from sublists
    """
    return [item for sublist in nested_list for item in sublist]


async def send_message_to_llm(
    message: list[dict[str, str]], model: str, sem: Semaphore
) -> Any:
    """Send a message to a language model with rate limiting.

    Args:
        message: The message to send to the LLM
        model: The model identifier to use
        sem: Semaphore for rate limiting requests

    Returns:
        The response from the language model
    """
    async with sem:
        return await litellm.acompletion(model=model, messages=message)


models = {
    "4o": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
}


async def process_model_batch(
    eval_df: pd.DataFrame, model_key: str, model_name: str, max_concurrent: int
) -> tuple[str, list[Any]]:
    """Process batch for a single model.

    Args:
        eval_df: Dataframe containing evaluation data
        model_key: Key for the model (e.g., "4o", "claude")
        model_name: Full name of the model to use
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Tuple of (model_key, results) for updating the dataframe
    """
    batch = eval_df.loc[eval_df.run_name.str.contains(model_key), "content"].tolist()
    results = await process_batch(batch, model_name, max_concurrent=max_concurrent)
    return model_key, results


async def run_eval_loop(
    eval_df: pd.DataFrame, max_concurrent: int = 10
) -> pd.DataFrame:
    """Process evaluation dataframe with multiple LLM models concurrently.

    Sends prompts from the dataframe to different LLM models based on the run_name
    and collects the results in the dataframe.

    Args:
        eval_df: Dataframe containing evaluation data with prompts
        max_concurrent: Maximum number of concurrent requests allowed

    Returns:
        Updated dataframe with model responses in the llm_answer column
    """
    # Ensure llm_answer column is of object type to handle mixed data types
    if "llm_answer" not in eval_df.columns:
        eval_df["llm_answer"] = None
    eval_df["llm_answer"] = eval_df["llm_answer"].astype("object")

    # Create tasks for all models to run concurrently
    tasks = [
        process_model_batch(eval_df, model_key, model_name, max_concurrent)
        for model_key, model_name in models.items()
    ]

    # Run all model processing tasks concurrently
    results = await asyncio.gather(*tasks)

    # Update the dataframe with results from all models
    for model_key, model_results in results:
        eval_df.loc[eval_df.run_name.str.contains(model_key), "llm_answer"] = (
            model_results
        )

    return eval_df


async def process_single(prompt: str, model: str, sem: Semaphore) -> str | None:
    """Process a single prompt with a language model with retry logic.

    Makes up to 5 attempts to get a response from the model.

    Args:
        prompt: The prompt to send to the model
        model: The model identifier to use
        sem: Semaphore for rate limiting requests

    Returns:
        The model's response content as string or None if all attempts fail
    """
    messages = [
        {"role": "user", "content": prompt},
    ]

    MAX_RETRIES = 4
    for attempt in range(5):
        try:
            res = await send_message_to_llm(messages, model, sem)
            return res.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES:  # Don't print on last attempt
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
            print(f"All 5 attempts failed. Last error: {e}")
            return None
    return None


async def process_with_progress(
    prompt: str, model: str, sem: Semaphore, pbar: tqdm
) -> str | None:
    """Process a single prompt and update progress bar.

    Callback function that processes a prompt and ensures the progress bar
    is updated even if an exception occurs.

    Args:
        prompt: The prompt to process
        model: The model identifier to use
        sem: Semaphore for rate limiting requests
        pbar: tqdm progress bar to update

    Returns:
        The result from processing the prompt or None if processing fails
    """
    try:
        return await process_single(prompt, model, sem)
    finally:
        pbar.update(1)


async def process_batch(
    prompts: list[str], model: str, max_concurrent: int = 5
) -> list[str | None]:
    """Process a batch of prompts concurrently with rate limiting and progress tracking.

    Args:
        prompts: List of prompts to process
        model: The model identifier to use
        max_concurrent: Maximum number of concurrent requests allowed

    Returns:
        List of results from processing each prompt (strings or None values)
    """
    sem = Semaphore(max_concurrent)

    # Setup progress bar
    pbar = tqdm(total=len(prompts), desc=f"Processing {model}")
    # Create tasks with the progress callback
    tasks = [process_with_progress(prompt, model, sem, pbar) for prompt in prompts]

    try:
        # Process tasks
        return await asyncio.gather(*tasks)
    finally:
        # Close the progress bar
        pbar.close()


def encode_image_to_base64(image: str) -> str:
    """Encode an already base64-encoded image string to base64 again.

    This function is used when the image data needs to be standardized
    to ensure consistent handling.

    Args:
        image: Base64 encoded image data

    Returns:
        Re-encoded base64 string
    """
    decoded_image = base64.b64decode(image)
    return base64.b64encode(decoded_image).decode("utf-8")


def load_notebook(notebook: str | dict[str, Any]):
    """Parse a notebook into nbformat.

    Attempts to parse a notebook into a dictionary format using nbformat.

    Args:
        notebook: The notebook to parse, which could be a string or a dictionary

    Returns:
        Dictionary representation of the notebook or empty dict if parsing fails
    """
    if isinstance(notebook, str):
        return nbformat.reads(json.dumps(ast.literal_eval(notebook)), as_version=4)
    return nbformat.from_dict(notebook)


def load_answer(answer: str | dict[str, Any]) -> str | dict[str, Any]:
    """Parse an answer into a dictionary format.

    Attempts multiple parsing methods: direct dict access, ast.literal_eval,
    and json.loads to handle different input formats.

    Args:
        answer: The answer to parse, which could be a string, dict, or other format

    Returns:
        Dictionary representation of the answer or empty dict if parsing fails
    """
    if not answer:
        return {}
    if isinstance(answer, dict):
        return answer
    try:
        # Try literal eval first
        return ast.literal_eval(answer)
    except (ValueError, SyntaxError):
        try:
            # Fallback to json loads
            return json.loads(answer)
        except (ValueError, TypeError, json.JSONDecodeError):
            # Return answer if parsing fails (for when answer is a string)
            return answer


def expand_open_answer_to_mcq(row: pd.Series, include_refusal: bool) -> pd.Series:
    """Expand an open answer row into an MCQ evaluation row.

    Args:
        row: DataFrame row containing open answer trajectory data
        include_refusal: Whether to include refusal option in MCQ

    Returns:
        Series representing an MCQ evaluation row
    """
    # Create a copy of the row for MCQ evaluation
    mcq_row = row.copy()

    # Set question format to MCQ
    mcq_row["question_format"] = "mcq"

    formatted_q, correct_letter, refusal_letter = questions_to_mcq(
        mcq_row.question,
        options=[
            mcq_row.ideal_answer,
            *mcq_row.metadata["distractors"],
        ],
        refusal_option=include_refusal,
    )

    mcq_row["formatted_question"] = formatted_q
    mcq_row["correct_letter"] = correct_letter
    mcq_row["refusal_letter"] = refusal_letter

    # Set evaluation mode
    mcq_row["eval_mode"] = (
        "mcq_with_refusal" if include_refusal else "mcq_without_refusal"
    )
    mcq_row["run_name"] = mcq_row["run_name"] + "_" + mcq_row["eval_mode"]

    return mcq_row


def create_eval_df(data: list[dict[str, Any]], expand_mcq: bool = True) -> pd.DataFrame:
    """Creates a dataframe for evaluation with one row per question.

    For open answer trajectories, expands them into additional MCQ evaluation rows.

    Args:
        data: List of dictionaries containing problem data
        expand_mcq: Whether to expand MCQ questions

    Returns:
        DataFrame with evaluation rows, including open answer and expanded MCQ rows
    """
    # Convert to DataFrame and filter out rows without agent answers
    evaluation_data = pd.DataFrame(data)
    evaluation_data = evaluation_data[evaluation_data["agent_answer"].apply(bool)]

    # Extract single question data from lists where needed
    # mcq_question is stored as a list with single question, extract it
    evaluation_data["question"] = evaluation_data["mcq_question"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
    )

    # mcq_options is stored as list of lists, extract the single list
    evaluation_data["mcq_options"] = evaluation_data["mcq_options"].apply(
        lambda x: (
            x[0] if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list) else x
        )
    )

    # agent_answer might be dict with single key, extract the value
    evaluation_data["agent_answer"] = evaluation_data["agent_answer"].apply(
        lambda x: next(iter(x.values())) if isinstance(x, dict) and len(x) == 1 else x
    )

    result = evaluation_data

    # Drop rows with no question or no format
    result = result.dropna(
        subset=["question", "question_format"], how="any"
    ).reset_index(drop=True)

    # Process open answer questions - expand to MCQ formats
    open_rows = result[result["question_format"] == "open"]
    expanded_rows = []

    for _, row in open_rows.iterrows():
        # Keep original open answer row
        open_row = row.copy()
        open_row["eval_mode"] = "open"
        assert open_row["question_format"] == "open"

        open_row["run_name"] = open_row["run_name"] + "_" + open_row["eval_mode"]

        expanded_rows.append(open_row)

        if expand_mcq:
            # Create MCQ with refusal option
            mcq_with_refusal = expand_open_answer_to_mcq(row, include_refusal=True)
            expanded_rows.append(mcq_with_refusal)

            # Create MCQ without refusal option
            mcq_without_refusal = expand_open_answer_to_mcq(row, include_refusal=False)
            expanded_rows.append(mcq_without_refusal)

    # Combine all rows
    eval_df = pd.DataFrame(expanded_rows) if expanded_rows else pd.DataFrame()

    eval_df["prompt"] = eval_df.apply(create_prompt, axis=1)
    eval_df["content"] = eval_df.apply(create_llm_message_content, axis=1)
    return eval_df


def questions_to_mcq(
    question: str, options: list[str | dict[str, Any]], refusal_option: bool = True
) -> tuple[str, str, str | None]:
    """Format a question and options into an MCQ format.

    Creates a formatted multiple-choice question with lettered options,
    randomly shuffles the options, and tracks the correct answer letter
    and optional refusal option letter.

    Args:
        question: The question text
        options: List of answer options with correct answer as first element
        refusal_option: Whether to include an "Insufficient information" option

    Returns:
        Tuple of (formatted question string, correct answer letter, refusal option letter)
    """
    options = options.copy()
    # Get all answer options
    correct_answer = options[0]
    if refusal_option:
        options.append("Insufficient information to answer the question")

    # Randomly shuffle options
    random.shuffle(options)

    # Find the index of the ideal answer to determine its letter
    correct_letter = chr(65 + options.index(correct_answer))
    if refusal_option:
        refusal_letter = chr(
            65 + options.index("Insufficient information to answer the question")
        )
    else:
        refusal_letter = None

    # Format the question with lettered options
    formatted_question = f"{question}\n"
    for j, opt in enumerate(options):
        formatted_question += f"{chr(65 + j)}. {opt}\n"

    # Join all questions with newlines
    return formatted_question, correct_letter, refusal_letter


def create_llm_message_content(row: pd.Series) -> list[dict[str, Any]]:
    """Create a message content structure for LLM API requests.

    Formats text and images from a dataframe row into the format expected
    by multimodal LLM APIs.

    Args:
        row: Dataframe row containing prompt and possibly images

    Returns:
        List of content elements (text and images) for the LLM API
    """
    content = [{"type": "text", "text": row.prompt}]

    if hasattr(row, "md_images") and row.md_images:
        for img_data in row.md_images:
            try:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{img_data}",
                        },
                    }
                )
            except Exception as e:
                print(f"Error adding image to content: {e}")

    return content


def create_prompt(row: pd.Series) -> str | float:
    """Create an appropriate prompt based on the question format.

    Selects either open-ended or MCQ prompt template and formats it
    with the data from the input row.

    Args:
        row: Dataframe row containing question and answer data

    Returns:
        Formatted prompt string or np.nan if no matching format
    """
    question_format = row.get("question_format", None)

    if question_format == "open":
        return prompts.OPEN_ENDED_EVAL_PROMPT.format(
            question=row.question,
            correct_answer=row.ideal_answer,
            proposed_answer=row.agent_answer,
        )
    if question_format == "mcq":
        return (
            prompts.MCQ_EVAL_PROMPT.replace("{{notebook}}", row.md_notebook)
            .replace("{{question}}", row.formatted_question)
            .replace("{{proposed_answer}}", str(row.agent_answer))
        )
    return np.nan


def xml_extract(text: str) -> str:
    """Extract an answer letter from XML tags in text.

    Looks for a pattern like <answer>A</answer> and extracts the letter.

    Args:
        text: The text to search for the answer pattern

    Returns:
        The extracted answer letter or 'Z' if no match is found
    """
    match = re.search(r"<answer>([A-Z])</answer>", text)
    return match[1] if match else "Z"


def majority_vote(row: pd.Series, k: int = 10) -> str | None:
    """Apply majority voting to a series of predictions.

    Randomly samples k predictions from the input and returns the most common value.

    Args:
        row: Series of predictions
        k: Number of predictions to sample

    Returns:
        The most common prediction or None if none can be determined
    """
    # Get all predictions excluding the 'answer' column
    predictions = row[:-1]
    # Randomly sample k predictions without replacement
    rng = np.random.default_rng()
    sampled_votes = rng.choice(
        predictions, size=min(k, len(predictions)), replace=False
    )
    # Get mode (most common value) of sampled votes
    # Check if all votes are integers
    if not all(isinstance(vote, int | float | str) for vote in sampled_votes):
        return None
    unique_values, counts = np.unique(sampled_votes, return_counts=True)

    return None if unique_values.size == 0 else unique_values[np.argmax(counts)]


def run_majority_voting(
    grouped_df: pd.DataFrame, k_values: list[int], n_trials: int
) -> tuple[list[int], list[float], list[float]]:
    """Run majority voting experiments with different k values.

    Applies majority voting with various k values over multiple trials
    and collects accuracy statistics.

    Args:
        grouped_df: Dataframe with predictions grouped by question
        k_values: List of k values to test for majority voting
        n_trials: Number of trials to run for each k value

    Returns:
        Tuple of (k_values, mean accuracies, standard deviations)
    """
    # Fix: Calculate majority predictions first
    majority_predictions = grouped_df["llm_answer"].apply(majority_vote)

    # Calculate accuracy
    accuracy = (majority_predictions == grouped_df["correct_letter"]).mean()
    print(f"Majority voting accuracy: {accuracy:.2%}")

    # Run multiple trials for different k values
    accuracies = {k: [] for k in k_values}

    for k in k_values:
        for _ in range(n_trials):
            # Apply majority voting with current k to each row
            predictions = grouped_df["llm_answer"].apply(
                lambda x, k_value=k: majority_vote(x, k=k_value)
            )

            # Calculate and store accuracy
            acc = (predictions == grouped_df["correct_letter"]).mean()
            accuracies[k].append(acc)

    # Calculate means and standard deviations
    means = [np.mean(accuracies[k]) for k in k_values]
    stds = [np.std(accuracies[k]) for k in k_values]
    return k_values, means, stds


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate Wilson confidence interval for a proportion.

    The Wilson score interval is used to calculate confidence intervals for
    binomial proportions, especially when sample sizes are small.

    Args:
        p: Observed proportion
        n: Sample size
        z: Z-score for desired confidence level (default 1.96 for 95% CI)

    Returns:
        Tuple of (lower bound, upper bound) of the confidence interval
    """
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - spread, center + spread


def calculate_results(
    df: pd.DataFrame, total_questions_per_run: int | None = None
) -> dict[str, dict[str, float]]:
    """
    Calculate means and confidence intervals for each model and format.

    Args:
        df: DataFrame containing model evaluation results
        total_questions_per_run: Total number of questions to normalize
            by as some runs may have failed and were not included in the eval_df

    Returns:
        Dictionary mapping run names to statistics including mean score and confidence intervals
    """
    results = {}
    for run in df["run_name"].unique():
        mask = df["run_name"].str.contains(run)
        scores = df[mask]["correct"]
        if len(scores) > 0:
            mean = (
                scores.sum() / total_questions_per_run
                if total_questions_per_run is not None
                else scores.mean()
            )
            n = (
                total_questions_per_run
                if total_questions_per_run is not None
                else len(scores)
            )
            ci_low, ci_high = wilson_ci(mean, n)
            results[run] = {
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
    return results
