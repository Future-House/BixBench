import ast
import asyncio
import base64
import json
import random
import re
from asyncio import Semaphore
from typing import Any, Optional

import litellm
import numpy as np
import pandas as pd
from tqdm import tqdm

from bixbench import prompts

litellm.set_verbose = False


def flatten_list(nested_list):
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
    eval_df: pd.DataFrame, max_concurrent: int = 100
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


async def process_single(prompt: str, model: str, sem: Semaphore) -> dict[str, Any]:
    """Process a single prompt with a language model with retry logic.

    Makes up to 5 attempts to get a response from the model.

    Args:
        prompt: The prompt to send to the model
        model: The model identifier to use
        sem: Semaphore for rate limiting requests

    Returns:
        The model's response content or None if all attempts fail
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
) -> dict[str, Any]:
    """Process a single prompt and update progress bar.

    Callback function that processes a prompt and ensures the progress bar
    is updated even if an exception occurs.

    Args:
        prompt: The prompt to process
        model: The model identifier to use
        sem: Semaphore for rate limiting requests
        pbar: tqdm progress bar to update

    Returns:
        The result from processing the prompt
    """
    try:
        return await process_single(prompt, model, sem)
    finally:
        pbar.update(1)


async def process_batch(
    prompts: list[dict[str, Any]], model: str, max_concurrent: int = 5
) -> list[dict[str, Any]]:
    """Process a batch of prompts concurrently with rate limiting and progress tracking.

    Args:
        prompts: List of prompts to process
        model: The model identifier to use
        max_concurrent: Maximum number of concurrent requests allowed

    Returns:
        List of results from processing each prompt
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


def load_answer(answer: str | dict[str, Any]) -> dict[str, Any]:
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
            # Return empty dict if parsing fails
            return {}


def create_eval_df(data: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a dataframe for evaluation with one row per question.

    Uses vectorized operations for better performance.
    """
    # First, apply load_answer to all relevant columns at once
    evaluation_data = data.copy()

    # Handle list type agent answers
    mask = evaluation_data["agent_answer"].apply(lambda x: isinstance(x, list))
    evaluation_data.loc[mask, "agent_answer"] = evaluation_data.loc[
        mask, "agent_answer"
    ].apply(lambda x: {f"q{i}": v for i, v in enumerate(x)})

    # Filter out rows without agent answers
    evaluation_data = evaluation_data[evaluation_data["agent_answer"].apply(bool)]

    # Now prepare for explosion
    # Create a column with question numbers from ideal_answer keys
    evaluation_data["question_keys"] = evaluation_data["ideal_answer"].apply(
        lambda x: list(x.keys())
    )

    # Explode the dataframe to create one row per question
    exploded = evaluation_data.explode("question_keys")

    # Now create the final dataframe in a vectorized way
    result = pd.DataFrame({
        "uuid": exploded["problem_id"] + "_" + exploded["question_keys"].astype(str),
        "problem_id": exploded["problem_id"],
        "question": exploded.apply(
            lambda row: row["mcq_question"].get(row["question_keys"], None), axis=1
        ),
        "question_num": exploded["question_keys"],
        "agent_answer": exploded.apply(
            lambda row: row["agent_answer"].get(row["question_keys"], None), axis=1
        ),
        "ideal_answer": exploded.apply(
            lambda row: row["ideal_answer"].get(row["question_keys"], None), axis=1
        ),
        "run_name": exploded["run_name"],
        "md_notebook": exploded["md_notebook"],
        "md_images": exploded["md_images"],
        "mcq_options": exploded.apply(
            lambda row: row["mcq_options"].get(row["question_keys"], None), axis=1
        ),
        "refusal_option": exploded["refusal_option"],
        "question_format": exploded["question_format"],
        "model": exploded["model"],
    })

    result[["formatted_question", "correct_letter", "insufficient_letter"]] = (
        result.apply(
            lambda row: pd.Series(
                questions_to_mcq(
                    row["question"],
                    row["mcq_options"],
                    refusal_option=row["refusal_option"],
                )
            ),
            axis=1,
        )
    )

    result["prompt"] = result.apply(create_prompt, axis=1)
    result["content"] = result.apply(create_llm_message_content, axis=1)

    return result


def questions_to_mcq(question, options: list[dict[str, Any]], refusal_option=True):
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


def create_llm_message_content(row) -> list[dict[str, Any]]:
    """Create a message content structure for LLM API requests.

    Formats text and images from a dataframe row into the format expected
    by multimodal LLM APIs.

    Args:
        row: Dataframe row containing prompt and possibly images

    Returns:
        List of content elements (text and images) for the LLM API
    """
    content = [{"type": "text", "text": row.prompt}]

    if row.md_images:
        for img_data in row.md_images:
            try:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"{img_data}",
                    },
                })
            except Exception as e:
                print(f"Error adding image to content: {e}")

    return content


def create_prompt(row: pd.Series) -> str:
    """Create an appropriate prompt based on the question format.

    Selects either open-ended or MCQ prompt template and formats it
    with the data from the input row.

    Args:
        row: Dataframe row containing question and answer data

    Returns:
        Formatted prompt string or np.nan if no matching format
    """
    if "open" in row["question_format"]:
        return prompts.OPEN_ENDED_EVAL_PROMPT.format(
            question=row.question,
            correct_answer=row.ideal_answer,
            proposed_answer=row.agent_answer,
        )
    if "mcq" in row["question_format"]:
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
    if match:
        return match.group(1)
    return "Z"


def majority_vote(row: pd.Series, k: int = 10) -> Optional[str]:
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

    if unique_values.size == 0:
        return None
    return unique_values[np.argmax(counts)]


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
    """Calculate Wilson confidence interval for a proportion."""
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - spread, center + spread


def calculate_results(
    df: pd.DataFrame, total_questions: int | None = None
) -> dict[str, dict[str, Any]]:
    """
    Calculate means and confidence intervals for each model and format.

    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results
        total_questions (int | None): Total number of questions to normalize
        by as sum runs may have failed and so were not included in the eval_df

    Returns:
        list: List of dictionaries containing statistical results for each model and format
    """
    results = {}
    for run in df["run_name"].unique():
        mask = df["run_name"].str.contains(run)
        scores = df[mask]["correct"]
        if len(scores) > 0:
            mean = (
                scores.sum() / total_questions
                if total_questions is not None
                else scores.mean()
            )
            n = total_questions if total_questions is not None else len(scores)
            ci_low, ci_high = wilson_ci(mean, n)
            results[run] = {
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
    return results
