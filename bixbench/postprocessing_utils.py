import ast
import asyncio
import base64
import json
import random
import re
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Optional

import litellm
import nbformat
import numpy as np
import pandas as pd
from tqdm import tqdm

from bixbench import prompts

litellm.set_verbose = False


def load_dataframe_from_json_directory(path: str) -> pd.DataFrame:
    """Load a dataframe from a json directory."""
    data = []
    for file in list(Path(path).glob("**/*.json")):
        with open(file, encoding="utf-8") as f:
            data.append(json.load(f))
    return pd.DataFrame(data)


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
    # Set appropriate max_tokens based on model to avoid output token rate limits
    max_tokens = 4000 if "claude" in model.lower() else 8000
    
    async with sem:
        return await litellm.acompletion(
            model=model, 
            messages=message,
            max_tokens=max_tokens
        )


models = {
    "4o": "gpt-4o",
    "claude35": "claude-3-5-sonnet-20241022",
    "claude37": "claude-3-7-sonnet-20250219",
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
    # Ensure llm_answer column is of object type to handle mixed data types
    if "llm_answer" not in eval_df.columns:
        eval_df["llm_answer"] = None
    eval_df["llm_answer"] = eval_df["llm_answer"].astype("object")

    # Create tasks for all models to run concurrently
    tasks = []
    for model_key, model_name in models.items():
        # Check if there are any rows for this model
        if eval_df.run_name.str.contains(model_key).any():
            # Determine appropriate concurrency for each model
            model_specific_concurrency = min(
                max_concurrent, 
                20 if "claude" in model_name.lower() else max_concurrent
            )
            tasks.append(
                process_model_batch(eval_df, model_key, model_name, model_specific_concurrency)
            )
    
    if not tasks:
        print("No matching model keys found in the dataframe")
        return eval_df

    # Run all model processing tasks concurrently
    results = await asyncio.gather(*tasks)

    # Update the dataframe with results from all models
    for model_key, model_results in results:
        mask = eval_df.run_name.str.contains(model_key)
        if mask.any():
            eval_df.loc[mask, "llm_answer"] = model_results
        else:
            print(f"No matches found for model key: {model_key}")

    return eval_df


async def process_single(prompt: str, model: str, sem: Semaphore) -> Optional[str]:
    """Process a single prompt with a language model with retry logic.

    Makes up to 5 attempts to get a response from the model, with
    exponential backoff specifically for rate limit errors.

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

    MAX_RETRIES = 6
    for attempt in range(7):  # 7 attempts total (1 initial + 6 retries)
        try:
            res = await send_message_to_llm(messages, model, sem)
            return res.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            
            # Enhanced rate limit error detection
            rate_limit_terms = [
                "rate limit", "ratelimit", "too many requests", "429", 
                "capacity", "quota", "exceeded", "throttl", "tps limit", 
                "token rate", "too fast", "server busy", "overloaded"
            ]
            is_rate_limit = any(term in error_str for term in rate_limit_terms)
            
            if attempt < MAX_RETRIES:
                # Calculate backoff time - exponential for rate limits, shorter for other errors
                if is_rate_limit:
                    # Enhanced exponential backoff with jitter for rate limits
                    # Start with 3s, then 9s, 27s, 81s, 243s, 729s for Claude models
                    is_claude = "claude" in model.lower()
                    base_delay = 3 if is_claude else 2
                    backoff_time = (base_delay ** (attempt + 1)) + (random.random() * 1.0)
                    print(f"Rate limit error on attempt {attempt + 1} for {model}: {e}")
                    print(f"Backing off for {backoff_time:.2f} seconds...")
                else:
                    # Shorter delay for other errors
                    backoff_time = 0.5 * (attempt + 1)
                    print(f"Attempt {attempt + 1} failed (non-rate limit) for {model}: {e}")
                    print(f"Retrying in {backoff_time:.2f} seconds...")
                
                # Wait before retry
                await asyncio.sleep(backoff_time)
                continue
            
            print(f"All 7 attempts failed. Last error: {e}")
            return None
    return None


async def process_with_progress(
    prompt: str, model: str, sem: Semaphore, pbar: tqdm
) -> Optional[str]:
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
) -> list[Optional[str]]:
    """Process a batch of prompts concurrently with rate limiting and progress tracking.
    
    Sets appropriate concurrency based on the model being used, with defaults
    that respect API rate limits.

    Args:
        prompts: List of prompts to process
        model: The model identifier to use
        max_concurrent: Maximum number of concurrent requests allowed

    Returns:
        List of results from processing each prompt (strings or None values)
    """
    # Model-specific concurrency limits based on known API limits
    model_concurrency = {
        "gpt-4o": 25,  # GPT-4o can handle 500 RPM, so ~25 concurrent with safety margin
        "claude-3-5-sonnet-20241022": 30,  # Claude 3.5 with 2000 RPM, 160k input tokens/min
        "claude-3-7-sonnet-20250219": 30,  # Claude 3.7 with 2000 RPM, 80k input tokens/min
    }
    
    # Choose the lower of the user-specified max_concurrent or the model-specific limit
    model_max = model_concurrency.get(model, 5)  # Default to 5 for unknown models
    effective_max = min(max_concurrent, model_max)
    
    print(f"Processing batch with {effective_max} concurrent requests for model {model}")
    sem = Semaphore(effective_max)

    # Setup progress bar
    pbar = tqdm(total=len(prompts), desc=f"Processing {model}")
    # Create tasks with the progress callback
    tasks = [process_with_progress(prompt, model, sem, pbar) for prompt in prompts]

    try:
        # Process tasks
        results = await asyncio.gather(*tasks)
        print(f"Batch processing completed for {model} with {len([r for r in results if r is not None])}/{len(prompts)} successful responses")
        return results
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


def load_notebook(notebook: str | dict[str, Any]) -> dict[str, Any]:
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
    """Creates a dataframe for evaluation with one row per question.

    Uses vectorized operations for better performance.

    Args:
        data: List of dictionaries containing problem data

    Returns:
        DataFrame with one row per question, including formatted questions and prompts
    """
    # First, apply load_answer to all relevant columns at once
    evaluation_data = data.copy()

    # Handle list type agent answers
    for col in ["agent_answer", "mcq_question", "mcq_options"]:
        mask = evaluation_data[col].apply(lambda x: isinstance(x, list))
        evaluation_data.loc[mask, col] = evaluation_data.loc[mask, col].apply(
            lambda x: {f"q{i + 1}": v for i, v in enumerate(x)}
        )

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
        "refusal_option": exploded.get("refusal_option", None),
        "question_format": exploded.get("question_format", None),
        "model": exploded.get("model", None),
    })

    # Drop rows with no question or no format
    result = result.dropna(
        subset=["question", "question_format"], how="any"
    ).reset_index(drop=True)

    # Drop MCQ questions with any NaN values
    mcq_mask = result["question_format"] == "mcq"
    result = result[~(mcq_mask & result.isna().any(axis=1))]

    # Apply MCQ formatting only to MCQ questions
    mcq_rows = result[result["question_format"] == "mcq"].index
    if len(mcq_rows) > 0:
        result.loc[
            mcq_rows, ["formatted_question", "correct_letter", "refusal_letter"]
        ] = result.loc[mcq_rows].apply(
            lambda row: pd.Series(
                questions_to_mcq(
                    row["question"],
                    row["mcq_options"],
                    refusal_option=row["refusal_option"],
                ),
                index=["formatted_question", "correct_letter", "refusal_letter"],
            ),
            axis=1,
        )
    result["prompt"] = result.apply(create_prompt, axis=1)
    result["content"] = result.apply(create_llm_message_content, axis=1)

    return result


def questions_to_mcq(
    question: str, options: list[str | dict[str, Any]], refusal_option: bool = True
) -> tuple[str, str, Optional[str]]:
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
    if text is None:
        return "Z"  # Return default for None inputs
        
    try:
        match = re.search(r"<answer>([A-Z])</answer>", text)
        if match:
            return match.group(1)
    except TypeError:
        # Handle any other type errors that might occur
        pass
        
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
    # Check if we have enough data for meaningful majority voting
    if len(grouped_df) < 2:
        print("Insufficient data for majority voting (less than 2 questions)")
        return [], [], []
        
    # Check if we have enough model answers for voting
    if all(len(answers) < 2 for answers in grouped_df["llm_answer"]):
        print("Insufficient model answers for majority voting (less than 2 answers per question)")
        return [], [], []
    
    # Check if we have any valid k values that are less than the number of answers
    max_answers = max(len(answers) for answers in grouped_df["llm_answer"])
    valid_k_values = [k for k in k_values if k <= max_answers]
    
    if not valid_k_values:
        print(f"No valid k values for majority voting (max answers: {max_answers})")
        return [], [], []
    
    # Calculate majority predictions for the maximum valid k value
    majority_predictions = grouped_df["llm_answer"].apply(
        lambda x: majority_vote(x, k=min(max_answers, max(valid_k_values, default=1)))
    )

    # Calculate and display overall accuracy
    accuracy = (majority_predictions == grouped_df["correct_letter"]).mean()
    print(f"Majority voting accuracy: {accuracy:.2%}")

    # Run multiple trials for different k values
    accuracies = {k: [] for k in valid_k_values}

    for k in valid_k_values:
        for _ in range(n_trials):
            # Apply majority voting with current k to each row
            predictions = grouped_df["llm_answer"].apply(
                lambda x, k_value=k: majority_vote(x, k=k_value)
            )

            # Calculate and store accuracy
            acc = (predictions == grouped_df["correct_letter"]).mean()
            accuracies[k].append(acc)

    # Calculate means and standard deviations
    means = [np.mean(accuracies[k]) for k in valid_k_values]
    stds = [np.std(accuracies[k]) for k in valid_k_values]
    return valid_k_values, means, stds


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
    # Return simple estimate if n is too small
    if n < 5:
        return p, p
        
    # Ensure inputs are valid to avoid NaN results
    p = max(0.0, min(1.0, p))  # Ensure p is between 0 and 1
    n = max(1, n)              # Ensure n is at least 1
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    
    # Handle edge cases to avoid sqrt of negative numbers
    sqrt_term = max(0, p * (1 - p) / n + z**2 / (4 * n**2))
    spread = z * np.sqrt(sqrt_term) / denominator
    
    # Return bounds, ensuring they stay in [0,1]
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    
    # Add debug message for unusual CI behavior
    if lower > p or upper < p:
        print(f"Warning: Wilson CI calculation produced bounds that don't contain the proportion: p={p}, n={n}, CI=[{lower}, {upper}]")
        
    return lower, upper


def calculate_results(
    df: pd.DataFrame, total_questions_per_run: Optional[int] = None
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
            
            # Handle the case where mean is 0 or very small with no correct answers
            if mean <= 0 or np.isnan(mean):
                ci_low, ci_high = 0.0, 0.0
                print(f"Warning: Zero or NaN mean for {run}, setting CI to [0,0]")
            else:
                ci_low, ci_high = wilson_ci(mean, n)
                
            results[run] = {
                "mean": float(mean),  # Convert numpy float64 to Python float
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
    return results
