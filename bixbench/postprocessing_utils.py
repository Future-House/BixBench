import copy
import json
import ast
import random
import base64
import asyncio
import re
from asyncio import Semaphore
from typing import Dict, List, Tuple, Any, Optional

import litellm
import nbconvert
import nbformat
import pandas as pd
import numpy as np
from tqdm import tqdm

# Local imports
import prompts

litellm.set_verbose = False


def notebook_to_md(nb: nbformat.NotebookNode) -> Tuple[str, Optional[Dict[str, Any]]]:
    image_counter = 1
    nb = copy.deepcopy(nb)
    # Convert non-image outputs to plain text and handle images
    for cell in nb.cells:
        if cell.get("outputs"):
            for output in cell.outputs:
                # Handle stream output type
                if output["output_type"] == "stream":
                    text_content = output.get("text", "")
                    # Truncate if needed
                    if len(text_content) > 6000:
                        first_half = text_content[:3000]
                        last_half = text_content[-3000:]
                        text_content = f"{first_half}\n...[truncated]...\n{last_half}"
                    output["text"] = text_content
                    continue

                # Handle image outputs
                if "data" in output and any(
                    key.startswith("image/") for key in output.data.keys()
                ):
                    output.data = {"text/plain": f"<{image_counter}>"}
                    image_counter += 1
                    continue

                # Convert everything else to plain text
                if "data" in output:
                    text_content = ""
                    if "text/plain" in output.data:
                        text_content = output.data["text/plain"]
                    elif "text/html" in output.data:
                        text_content = output.data["text/html"]

                    # Truncate long text outputs
                    if len(text_content) > 6000:
                        first_half = text_content[:3000]
                        last_half = text_content[-3000:]
                        text_content = f"{first_half}\n...[truncated]...\n{last_half}"

                    output.data = {"text/plain": text_content}

    markdown_exporter = nbconvert.MarkdownExporter()
    markdown_exporter.exclude_input = False
    markdown_exporter.exclude_output = False
    markdown, resources = markdown_exporter.from_notebook_node(nb)
    return markdown, resources.get("outputs", None)


async def send_message_to_llm(
    message: List[Dict[str, str]], model: str, sem: Semaphore
) -> Any:
    async with sem:
        response = await litellm.acompletion(model=model, messages=message)
        return response


models = {
    "4o": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
}


async def run_eval_loop(eval_df, max_concurrent=100):
    for model, llm_name in models.items():
        batch = eval_df.loc[eval_df.run_name.str.contains(model), "content"].tolist()
        results = await process_batch(batch, llm_name, max_concurrent=max_concurrent)
        eval_df.loc[eval_df.run_name.str.contains(model), "llm_answer"] = results
    return eval_df


def collect_notebook_stats(nb: nbformat.NotebookNode):
    """Count lines, cells, outputs, and different language usage in a Jupyter notebook."""

    stats = {
        "code_lines": 0,
        "comment_lines": 0,  # New: track comment lines separately
        "markdown_lines": 0,
        "code_cells": 0,
        "markdown_cells": 0,
        "images": 0,
        "tables": 0,
        "r_cells": 0,
        "bash_cells": 0,
        "shell_commands": 0,
    }
    try:
        for cell in nb.cells:
            # Split cell source into lines and count non-empty lines
            lines = [line for line in cell.source.split("\n") if line.strip()]

            if cell.cell_type == "code":
                stats["code_cells"] += 1

                # Process each line in code cells
                for line in lines:
                    line = line.strip()
                    # Check if line is a comment (starts with # but not #!)
                    if line.startswith("#") and not line.startswith("#!"):
                        stats["comment_lines"] += 1
                    else:
                        stats["code_lines"] += 1

                # Check for R and bash cells
                if lines:
                    first_line = lines[0].strip()
                    if first_line.startswith("%%R"):
                        stats["r_cells"] += 1
                    elif first_line.startswith("%%bash"):
                        stats["bash_cells"] += 1

                    # Count shell commands (lines starting with !)
                    stats["shell_commands"] += sum(
                        1 for line in lines if line.strip().startswith("!")
                    )

                # Check outputs for images and tables
                if hasattr(cell, "outputs"):
                    for output in cell.outputs:
                        # Check for images
                        if (
                            output.get("output_type") == "display_data"
                            or output.get("output_type") == "execute_result"
                        ):
                            if "image/png" in output.get("data", {}):
                                stats["images"] += 1

                            # Check for HTML tables or DataFrame representations
                            if "text/html" in output.get("data", {}):
                                html_content = output["data"]["text/html"]
                                if isinstance(html_content, list):
                                    html_content = "".join(html_content)
                                if "<table" in html_content:
                                    stats["tables"] += 1

                            # Check for plain text DataFrame representations
                            elif "text/plain" in output.get("data", {}):
                                text_content = output["data"]["text/plain"]
                                if isinstance(text_content, list):
                                    text_content = "".join(text_content)
                                if any(
                                    marker in text_content
                                    for marker in ["DataFrame", "Series"]
                                ):
                                    stats["tables"] += 1

            elif cell.cell_type == "markdown":
                stats["markdown_lines"] += len(lines)
                stats["markdown_cells"] += 1

                # Count markdown images
                for line in lines:
                    if "![" in line or "<img" in line:
                        stats["images"] += 1
    except Exception as e:
        print(f"Error in collect_notebook_stats: {e}")
        return {k: np.nan for k in stats.keys()}
    return stats


async def process_single(prompt: str, model: str, sem: Semaphore) -> Dict[str, Any]:
    messages = [
        {"role": "user", "content": prompt},
    ]

    for attempt in range(5):
        try:
            res = await send_message_to_llm(messages, model, sem)
            return res.choices[0].message.content
        except Exception as e:
            if attempt < 4:  # Don't print on last attempt
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
            print(f"All 5 attempts failed. Last error: {e}")
            return None


async def process_with_progress(prompt, model, sem, pbar):
    "Callback function to update the progress bar"
    try:
        result = await process_single(prompt, model, sem)
        return result
    finally:
        pbar.update(1)


async def process_batch(
    prompts: List[Dict[str, Any]], model: str, max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """Process a batch of prompts concurrently with rate limiting and progress bar"""
    sem = Semaphore(max_concurrent)

    # Setup progress bar
    pbar = tqdm(total=len(prompts), desc=f"Processing {model}")
    # Create tasks with the progress callback
    tasks = [process_with_progress(prompt, model, sem, pbar) for prompt in prompts]

    try:
        # Process tasks
        results = await asyncio.gather(*tasks)
        return results
    finally:
        # Close the progress bar
        pbar.close()


# MCQ

JUPYTER_IMAGE_OUTPUT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}

JUPYTER_TABLE_OUTPUT_TYPES_TO_IGNORE = {
    "text/latex",
    "text/html",
    "text/markdown",
}
NB_OUTPUT_LIMIT = 3000  # chars


def limit_notebook_output(output: str | list[str]) -> str:
    """Limit notebook output to configured length.

    Args:
        output: String output from notebook cell

    Returns:
        String output, truncated if longer than configured limit with
        indication of truncation
    """
    if isinstance(output, list):
        raise TypeError("Only string output truncation is supported")
    output_length = len(output)
    if output_length < NB_OUTPUT_LIMIT:
        return output
    cutoff = int(NB_OUTPUT_LIMIT / 2)
    # Sometimes error tracebacks have important information at the end
    # and at the beginning so important to keep those sections
    return output[:cutoff] + "\n<...output limited...>\n" + output[-cutoff:]


def process_cell_output(
    output, md: list[str], images: list[str], cell_streams: list[str]
) -> None:
    """Process a single output from a notebook cell."""
    if output.output_type == "stream":
        cell_streams.append(output.text)
    elif output.output_type == "execute_result":
        data = output.get("data", {}).get("text/plain", "")
        md.append(limit_notebook_output(data))
    elif output.output_type == "error":
        traceback_str = (
            "\n".join(output.traceback)
            if isinstance(output.traceback, list)
            else output.traceback
        )
        md.append(limit_notebook_output(traceback_str))
    elif output.output_type in {"display_data"}.union(JUPYTER_IMAGE_OUTPUT_TYPES):
        data_type = next(iter(output.data.keys()), "")
        if data_type in JUPYTER_TABLE_OUTPUT_TYPES_TO_IGNORE:
            return
        if data_type == "text/plain":
            md.append(limit_notebook_output(output.data[data_type]))
        elif data_type in JUPYTER_IMAGE_OUTPUT_TYPES:
            md.append(f"<{len(images) + 1}>")
            image_format = data_type.split("/")[-1]
            image_prefix = f"data:image/{image_format};base64,"
            try:
                images.append(
                    image_prefix + encode_image_to_base64(output.data[data_type])
                )
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            md.append(limit_notebook_output(output.data[data_type]))


def view_notebook(
    cells: list[nbformat.NotebookNode], language: str
) -> tuple[str, list[str]]:
    """Process notebook cells and convert them to markdown format with images.

    Args:
        cells: List of notebook cells to process
        language: Programming language of the notebook code cells

    Returns:
        tuple containing:
            - Markdown string with cell contents and outputs
            - List of base64 encoded images found in cell outputs
    """
    md: list[str] = []
    images: list[str] = []

    for idx, cell in enumerate(cells):
        md.append(f"### Cell {idx}:")
        if cell.cell_type == "code":
            md.extend((f"```{language}", cell.source, "```"))

            outputs = cell.get("outputs", [])
            if outputs:
                md.extend([f"### Output {idx}:", "```"])
                cell_streams: list[str] = []

                for output in outputs:
                    process_cell_output(output, md, images, cell_streams)

                if cell_streams:
                    combined_stream = "\n".join(cell_streams)
                    md.append(limit_notebook_output(combined_stream))
                md.append("```")
        elif cell.cell_type in {"markdown", "raw"}:
            md.append(cell.source)

    return "\n".join(md), images


def encode_image_to_base64(image: str) -> str:
    decoded_image = base64.b64decode(image)
    return base64.b64encode(decoded_image).decode("utf-8")


def load_answer(answer):
    if not answer:
        return {}
    if isinstance(answer, dict):
        return answer
    try:
        # Try literal eval first
        return ast.literal_eval(answer)
    except:
        try:
            # Fallback to json loads
            return json.loads(answer)
        except:
            # Return empty dict if parsing fails
            return {}


def create_eval_df(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a dataframe for evaluation with one row per question.
    Uses vectorized operations for better performance.
    """
    # First, apply load_answer to all relevant columns at once
    df = data.copy()

    # Handle list type agent answers
    mask = df["agent_answer"].apply(lambda x: isinstance(x, list))
    df.loc[mask, "agent_answer"] = df.loc[mask, "agent_answer"].apply(
        lambda x: {f"q{i}": v for i, v in enumerate(x)}
    )

    # Filter out rows without agent answers
    df = df[df["agent_answer"].apply(bool)]

    # Now prepare for explosion
    # Create a column with question numbers from ideal_answer keys
    df["question_keys"] = df["ideal_answer"].apply(lambda x: list(x.keys()))

    # Explode the dataframe to create one row per question
    exploded = df.explode("question_keys")

    # Now create the final dataframe in a vectorized way
    result = pd.DataFrame(
        {
            "uuid": exploded["problem_id"]
            + "_"
            + exploded["question_keys"].astype(str),
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
        }
    )

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


def questions_to_mcq(question, options: List[Dict[str, Any]], refusal_option=True):
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


def create_llm_message_content(row) -> List[Dict[str, Any]]:
    """Create content for LLM message including notebook content and images."""

    content = [{"type": "text", "text": row.prompt}]

    if row.md_images:
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


def create_prompt(row):
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


def xml_extract(text):
    match = re.search(r"<answer>([A-Z])</answer>", text)
    if match:
        return match.group(1)
    return "Z"


def majority_vote(row: pd.Series, k: int = 10) -> Optional[str]:
    # Get all predictions excluding the 'answer' column
    predictions = row[:-1]
    # Randomly sample k predictions without replacement
    sampled_votes = np.random.choice(
        predictions, size=min(k, len(predictions)), replace=False
    )
    # Get mode (most common value) of sampled votes
    # Check if all votes are integers
    if not all(isinstance(vote, (int, float, str)) for vote in sampled_votes):
        return None
    unique_values, counts = np.unique(sampled_votes, return_counts=True)

    if unique_values.size == 0:
        return None
    return unique_values[np.argmax(counts)]


def run_majority_voting(
    grouped_df: pd.DataFrame, k_values: List[int], n_trials: int
) -> Tuple[List[int], List[float], List[float]]:
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
                lambda x: majority_vote(x, k=k)
            )

            # Calculate and store accuracy
            acc = (predictions == grouped_df["correct_letter"]).mean()
            accuracies[k].append(acc)

    # Calculate means and standard deviations
    means = [np.mean(accuracies[k]) for k in k_values]
    stds = [np.std(accuracies[k]) for k in k_values]
    return k_values, means, stds


def wilson_ci(p, n, z=1.96):
    """Calculate Wilson confidence interval for a proportion."""
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - spread, center + spread
