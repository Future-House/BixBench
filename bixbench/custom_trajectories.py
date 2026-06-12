"""Helpers for writing BixBench-compatible custom agent trajectories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CustomTrajectoryValidationError(ValueError):
    """Raised when a custom trajectory cannot be postprocessed by BixBench."""


REQUIRED_TRAJECTORY_FIELDS = frozenset(
    {
        "problem_id",
        "agent_answer",
        "ideal_answer",
        "problem",
        "mcq_options",
        "mcq_question",
        "notebook_stats",
        "num_actions",
        "question_format",
        "refusal_option",
        "metadata",
        "refusal_options",
        "nb",
        "run_name",
    }
)


def minimal_notebook(source: str = "", language: str = "python") -> dict[str, Any]:
    """Return the smallest notebook object accepted by postprocessing."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": language,
                "language": language,
                "name": language,
            },
            "language_info": {"name": language},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_custom_trajectory(
    record: Mapping[str, Any], *, require_open_distractors: bool = True
) -> dict[str, Any]:
    """Validate and normalize a custom trajectory before writing it to JSON.

    The returned dictionary uses the same top-level fields emitted by
    ``generate_trajectories.py`` so it can be loaded by ``postprocessing.py``.
    """
    missing = sorted(REQUIRED_TRAJECTORY_FIELDS.difference(record))
    if missing:
        raise CustomTrajectoryValidationError(
            f"Custom trajectory is missing required fields: {', '.join(missing)}"
        )

    normalized = dict(record)
    _require_non_empty_string(normalized, "problem_id")
    _require_non_empty_string(normalized, "run_name")
    _require_non_empty_string(normalized, "problem")

    question_format = normalized["question_format"]
    if question_format not in {"open", "mcq", "hypothesis"}:
        raise CustomTrajectoryValidationError(
            "question_format must be one of: open, mcq, hypothesis"
        )

    if not normalized["agent_answer"]:
        raise CustomTrajectoryValidationError(
            "agent_answer must be truthy; postprocessing drops empty answers"
        )
    if not normalized["ideal_answer"]:
        raise CustomTrajectoryValidationError("ideal_answer must be truthy")

    mcq_question = normalized["mcq_question"]
    if isinstance(mcq_question, list):
        if not mcq_question or not all(isinstance(item, str) for item in mcq_question):
            raise CustomTrajectoryValidationError(
                "mcq_question must be a string or a non-empty list of strings"
            )
    elif not isinstance(mcq_question, str):
        raise CustomTrajectoryValidationError(
            "mcq_question must be a string or a non-empty list of strings"
        )

    if not isinstance(normalized["mcq_options"], list):
        raise CustomTrajectoryValidationError("mcq_options must be a list")

    _require_mapping(normalized, "metadata")
    _require_mapping(normalized, "notebook_stats")
    _require_mapping(normalized, "nb")

    nb = normalized["nb"]
    if not isinstance(nb.get("cells"), list):
        raise CustomTrajectoryValidationError("nb must contain a cells list")

    if not isinstance(normalized["num_actions"], int) or normalized["num_actions"] < 0:
        raise CustomTrajectoryValidationError("num_actions must be a non-negative int")

    if require_open_distractors and question_format == "open":
        distractors = normalized["metadata"].get("distractors")
        if not isinstance(distractors, list) or not distractors:
            raise CustomTrajectoryValidationError(
                "open trajectories need metadata['distractors'] for MCQ expansion"
            )

    return normalized


def write_custom_trajectory(
    record: Mapping[str, Any],
    output_dir: str | Path,
    *,
    replica_id: int = 0,
    require_open_distractors: bool = True,
) -> Path:
    """Validate and write one trajectory JSON file.

    The filename includes ``replica_<id>`` so directory loading can infer the
    replica number consistently for single-run and batched custom agents.
    """
    if replica_id < 0:
        raise CustomTrajectoryValidationError("replica_id must be non-negative")

    validated = validate_custom_trajectory(
        record, require_open_distractors=require_open_distractors
    )
    output_path = (
        Path(output_dir)
        / f"{validated['problem_id']}_replica_{replica_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
    return output_path


def _require_non_empty_string(record: Mapping[str, Any], field: str) -> None:
    if not isinstance(record[field], str) or not record[field].strip():
        raise CustomTrajectoryValidationError(f"{field} must be a non-empty string")


def _require_mapping(record: Mapping[str, Any], field: str) -> None:
    if not isinstance(record[field], Mapping):
        raise CustomTrajectoryValidationError(f"{field} must be an object")
