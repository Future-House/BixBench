import json

import pytest

from bixbench.custom_trajectories import (
    CustomTrajectoryValidationError,
    minimal_notebook,
    validate_custom_trajectory,
    write_custom_trajectory,
)
from bixbench.postprocessing_utils import load_dataframe_from_json_directory


def valid_open_record():
    return {
        "problem_id": "capsule-001",
        "agent_answer": "The treatment increased expression by 2x.",
        "ideal_answer": "The treatment increased expression by 2x.",
        "problem": "Compare expression in control and treatment samples.",
        "mcq_options": [],
        "mcq_question": "Which conclusion is best supported?",
        "notebook_stats": {"num_cells": 1},
        "num_actions": 3,
        "question_format": "open",
        "refusal_option": True,
        "metadata": {"distractors": ["No change.", "Expression decreased."]},
        "refusal_options": None,
        "nb": minimal_notebook("print('analysis complete')"),
        "run_name": "custom-agent-smoke",
    }


def test_validate_custom_open_trajectory():
    record = validate_custom_trajectory(valid_open_record())

    assert record["problem_id"] == "capsule-001"
    assert record["nb"]["cells"][0]["source"] == "print('analysis complete')"


def test_validate_custom_open_trajectory_requires_distractors():
    record = valid_open_record()
    record["metadata"] = {}

    with pytest.raises(CustomTrajectoryValidationError, match="distractors"):
        validate_custom_trajectory(record)


def test_write_custom_trajectory_uses_replica_filename(tmp_path):
    path = write_custom_trajectory(valid_open_record(), tmp_path)

    assert path.name == "capsule-001_replica_0.json"
    assert json.loads(path.read_text())["run_name"] == "custom-agent-smoke"


def test_directory_loader_accepts_custom_file_without_replica(tmp_path):
    path = tmp_path / "capsule-001.json"
    path.write_text(json.dumps(valid_open_record()), encoding="utf-8")

    df = load_dataframe_from_json_directory(str(tmp_path))

    assert df.loc[0, "replica"] == 0
    assert df.loc[0, "problem_id"] == "capsule-001"
