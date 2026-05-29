# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for compute_accuracy_status.py and accuracy_windows_config.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from smoke_tests.accuracy_windows_config import ACCURACY_WINDOWS
from smoke_tests.compute_accuracy_status import (
    _WINDOWS,
    _get_arrived_pairs,
    _get_expected_pairs,
    _insert_infra_rows,
    _load_current_rows,
    _row_status,
)
from smoke_tests.smoke_test_github_matrix import MODELS

# ---------------------------------------------------------------------------
# accuracy_windows_config.py — structural checks
# ---------------------------------------------------------------------------


def test_all_windowed_models_in_smoke_test_matrix() -> None:
    """Every model in ACCURACY_WINDOWS must be a known smoke test model."""
    models_lower = {m.lower() for m in MODELS}
    unknown = [m for m in ACCURACY_WINDOWS if m.lower() not in models_lower]
    assert not unknown, (
        f"Models in ACCURACY_WINDOWS not found in smoke_test_github_matrix.MODELS: {unknown}"
    )


def test_active_windows_have_valid_start_and_tolerances() -> None:
    """Active windows (status=None) must have window_start and non-empty tolerances."""
    for model, w in ACCURACY_WINDOWS.items():
        if w.status is not None:
            continue
        assert w.window_start is not None, f"{model}: window_start is None"
        assert w.tolerances, (
            f"{model}: tolerances must be non-empty for active windows"
        )


def test_no_data_windows_have_no_dates() -> None:
    """Models with status='no_data' should not have window_start set."""
    for model, w in ACCURACY_WINDOWS.items():
        if w.status == "no_data":
            assert w.window_start is None, (
                f"{model}: no_data window has window_start"
            )


def test_windows_lowercase_lookup_covers_all_models() -> None:
    """_WINDOWS (lowercase keys) must have the same number of entries as ACCURACY_WINDOWS."""
    assert len(_WINDOWS) == len(ACCURACY_WINDOWS), (
        "Duplicate model names (case-insensitive) in ACCURACY_WINDOWS would cause silent drops"
    )


# ---------------------------------------------------------------------------
# _row_status — status computation logic
# ---------------------------------------------------------------------------

_TASK = "gsm8k_cot_llama"
_MODEL = "google/gemma-3-1b-it"  # has a real active window
_FLOORS = {(_MODEL, _TASK): 0.75}


def test_row_status_ok() -> None:
    assert _row_status(_MODEL, _TASK, 0.80, _FLOORS) == "ok"


def test_row_status_ok_at_floor() -> None:
    assert _row_status(_MODEL, _TASK, 0.75, _FLOORS) == "ok"


def test_row_status_invalid_below_floor() -> None:
    assert _row_status(_MODEL, _TASK, 0.74, _FLOORS) == "invalid"


def test_row_status_no_data_unknown_model() -> None:
    assert _row_status("unknown/model", _TASK, 0.80, _FLOORS) == "no_data"


def test_row_status_no_data_floor_missing() -> None:
    assert _row_status(_MODEL, _TASK, 0.80, {}) == "no_data"


def test_row_status_no_data_explicit() -> None:
    model = "nvidia/gemma-4-26b-a4b-nvfp4"  # has status="no_data" in config
    assert _row_status(model, _TASK, 0.80, {}) == "no_data"


def test_row_status_insufficient_data() -> None:
    model = "amd/kimi-k2.5-mxfp4"  # has status="insufficient_data"
    assert _row_status(model, _TASK, 0.80, {}) == "insufficient_data"


# ---------------------------------------------------------------------------
# _get_expected_pairs / _get_arrived_pairs / infra detection
# ---------------------------------------------------------------------------


def test_get_expected_pairs_b200_only() -> None:
    pairs = _get_expected_pairs("max-ci", {"B200": True}, models_override="")
    runner_labels = {runner for _, runner in pairs}
    assert runner_labels == {"modrunner-b200"}
    assert len(pairs) > 0


def test_get_expected_pairs_no_gpus() -> None:
    pairs = _get_expected_pairs("max-ci", {}, models_override="")
    assert pairs == set()


def test_get_expected_pairs_models_override_ignores_exclusions() -> None:
    # meta-llama/Llama-3.1-8B-Instruct has MULTI exclusions, so normally
    # not on 2xB200; override should bypass that.
    model = "meta-llama/Llama-3.1-8B-Instruct"
    without_override = _get_expected_pairs(
        "max-ci", {"2xB200": True}, models_override=""
    )
    with_override = _get_expected_pairs(
        "max-ci", {"2xB200": True}, models_override=model
    )
    assert (model.lower(), "modrunner-b200-2x") not in without_override
    assert (model.lower(), "modrunner-b200-2x") in with_override


def test_get_arrived_pairs() -> None:
    rows: list[dict[str, object]] = [
        {"model": "google/gemma-3-1b-it", "runner_label": "modrunner-b200"},
        {
            "model": "google/gemma-3-1b-it",
            "runner_label": "modrunner-b200",
        },  # dup
        {"model": "Qwen/Qwen3-8B", "runner_label": "modrunner-mi355"},
    ]
    assert _get_arrived_pairs(rows) == {
        ("google/gemma-3-1b-it", "modrunner-b200"),
        ("qwen/qwen3-8b", "modrunner-mi355"),
    }


def test_infra_detection_missing_model() -> None:
    expected = {
        ("google/gemma-3-1b-it", "modrunner-b200"),
        ("qwen/qwen3-8b", "modrunner-b200"),
    }
    arrived = {("google/gemma-3-1b-it", "modrunner-b200")}
    infra = expected - arrived
    assert infra == {("qwen/qwen3-8b", "modrunner-b200")}


def test_infra_detection_all_arrived() -> None:
    pairs = {("google/gemma-3-1b-it", "modrunner-b200")}
    assert pairs - pairs == set()


# ---------------------------------------------------------------------------
# _load_current_rows — artifact loading
# ---------------------------------------------------------------------------


def test_load_current_rows_reads_metadata() -> None:
    metadata = {
        "modular_metadata": [
            {
                "model": "google/gemma-3-1b-it",
                "accuracy": 0.80,
                "eval_task": "gsm8k_cot_llama",
            },
            {
                "model": "google/gemma-3-1b-it",
                "accuracy": 0.72,
                "eval_task": "chartqa",
            },
        ]
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        job_dir = results_dir / "max-ci-b200" / "gemma"
        job_dir.mkdir(parents=True)
        (job_dir / "run_metadata.json").write_text(json.dumps(metadata))

        rows = _load_current_rows(str(results_dir), "max-ci")

    assert len(rows) == 2
    assert rows[0]["model"] == "google/gemma-3-1b-it"
    assert rows[1]["eval_task"] == "chartqa"


def test_load_current_rows_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _load_current_rows(tmpdir, "max-ci")
    assert rows == []


# ---------------------------------------------------------------------------
# _insert_infra_rows — BQ interaction (mocked)
# ---------------------------------------------------------------------------


def test_insert_infra_rows_calls_insert() -> None:
    mock_job = MagicMock()
    client = MagicMock()
    client.load_table_from_json.return_value = mock_job

    _insert_infra_rows(
        client,
        "proj.dataset.table",
        {("google/gemma-3-1b-it", "modrunner-b200")},
        run_id="12345",
        git_revision="abc" * 14,
        framework="max-ci",
    )

    client.load_table_from_json.assert_called_once()
    rows, table_ref = client.load_table_from_json.call_args[0]
    assert table_ref == "proj.dataset.table"
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "google/gemma-3-1b-it"
    assert row["status"] == "infra"
    assert row["run_id"] == "12345"
    assert row["github_run_id"] == "12345"
    assert row["framework"] == "max-ci"
    assert row["gpu"] == "NVIDIA B200"
    mock_job.result.assert_called_once()


def test_insert_infra_rows_noop_when_empty() -> None:
    client = MagicMock()
    _insert_infra_rows(
        client, "proj.dataset.table", set(), "run1", "sha1", "max-ci"
    )
    client.load_table_from_json.assert_not_called()
