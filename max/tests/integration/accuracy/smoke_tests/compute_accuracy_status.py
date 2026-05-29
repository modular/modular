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

# /// script
# dependencies = ["google-cloud-bigquery>=3.0,<4.0", "click>=8,<9"]
# ///

"""Compute per-row accuracy status for the current smoke test run and write to BQ.

For each (model, gpu, eval_task) row in the current CI run:
  - Look up AccuracyWindow from ACCURACY_WINDOWS config
  - If no window or status="no_data": mark as "no_data"
  - If status="insufficient_data": mark as "insufficient_data"
  - Otherwise: query MIN(accuracy) from BQ within the model's window as the
    floor and compare: accuracy >= floor → "ok", accuracy < floor → "invalid"

For each (model, runner) pair that was scheduled but produced no artifact:
  - Mark as "infra" (runner died before evaluation completed)

Then write to BQ:
  - DML INSERT one "infra" row per missing (model, runner) pair
  - UPDATE git_revision for all artifact rows in this run
  - UPDATE status per (model, gpu, eval_task) for artifact rows

Infra rows use DML INSERT (not the streaming API) so they are committed to
permanent storage immediately and the subsequent UPDATE statements can touch
them without hitting BQ's streaming-buffer restriction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow package-qualified imports without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import bigquery
from smoke_tests.accuracy_windows_config import (
    ACCURACY_WINDOWS,
    AccuracyWindow,
)
from smoke_tests.smoke_test_github_matrix import (
    MODELS,
    RUNNERS,
    excluded,
    parse_override,
)

# Normalized (lowercase-keyed) lookup so BQ's ascii_downcase values match.
_WINDOWS: dict[str, AccuracyWindow] = {
    k.lower(): v for k, v in ACCURACY_WINDOWS.items()
}
_RUNNER_TO_GPU: dict[str, str] = {
    "modrunner-b200": "NVIDIA B200",
    "modrunner-mi355": "AMD Instinct MI355X",
    "modrunner-b200-2x": "NVIDIA B200",
    "modrunner-mi355-2x": "AMD Instinct MI355X",
    "modrunner-mi355-4x": "AMD Instinct MI355X",
    "modrunner-b200-8x": "NVIDIA B200",
    "modrunner-prod-2-b200-8x": "NVIDIA B200",
}


def _load_current_rows(
    results_dir: str, framework: str
) -> list[dict[str, object]]:
    rows = []
    for path in Path(results_dir).glob(f"{framework}-*/*/run_metadata.json"):
        data = json.loads(path.read_text())
        rows.extend(data.get("modular_metadata", []))
    return rows


def _get_expected_pairs(
    framework: str,
    gpu_flags: dict[str, bool],
    models_override: str,
) -> set[tuple[str, str]]:
    """Return {(model_lower, runner_label)} for all scheduled (model, runner) pairs."""
    models = parse_override(models_override) or list(MODELS)
    ignore_exclusions = bool(models_override)
    pairs: set[tuple[str, str]] = set()
    for gpu_name, runner_label in RUNNERS.items():
        if not gpu_flags.get(gpu_name, False):
            continue
        for model in models:
            if ignore_exclusions or not excluded(framework, gpu_name, model):
                pairs.add((model.lower(), runner_label))
    return pairs


def _get_arrived_pairs(rows: list[dict[str, object]]) -> set[tuple[str, str]]:
    """Return {(model_lower, runner_label)} for jobs that produced artifact rows."""
    return {
        (str(row["model"]).lower(), str(row["runner_label"])) for row in rows
    }


def _ensure_columns(client: bigquery.Client, table_ref: str) -> None:
    """Add status and git_revision columns if they don't already exist."""
    client.query(f"""
        ALTER TABLE `{table_ref}`
        ADD COLUMN IF NOT EXISTS status STRING,
        ADD COLUMN IF NOT EXISTS git_revision STRING
    """).result()


def _query_floors(
    client: bigquery.Client,
    floor_table: str,
    windows: list[tuple[str, AccuracyWindow]],
    commit_date: str,
) -> dict[tuple[str, str], float]:
    """Return {(model_lower, eval_task): floor} for models with tolerances.

    Floor = MAX(accuracy from window_start to commit_date) - tolerance.
    Only models with tolerances defined are included.
    """
    windows_with_tolerances = [(m, w) for m, w in windows if w.tolerances]
    if not windows_with_tolerances:
        return {}

    clauses = "\n    OR ".join(
        f"(LOWER(model) = '{m}' AND timestamp >= '{w.window_start}')"
        for m, w in windows_with_tolerances
    )
    query = f"""
        SELECT LOWER(model) AS model, eval_task, MAX(accuracy) AS max_accuracy
        FROM `{floor_table}`
        WHERE LOWER(framework) = 'max-ci'
          AND accuracy > 0.0
          AND COALESCE(status, '') != 'infra'
          AND timestamp <= '{commit_date}'
          AND ({clauses})
        GROUP BY 1, 2
    """
    floors: dict[tuple[str, str], float] = {}
    for row in client.query(query).result():
        model_lower = str(row.model)
        eval_task = str(row.eval_task)
        window = _WINDOWS.get(model_lower)
        if window and window.tolerances:
            tolerance = window.tolerances.get(eval_task)
            if tolerance is not None:
                floors[(model_lower, eval_task)] = (
                    float(row.max_accuracy) - tolerance
                )
    return floors


def _q(s: str) -> str:
    """Wrap a string in single quotes for inline SQL. Values are CI-internal."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _row_status(
    model_lower: str,
    eval_task: str,
    accuracy: float,
    floors: dict[tuple[str, str], float],
) -> str:
    """Return the status string for one (model, eval_task, accuracy) triple."""
    window = _WINDOWS.get(model_lower)
    if window is None:
        return "no_data"
    if window.status == "no_data":
        return "no_data"
    if window.status == "insufficient_data":
        return "insufficient_data"
    if not window.tolerances:
        return "no_data"
    floor = floors.get((model_lower, eval_task))
    if floor is None:
        return "no_data"
    return "invalid" if accuracy < floor else "ok"


def _insert_infra_rows(
    client: bigquery.Client,
    table_ref: str,
    infra_pairs: set[tuple[str, str]],
    run_id: str,
    git_revision: str,
    framework: str,
) -> None:
    """Insert one infra row per (model, runner) pair that produced no artifact.

    Uses load_table_from_json so rows are committed to permanent storage
    immediately, allowing the subsequent UPDATE statements to touch them.
    ignore_unknown_values=True handles schema differences between staging and
    prod tables gracefully.
    """
    if not infra_pairs:
        return
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "id": str(uuid.uuid4()),
            "framework": framework,
            "framework_version": "",
            "gpu": _RUNNER_TO_GPU.get(runner_label, runner_label),
            "gpu_count": 0,
            "model": model,
            "run_id": run_id,
            "startup_time_seconds": 0.0,
            "timestamp": now,
            "eval_task": "",
            "task_type": "",
            "task_hash": "",
            "accuracy": 0.0,
            "accuracy_stderr": 0.0,
            "total_evaluation_time_seconds": 0.0,
            "runner_label": runner_label,
            "status": "infra",
            "git_revision": git_revision,
            "github_run_id": run_id,
        }
        for model, runner_label in sorted(infra_pairs)
    ]
    job = client.load_table_from_json(
        rows,
        table_ref,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            ignore_unknown_values=True,
        ),
    )
    job.result()
    print(f"Inserted {len(infra_pairs)} infra rows:")
    for model, runner_label in sorted(infra_pairs):
        print(f"  {model} [{_RUNNER_TO_GPU.get(runner_label, runner_label)}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accuracy status for a smoke test run and update BQ."
    )
    parser.add_argument(
        "--results-dir", required=True, help="Artifact download dir"
    )
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--git-revision", required=True, help="Full git SHA")
    parser.add_argument("--framework", default="max-ci")
    parser.add_argument(
        "--dataset",
        required=True,
        help="BQ dataset (raw_prod_data or raw_stage_data)",
    )
    parser.add_argument("--project", default="modular-metrics")
    parser.add_argument("--table", default="serve_smoke_test_results")
    # GPU flags — mirror serveSmokeTestBase.yaml inputs
    parser.add_argument("--run-on-b200", action="store_true")
    parser.add_argument("--run-on-mi355", action="store_true")
    parser.add_argument("--run-on-2xb200", action="store_true")
    parser.add_argument("--run-on-2xmi355", action="store_true")
    parser.add_argument("--run-on-8xb200", action="store_true")
    parser.add_argument("--run-on-4xmi355", action="store_true")
    parser.add_argument("--run-on-8xb200-internal", action="store_true")
    parser.add_argument("--models-override", default="")
    args = parser.parse_args()

    gpu_flags = {
        "B200": args.run_on_b200,
        "MI355": args.run_on_mi355,
        "2xB200": args.run_on_2xb200,
        "2xMI355": args.run_on_2xmi355,
        "4xMI355": args.run_on_4xmi355,
        "8xB200": args.run_on_8xb200,
        "8xB200_internal": args.run_on_8xb200_internal,
    }

    # Resolve commit date from git SHA for window upper bound.
    try:
        commit_date = subprocess.check_output(
            ["git", "log", "--format=%ci", "-1", args.git_revision],
            text=True,
        ).strip()[:10]
    except subprocess.CalledProcessError:
        commit_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    table_ref = f"{args.project}.{args.dataset}.{args.table}"
    floor_table = f"{args.project}.raw_prod_data.{args.table}"
    client = bigquery.Client(project=args.project)

    _ensure_columns(client, table_ref)

    rows = _load_current_rows(args.results_dir, args.framework)

    # Detect and insert infra failures (scheduled jobs that produced no artifact).
    expected = _get_expected_pairs(
        args.framework, gpu_flags, args.models_override
    )
    arrived = _get_arrived_pairs(rows)
    infra_pairs = expected - arrived
    _insert_infra_rows(
        client,
        table_ref,
        infra_pairs,
        args.run_id,
        args.git_revision,
        args.framework,
    )

    if not rows:
        print("No artifact rows found — only infra rows written.")
        return

    # Collect unique models that need floor queries (active windows only).
    seen: set[str] = set()
    windows_to_query: list[tuple[str, AccuracyWindow]] = []
    for row in rows:
        model_lower = str(row["model"]).lower()
        if model_lower in seen:
            continue
        seen.add(model_lower)
        window = _WINDOWS.get(model_lower)
        if window and window.status is None:
            windows_to_query.append((model_lower, window))

    floors = _query_floors(client, floor_table, windows_to_query, commit_date)

    # Compute status per (model, gpu, eval_task).
    status_rows: list[dict[str, object]] = []
    for row in rows:
        model_lower = str(row["model"]).lower()
        eval_task = str(row["eval_task"])
        status_rows.append(
            {
                "model": model_lower,
                "gpu": str(row["gpu"]),
                "eval_task": eval_task,
                "status": _row_status(
                    model_lower,
                    eval_task,
                    float(str(row["accuracy"])),
                    floors,
                ),
            }
        )

    # UPDATE 1: stamp git_revision on every row in this run (including infra rows).
    client.query(
        f"UPDATE `{table_ref}` SET git_revision = @rev WHERE run_id = @rid",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "rev", "STRING", args.git_revision
                ),
                bigquery.ScalarQueryParameter("rid", "STRING", args.run_id),
            ]
        ),
    ).result()
    print(f"Set git_revision={args.git_revision[:12]} for run_id={args.run_id}")

    # UPDATE 2: set status per (model, gpu, eval_task) for artifact rows.
    values_sql = "\nUNION ALL\n".join(
        f"  SELECT {_q(str(s['model']))} AS model, {_q(str(s['gpu']))} AS gpu,"
        f" {_q(str(s['eval_task']))} AS eval_task, {_q(str(s['status']))} AS status"
        for s in status_rows
    )
    client.query(f"""
        UPDATE `{table_ref}` AS t
        SET t.status = s.status
        FROM ({values_sql}) AS s
        WHERE t.run_id = {_q(args.run_id)}
          AND LOWER(t.model) = s.model
          AND t.gpu = s.gpu
          AND t.eval_task = s.eval_task
    """).result()

    ok_count = sum(1 for s in status_rows if s["status"] == "ok")
    invalid_count = sum(1 for s in status_rows if s["status"] == "invalid")
    other_count = len(status_rows) - ok_count - invalid_count
    print(
        f"Status summary: {ok_count} ok, {invalid_count} invalid,"
        f" {other_count} other (no_data/insufficient_data),"
        f" {len(infra_pairs)} infra"
    )
    if invalid_count:
        print("Invalid rows:")
        for s in status_rows:
            if s["status"] == "invalid":
                print(f"  {s['model']} [{s['gpu']}] {s['eval_task']}")


if __name__ == "__main__":
    main()
