#!/usr/bin/env python3
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

"""
Determine whether a nightly smoke-test run qualifies for the golden container tag.

Queries the GitHub Actions API for all jobs in the current workflow run,
identifies per-model failures under the smoke-test sub-workflow, and checks
each failure against the checked-in required list
(``golden_required_list.yaml``).

Only models in the required list can block the golden tag. All other
failures are reported but do not affect eligibility.

Exit codes
    0  All required models passed (or required list is empty) → tag golden
    1  One or more required models failed, or a script error → do not tag golden

Typical invocation in a GitHub Actions step::

    uv run max/tests/integration/accuracy/smoke_tests/check_golden_eligibility.py \\
        --run-id  "$GITHUB_RUN_ID" \\
        --repo    "$GITHUB_REPOSITORY" \\
        --job-prefix "Smoke test nightly MAX Serve container /"
"""

# /// script
# dependencies = ["click>=8,<9", "requests>=2,<3", "pyyaml>=6,<7"]
# ///

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import click
import requests
import yaml

_HERE = Path(__file__).parent
DEFAULT_REQUIRED_LIST = _HERE / "golden_required_list.yaml"

# Non-zero conclusions that indicate a model failure.
# When no GPU is specified in a required-list entry, match only these two base
# configs (one per GPU architecture).  Failures on multi-GPU runners (2xB200,
# 8xB200, 8xMI355, …) are treated as optional and must be listed explicitly.
DEFAULT_GPUS = frozenset({"B200", "MI355"})

BLOCKING_CONCLUSIONS = {"failure", "timed_out", "cancelled"}


@dataclass(frozen=True)
class JobResult:
    full_name: str
    gpu: str
    model: str
    conclusion: str  # success | failure | timed_out | cancelled | skipped


@dataclass(frozen=True)
class RequiredModel:
    model: str
    gpu: str | None  # None matches DEFAULT_GPUS only
    reason: str


def load_required_list(path: Path) -> list[RequiredModel]:
    """Load and parse the golden required list from *path*."""
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        click.echo(f"[ERROR] Required list not found: {path}", err=True)
        sys.exit(1)
    except yaml.YAMLError as exc:
        click.echo(f"[ERROR] Failed to parse required list: {exc}", err=True)
        sys.exit(1)

    entries: list[RequiredModel] = []
    for item in data.get("required_for_golden", []):
        entries.append(
            RequiredModel(
                model=item["model"],
                gpu=item.get("gpu"),
                reason=item.get("reason", "(no reason given)"),
            )
        )
    return entries


def is_required(job: JobResult, required_list: list[RequiredModel]) -> bool:
    """Return True if *job* matches an entry in *required_list*.

    When an entry has no ``gpu`` field, it matches only B200 and MI355 (the two
    single-card baseline configs).  Multi-GPU results (2xB200, 8xB200, 8xMI355,
    etc.) must be listed with an explicit ``gpu`` value.
    """
    for entry in required_list:
        if entry.model.lower() != job.model.lower():
            continue
        if entry.gpu is None:
            gpu_match = job.gpu.upper() in DEFAULT_GPUS
        else:
            gpu_match = entry.gpu.lower() == job.gpu.lower()
        if gpu_match:
            return True
    return False


def iter_workflow_jobs(
    repo: str, run_id: str, token: str
) -> Iterator[dict[str, object]]:
    """Yield every job dict for *run_id*, following pagination."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url: str | None = (
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    )
    params: dict[str, str | int] = {"per_page": 100, "filter": "latest"}

    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 404:
            click.echo(
                f"[ERROR] Workflow run {run_id} not found in {repo}. "
                "Check that GITHUB_TOKEN has 'actions: read' permission.",
                err=True,
            )
            sys.exit(1)
        resp.raise_for_status()
        body = resp.json()
        yield from body.get("jobs", [])

        # Follow Link: rel="next" header for pagination.
        next_url: str | None = None
        for part in resp.headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip().strip("<>")
                break
        url = next_url
        params = {}


def parse_model_job(raw_name: str, prefix: str) -> JobResult | None:
    """
    Parse a workflow job name into a :class:`JobResult`.

    Expected format::

        "{prefix} / {GPU} - {model}"

    e.g.::

        "Smoke test nightly MAX Serve container / B200 - microsoft/Phi-4-mini-instruct"

    Returns ``None`` if the name does not match the expected pattern (e.g.,
    the "Summarize" or "Decide on models" meta-jobs).
    """
    prefix_core = prefix.rstrip("/ ").strip()
    sep = re.compile(r"^" + re.escape(prefix_core) + r"\s*/\s*(.+)$")
    m = sep.match(raw_name)
    if not m:
        return None
    inner = m.group(1).strip()

    # Split on the first " - " to get GPU and model.
    # GPU names: B200, MI355, 2xB200, 2xMI355, 4xMI355, 8xB200, 8xB200_internal
    parts = inner.split(" - ", maxsplit=1)
    if len(parts) != 2:
        return None
    gpu, model = parts[0].strip(), parts[1].strip()

    # Require non-empty model and a "/" in it (HF repo format) to filter out
    # meta-jobs like "Summarize serve smoke test results".
    if not model or "/" not in model:
        return None

    return JobResult(full_name=raw_name, gpu=gpu, model=model, conclusion="")


@click.command()
@click.option(
    "--run-id",
    required=True,
    envvar="GITHUB_RUN_ID",
    help="GitHub Actions workflow run ID to inspect.",
)
@click.option(
    "--repo",
    required=True,
    envvar="GITHUB_REPOSITORY",
    help='Repository in "owner/name" format.',
)
@click.option(
    "--job-prefix",
    required=True,
    help=(
        "Prefix of smoke-test job names, e.g. "
        '"Smoke test nightly MAX Serve container /".'
    ),
)
@click.option(
    "--required-list",
    "required_list_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_REQUIRED_LIST,
    show_default=True,
    help="Path to the golden required list YAML.",
)
@click.option(
    "--token",
    envvar="GITHUB_TOKEN",
    default="",
    help="GitHub API token (defaults to $GITHUB_TOKEN).",
)
def main(
    run_id: str,
    repo: str,
    job_prefix: str,
    required_list_path: Path,
    token: str,
) -> None:
    """Check smoke-test results and exit 0 if all required models passed."""
    if not token:
        click.echo(
            "[ERROR] No GitHub token found. Set GITHUB_TOKEN or pass --token.",
            err=True,
        )
        sys.exit(1)

    required_list = load_required_list(required_list_path)
    click.echo(
        f"Loaded {len(required_list)} required-list entr"
        f"{'y' if len(required_list) == 1 else 'ies'} from {required_list_path}"
    )

    # Fetch all jobs and filter to the smoke-test model jobs.
    click.echo(f"\nFetching jobs for run {run_id} in {repo} …")
    model_jobs: list[JobResult] = []
    for raw in iter_workflow_jobs(repo, run_id, token):
        name = raw.get("name", "")
        assert isinstance(name, str)
        parsed = parse_model_job(name, job_prefix)
        if parsed is not None:
            conclusion = raw.get("conclusion") or "in_progress"
            assert isinstance(conclusion, str)
            model_jobs.append(
                JobResult(
                    full_name=parsed.full_name,
                    gpu=parsed.gpu,
                    model=parsed.model,
                    conclusion=conclusion,
                )
            )

    if not model_jobs:
        click.echo(
            f"[ERROR] No model jobs found matching prefix '{job_prefix}'. "
            "Check --job-prefix matches the workflow's job display names.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Found {len(model_jobs)} model job(s).\n")

    # Categorise: passed / blocking-failed / not-required-failed.
    passed: list[JobResult] = []
    not_required: list[JobResult] = []
    blocking: list[JobResult] = []

    for job in sorted(model_jobs, key=lambda j: (j.gpu, j.model)):
        if job.conclusion not in BLOCKING_CONCLUSIONS:
            passed.append(job)
        elif is_required(job, required_list):
            blocking.append(job)
        else:
            not_required.append(job)

    # Print a structured summary table.
    col_w = max((len(j.model) for j in model_jobs), default=20) + 2
    header = f"  {'GPU':<14}  {'Model':<{col_w}}  Result"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for job in sorted(model_jobs, key=lambda j: (j.gpu, j.model)):
        if job.conclusion not in BLOCKING_CONCLUSIONS:
            tag = "✓ passed"
        elif is_required(job, required_list):
            tag = f"✗ BLOCKED ({job.conclusion})"
        else:
            tag = f"~ not required ({job.conclusion})"
        click.echo(f"  {job.gpu:<14}  {job.model:<{col_w}}  {tag}")

    click.echo()

    # Final verdict.
    if blocking:
        click.echo(
            f"[FAIL] {len(blocking)} required model(s) failed — "
            "container is NOT eligible for the golden tag.\n"
            "Blocking jobs:",
            err=True,
        )
        for job in blocking:
            click.echo(
                f"  • {job.gpu} - {job.model}  ({job.conclusion})", err=True
            )
        click.echo(
            "\nTo remove a model from the golden gate, delete its entry from "
            "golden_required_list.yaml.",
            err=True,
        )
        sys.exit(1)

    if not_required:
        click.echo(
            f"[WARN] {len(not_required)} failure(s) on non-required models "
            "(not blocking):"
        )
        for job in not_required:
            click.echo(f"  • {job.gpu} - {job.model}  ({job.conclusion})")
        click.echo()

    click.echo(
        f"[PASS] {len(passed)} passed, {len(not_required)} not-required, "
        f"{len(blocking)} blocking — container is eligible for the golden tag."
    )


if __name__ == "__main__":
    main()
