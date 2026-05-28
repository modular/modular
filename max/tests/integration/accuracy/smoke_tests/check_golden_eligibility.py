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
each failure against the checked-in ignore list
(``golden_ignore_list.yaml``).

Exit codes
    0  All failures are in the ignore list → tag the container golden
    1  One or more unexpected failures, or a script error → do not tag golden

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
DEFAULT_IGNORE_LIST = _HERE / "golden_ignore_list.yaml"

# Non-zero conclusions that block golden.
# When no GPU is specified in an ignore-list entry, match only these two base
# configs (one per GPU architecture).  Failures on multi-GPU runners (2xB200,
# 8xB200, 8xMI355, …) are considered intentional and must be listed explicitly.
DEFAULT_GPUS = frozenset({"B200", "MI355"})

BLOCKING_CONCLUSIONS = {"failure", "timed_out", "cancelled"}


@dataclass(frozen=True)
class JobResult:
    full_name: str
    gpu: str
    model: str
    conclusion: str  # success | failure | timed_out | cancelled | skipped


@dataclass(frozen=True)
class IgnoredFailure:
    model: str
    gpu: str | None  # None matches any GPU
    reason: str
    ticket: str | None


def load_ignore_list(path: Path) -> list[IgnoredFailure]:
    """Load and parse the golden ignore list from *path*."""
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        click.echo(f"[ERROR] Ignore list not found: {path}", err=True)
        sys.exit(1)
    except yaml.YAMLError as exc:
        click.echo(f"[ERROR] Failed to parse ignore list: {exc}", err=True)
        sys.exit(1)

    entries: list[IgnoredFailure] = []
    for item in data.get("ignored_failures", []):
        entries.append(
            IgnoredFailure(
                model=item["model"],
                gpu=item.get("gpu"),
                reason=item.get("reason", "(no reason given)"),
                ticket=item.get("ticket"),
            )
        )
    return entries


def is_ignored(job: JobResult, ignore_list: list[IgnoredFailure]) -> bool:
    """Return True if *job*'s failure is explicitly permitted by *ignore_list*.

    When an entry has no ``gpu`` field, it matches only B200 and MI355 (the two
    single-card baseline configs).  Multi-GPU failures (2xB200, 8xB200, 8xMI355,
    etc.) must be listed with an explicit ``gpu`` value.
    """
    for entry in ignore_list:
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
    "--ignore-list",
    "ignore_list_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_IGNORE_LIST,
    show_default=True,
    help="Path to the golden ignore list YAML.",
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
    ignore_list_path: Path,
    token: str,
) -> None:
    """Check smoke-test results and exit 0 if the container is golden-eligible."""
    if not token:
        click.echo(
            "[ERROR] No GitHub token found. Set GITHUB_TOKEN or pass --token.",
            err=True,
        )
        sys.exit(1)

    ignore_list = load_ignore_list(ignore_list_path)
    click.echo(
        f"Loaded {len(ignore_list)} ignore-list entr"
        f"{'y' if len(ignore_list) == 1 else 'ies'} from {ignore_list_path}"
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

    # Categorise: passed / blocking-failed / ignored-failed.
    passed: list[JobResult] = []
    ignored: list[JobResult] = []
    blocking: list[JobResult] = []

    for job in sorted(model_jobs, key=lambda j: (j.gpu, j.model)):
        if job.conclusion not in BLOCKING_CONCLUSIONS:
            passed.append(job)
        elif is_ignored(job, ignore_list):
            ignored.append(job)
        else:
            blocking.append(job)

    # Print a structured summary table.
    col_w = max((len(j.model) for j in model_jobs), default=20) + 2
    header = f"  {'GPU':<14}  {'Model':<{col_w}}  Result"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for job in sorted(model_jobs, key=lambda j: (j.gpu, j.model)):
        if job.conclusion not in BLOCKING_CONCLUSIONS:
            tag = "✓ passed"
        elif is_ignored(job, ignore_list):
            entry = next(
                e
                for e in ignore_list
                if e.model.lower() == job.model.lower()
                and (e.gpu is None or e.gpu.lower() == job.gpu.lower())
            )
            ticket = f"  [{entry.ticket}]" if entry.ticket else ""
            tag = f"~ ignored ({job.conclusion}){ticket}"
        else:
            tag = f"✗ BLOCKED ({job.conclusion})"
        click.echo(f"  {job.gpu:<14}  {job.model:<{col_w}}  {tag}")

    click.echo()

    # Final verdict.
    if blocking:
        click.echo(
            f"[FAIL] {len(blocking)} unexpected failure(s) — "
            "container is NOT eligible for the golden tag.\n"
            "Blocking jobs:",
            err=True,
        )
        for job in blocking:
            click.echo(
                f"  • {job.gpu} - {job.model}  ({job.conclusion})", err=True
            )
        click.echo(
            "\nTo suppress a known failure, add it to golden_ignore_list.yaml "
            "with a ticket reference.",
            err=True,
        )
        sys.exit(1)

    if ignored:
        click.echo(
            f"[WARN] {len(ignored)} failure(s) were ignored per the ignore list:"
        )
        for job in ignored:
            entry = next(
                e
                for e in ignore_list
                if e.model.lower() == job.model.lower()
                and (e.gpu is None or e.gpu.lower() == job.gpu.lower())
            )
            ticket_str = f" [{entry.ticket}]" if entry.ticket else ""
            click.echo(
                f"  • {job.gpu} - {job.model}: {entry.reason}{ticket_str}"
            )
        click.echo()

    click.echo(
        f"[PASS] {len(passed)} passed, {len(ignored)} ignored, "
        f"{len(blocking)} blocking — container is eligible for the golden tag."
    )


if __name__ == "__main__":
    main()
