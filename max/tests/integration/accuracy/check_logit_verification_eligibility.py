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
Determine whether a logit-verification run is free of unexpected failures.

Reads per-runner verdict JSON files from a local directory (one file per
runner label, named ``<label>.json``), identifies models with ``error`` or
``invalid`` status, and checks each failure against the checked-in ignore list
(``logit_verification_ignore_list.yaml``).

Exit codes
    0  All failures are in the ignore list → run is eligible (no real regressions)
    1  One or more unexpected failures, or a script error → regression detected

Typical invocation in a GitHub Actions step::

    uv run max/tests/integration/accuracy/check_logit_verification_eligibility.py \\
        --verdicts-dir "$GITHUB_WORKSPACE/verdicts"
"""

# /// script
# dependencies = ["click>=8,<9", "pyyaml>=6,<7"]
# ///

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import yaml

logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
DEFAULT_IGNORE_LIST = _HERE / "logit_verification_ignore_list.yaml"

# Statuses that represent a genuine model verification failure.
BLOCKING_STATUSES = frozenset({"error", "invalid"})

# When an ignore-list entry has no ``runner`` field, it matches only these two
# single-card baseline runners.  Multi-GPU variants (intel-gpu-8xb200,
# intel-gpu-b200-multi, intel-gpu-4xmi355, etc.) must be listed explicitly.
DEFAULT_RUNNERS = frozenset({"intel-gpu-b200", "intel-gpu-mi355"})


@dataclass(frozen=True)
class ModelVerdict:
    runner: str
    model: str
    status: str


@dataclass(frozen=True)
class IgnoredFailure:
    model: str
    runner: str | None  # None → DEFAULT_RUNNERS only
    reason: str
    ticket: str | None


def load_ignore_list(path: Path) -> list[IgnoredFailure]:
    """Load and parse the logit verification ignore list from *path*."""
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        logger.error("Ignore list not found: %s", path)
        sys.exit(1)
    except yaml.YAMLError as exc:
        logger.error("Failed to parse ignore list: %s", exc)
        sys.exit(1)

    entries: list[IgnoredFailure] = []
    for item in data.get("ignored_failures", []):
        entries.append(
            IgnoredFailure(
                model=item["model"],
                runner=item.get("runner"),
                reason=item.get("reason", "(no reason given)"),
                ticket=item.get("ticket"),
            )
        )
    return entries


def is_ignored(
    verdict: ModelVerdict, ignore_list: list[IgnoredFailure]
) -> bool:
    """Return True if *verdict*'s failure is explicitly permitted by *ignore_list*.

    When an entry has no ``runner`` field it matches only the two single-card
    baseline runners (intel-gpu-b200 and intel-gpu-mi355).  Multi-GPU runner
    failures (intel-gpu-8xb200, intel-gpu-b200-multi, etc.) must be listed
    with an explicit ``runner`` value.
    """
    for entry in ignore_list:
        if entry.model.lower() != verdict.model.lower():
            continue
        if entry.runner is None:
            runner_match = verdict.runner in DEFAULT_RUNNERS
        else:
            runner_match = entry.runner.lower() == verdict.runner.lower()
        if runner_match:
            return True
    return False


def read_verdicts(verdicts_dir: Path) -> list[ModelVerdict]:
    """Read all verdict JSON files in *verdicts_dir* and return model results."""
    results: list[ModelVerdict] = []
    json_files = sorted(verdicts_dir.glob("*.json"))

    if not json_files:
        logger.error("No verdict JSON files found in '%s'.", verdicts_dir)
        sys.exit(1)

    for json_file in json_files:
        runner = json_file.stem  # e.g. "intel-gpu-8xb200"
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse '%s': %s", json_file.name, exc)
            sys.exit(1)

        if not isinstance(data, dict):
            logger.error(
                "Expected a JSON object in '%s', got %s.",
                json_file.name,
                type(data).__name__,
            )
            sys.exit(1)

        for model, model_data in data.items():
            if not isinstance(model_data, dict):
                logger.error(
                    "Expected a JSON object for model '%s' in '%s'.",
                    model,
                    json_file.name,
                )
                sys.exit(1)
            status = model_data.get("status", "unknown")
            results.append(
                ModelVerdict(runner=runner, model=model, status=status)
            )

    return results


@click.command()
@click.option(
    "--verdicts-dir",
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=Path
    ),
    help="Directory containing per-runner verdict JSON files.",
)
@click.option(
    "--ignore-list",
    "ignore_list_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_IGNORE_LIST,
    show_default=True,
    help="Path to the logit-verification ignore list YAML.",
)
def main(verdicts_dir: Path, ignore_list_path: Path) -> None:
    """Check logit-verification results and exit 0 if no unexpected failures."""
    ignore_list = load_ignore_list(ignore_list_path)
    click.echo(
        f"Loaded {len(ignore_list)} ignore-list entr"
        f"{'y' if len(ignore_list) == 1 else 'ies'} from {ignore_list_path}"
    )

    all_verdicts = read_verdicts(verdicts_dir)
    click.echo(
        f"Found {len(all_verdicts)} model result(s) across all runners.\n"
    )

    passed: list[ModelVerdict] = []
    ignored: list[ModelVerdict] = []
    blocking: list[ModelVerdict] = []

    for v in sorted(all_verdicts, key=lambda v: (v.runner, v.model)):
        if v.status not in BLOCKING_STATUSES:
            passed.append(v)
        elif is_ignored(v, ignore_list):
            ignored.append(v)
        else:
            blocking.append(v)

    # Print a summary table (only failures and ignores; omit the long ok list).
    col_w = max((len(v.model) for v in all_verdicts), default=20) + 2
    runner_w = max((len(v.runner) for v in all_verdicts), default=20) + 2
    header = f"  {'Runner':<{runner_w}}  {'Model':<{col_w}}  Result"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for v in sorted(all_verdicts, key=lambda v: (v.runner, v.model)):
        if v.status not in BLOCKING_STATUSES:
            tag = f"✓ {v.status}"
        elif is_ignored(v, ignore_list):
            entry = next(
                e
                for e in ignore_list
                if e.model.lower() == v.model.lower()
                and (e.runner is None or e.runner.lower() == v.runner.lower())
            )
            ticket = f"  [{entry.ticket}]" if entry.ticket else ""
            tag = f"~ ignored ({v.status}){ticket}"
        else:
            tag = f"✗ BLOCKED ({v.status})"
        click.echo(f"  {v.runner:<{runner_w}}  {v.model:<{col_w}}  {tag}")

    click.echo()

    if blocking:
        click.echo(
            f"[FAIL] {len(blocking)} unexpected failure(s) — "
            "run has genuine regressions.\n"
            "Blocking models:",
            err=True,
        )
        for v in blocking:
            click.echo(f"  • {v.runner} - {v.model}  ({v.status})", err=True)
        click.echo(
            "\nTo suppress a known failure, add it to "
            "logit_verification_ignore_list.yaml with a ticket reference.",
            err=True,
        )
        sys.exit(1)

    if ignored:
        click.echo(
            f"[WARN] {len(ignored)} failure(s) suppressed by the ignore list:"
        )
        for v in ignored:
            entry = next(
                e
                for e in ignore_list
                if e.model.lower() == v.model.lower()
                and (e.runner is None or e.runner.lower() == v.runner.lower())
            )
            ticket_str = f" [{entry.ticket}]" if entry.ticket else ""
            click.echo(
                f"  • {v.runner} - {v.model}: {entry.reason}{ticket_str}"
            )
        click.echo()

    n_pass = len([v for v in passed if v.status == "ok"])
    click.echo(
        f"[PASS] {n_pass} passed, {len(ignored)} ignored, "
        f"{len(blocking)} blocking — no unexpected regressions."
    )


if __name__ == "__main__":
    main()
