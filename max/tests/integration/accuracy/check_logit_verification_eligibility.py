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
``invalid`` status, and checks each failure against the checked-in required
list (``logit_verification_required_list.yaml``).

Only models in the required list can block the golden tag. All other
failures are reported but do not affect eligibility.

Exit codes
    0  All required models passed (or required list is empty) → eligible
    1  One or more required models failed, or a script error → not eligible

Typical invocation in a GitHub Actions step::

    uv run max/tests/integration/accuracy/check_logit_verification_eligibility.py \\
        --verdicts-dir "$GITHUB_WORKSPACE/verdicts"
"""

# TODO: This script and check_logit_flakes.py both read the same verdict JSON
# format and share JSON-parsing logic.  They should ideally be unified into a
# single tool (e.g. by adding an optional --required-list flag to
# check_logit_flakes.py and consolidating the verdict-reading code).  Kept
# separate for now to avoid changing check_logit_flakes.py's exit-code
# behavior and its existing callers in pipelineVerification.yaml.

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
DEFAULT_REQUIRED_LIST = _HERE / "logit_verification_required_list.yaml"

# Statuses that represent a genuine model verification failure.
BLOCKING_STATUSES = frozenset({"error", "invalid"})

# When a required-list entry has no ``runner`` field, it matches only these two
# single-card baseline runners.  Multi-GPU variants (intel-gpu-8xb200,
# intel-gpu-b200-multi, intel-gpu-4xmi355, etc.) must be listed explicitly.
DEFAULT_RUNNERS = frozenset({"intel-gpu-b200", "intel-gpu-mi355"})


@dataclass(frozen=True)
class ModelVerdict:
    runner: str
    model: str
    status: str


@dataclass(frozen=True)
class RequiredModel:
    model: str
    runner: str | None  # None → DEFAULT_RUNNERS only
    reason: str


def load_required_list(path: Path) -> list[RequiredModel]:
    """Load and parse the logit verification required list from *path*."""
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        logger.error("Failed to parse required list: %s", exc)
        sys.exit(1)

    entries: list[RequiredModel] = []
    for item in data.get("required_for_golden", []):
        entries.append(
            RequiredModel(
                model=item["model"],
                runner=item.get("runner"),
                reason=item.get("reason", "(no reason given)"),
            )
        )
    return entries


def is_required(
    verdict: ModelVerdict, required_list: list[RequiredModel]
) -> bool:
    """Return True if *verdict* matches an entry in *required_list*.

    When an entry has no ``runner`` field it matches only the two single-card
    baseline runners (intel-gpu-b200 and intel-gpu-mi355).  Multi-GPU runner
    results (intel-gpu-8xb200, intel-gpu-b200-multi, etc.) must be listed
    with an explicit ``runner`` value.
    """
    for entry in required_list:
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
        # No artifacts were uploaded — logit verification did not run (e.g. the
        # calling workflow was cancelled or skipped).  Treat as eligible so that
        # a cancelled verification run does not permanently block the golden tag.
        click.echo(
            "[WARN] No verdict JSON files found in "
            f"'{verdicts_dir}' — logit verification did not produce results "
            "(run may have been cancelled or skipped). Treating as eligible."
        )
        click.echo(
            "[PASS] 0 passed, 0 not-required, 0 blocking — no unexpected regressions."
        )
        return results

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
    "--required-list",
    "required_list_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_REQUIRED_LIST,
    show_default=True,
    help="Path to the logit-verification required list YAML.",
)
def main(verdicts_dir: Path, required_list_path: Path) -> None:
    """Check logit-verification results and exit 0 if all required models passed."""
    required_list = load_required_list(required_list_path)
    click.echo(
        f"Loaded {len(required_list)} required-list entr"
        f"{'y' if len(required_list) == 1 else 'ies'} from {required_list_path}"
    )

    all_verdicts = read_verdicts(verdicts_dir)
    click.echo(
        f"Found {len(all_verdicts)} model result(s) across all runners.\n"
    )

    passed: list[ModelVerdict] = []
    not_required: list[ModelVerdict] = []
    blocking: list[ModelVerdict] = []

    for v in sorted(all_verdicts, key=lambda v: (v.runner, v.model)):
        if v.status not in BLOCKING_STATUSES:
            passed.append(v)
        elif is_required(v, required_list):
            blocking.append(v)
        else:
            not_required.append(v)

    # Print a summary table.
    col_w = max((len(v.model) for v in all_verdicts), default=20) + 2
    runner_w = max((len(v.runner) for v in all_verdicts), default=20) + 2
    header = f"  {'Runner':<{runner_w}}  {'Model':<{col_w}}  Result"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for v in sorted(all_verdicts, key=lambda v: (v.runner, v.model)):
        if v.status not in BLOCKING_STATUSES:
            tag = f"✓ {v.status}"
        elif is_required(v, required_list):
            tag = f"✗ BLOCKED ({v.status})"
        else:
            tag = f"~ not required ({v.status})"
        click.echo(f"  {v.runner:<{runner_w}}  {v.model:<{col_w}}  {tag}")

    click.echo()

    if blocking:
        click.echo(
            f"[FAIL] {len(blocking)} required model(s) failed — "
            "run is NOT eligible for the golden tag.\n"
            "Blocking models:",
            err=True,
        )
        for v in blocking:
            click.echo(f"  • {v.runner} - {v.model}  ({v.status})", err=True)
        click.echo(
            "\nTo remove a model from the golden gate, delete its entry from "
            "logit_verification_required_list.yaml.",
            err=True,
        )
        sys.exit(1)

    if not_required:
        click.echo(
            f"[WARN] {len(not_required)} failure(s) on non-required models "
            "(not blocking):"
        )
        for v in not_required:
            click.echo(f"  • {v.runner} - {v.model}  ({v.status})")
        click.echo()

    n_pass = len([v for v in passed if v.status == "ok"])
    click.echo(
        f"[PASS] {n_pass} passed, {len(not_required)} not-required, "
        f"{len(blocking)} blocking — run is eligible for the golden tag."
    )


if __name__ == "__main__":
    main()
