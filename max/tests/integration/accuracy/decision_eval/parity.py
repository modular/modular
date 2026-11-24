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

"""Compares two systems' probabilities on the same cases.

Used to check MAX against the reference implementation: same prompts and
temperatures, so the top option should agree and the probabilities should differ
only by numeric noise (bf16 on MAX, fp32 in the reference).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click
import numpy as np
from predictions import Prediction, read_predictions


def compare(
    candidate: Sequence[Prediction], reference: Sequence[Prediction]
) -> dict[str, Any]:
    """Top-option agreement and probability differences over shared cases."""
    reference_by_case = {p.case_id: p for p in reference}
    pairs = [
        (c, reference_by_case[c.case_id])
        for c in candidate
        if c.case_id in reference_by_case
    ]
    assert pairs, "the two prediction files share no cases"
    differences = []
    agree = 0
    disagreements = []
    for ours, theirs in pairs:
        assert len(ours.probabilities) == len(theirs.probabilities), (
            ours.case_id
        )
        differences.append(
            np.abs(
                np.array(ours.probabilities) - np.array(theirs.probabilities)
            )
        )
        if ours.predicted_index == theirs.predicted_index:
            agree += 1
        else:
            disagreements.append(
                {
                    "case_id": ours.case_id,
                    "candidate": ours.probabilities,
                    "reference": theirs.probabilities,
                }
            )
    flat = np.concatenate(differences)
    per_case_max = np.array([d.max() for d in differences])
    return {
        "cases": len(pairs),
        "top_option_agreement": agree / len(pairs),
        "mean_abs_diff": float(flat.mean()),
        "p99_case_max_abs_diff": float(np.percentile(per_case_max, 99)),
        "max_abs_diff": float(flat.max()),
        "disagreements": disagreements[:25],
    }


@click.command()
@click.argument("candidate", type=click.Path(path_type=Path, exists=True))
@click.argument("reference", type=click.Path(path_type=Path, exists=True))
@click.option("--min-agreement", type=float, default=0.0, show_default=True)
@click.option("--max-abs-diff", type=float, default=1.0, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(
    candidate: Path,
    reference: Path,
    min_agreement: float,
    max_abs_diff: float,
    output: Path | None,
) -> None:
    """Compares CANDIDATE and REFERENCE predictions.jsonl files."""
    report = compare(read_predictions(candidate), read_predictions(reference))
    if output is not None:
        output.write_text(json.dumps(report, indent=2))
    shown = {k: v for k, v in report.items() if k != "disagreements"}
    click.echo(json.dumps(shown, indent=2))
    click.echo(f"{len(report['disagreements'])} disagreeing cases listed")
    if report["top_option_agreement"] < min_agreement:
        raise click.ClickException(
            f"top-option agreement {report['top_option_agreement']:.4f} "
            f"is below {min_agreement}"
        )
    if report["max_abs_diff"] > max_abs_diff:
        raise click.ClickException(
            f"max |dp| {report['max_abs_diff']:.4f} exceeds {max_abs_diff}"
        )


if __name__ == "__main__":
    main()
