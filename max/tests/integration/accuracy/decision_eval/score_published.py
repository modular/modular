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

"""Scores a published JevBench system on the cases this harness runs.

Reads ``predictions/<system>/predictions.jsonl`` from the JevBench dataset and
scores it with the same metrics as MAX, restricted to the cases that
:mod:`run_jevbench` would send, so the numbers sit side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from cases import (
    load_cases,
    make_prediction,
    probabilities_from_published,
)
from metrics import format_table, summarize_by_slice
from predictions import write_predictions


@click.command()
@click.option(
    "--cases-dir", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--published",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="A predictions.jsonl from the dataset.",
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--max-options", type=click.IntRange(min=1), default=10, show_default=True
)
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Score only the cases the published file answers. The result is "
    "then not comparable to a run that answers every selected case.",
)
def main(
    cases_dir: Path,
    published: Path,
    output_dir: Path,
    max_options: int,
    allow_partial: bool,
) -> None:
    cases = {
        c["case_id"]: c for c in load_cases(cases_dir, max_options=max_options)
    }
    predictions = []
    with published.open() as source:
        published_rows = [json.loads(line) for line in source if line.strip()]
    for row in published_rows:
        case = cases.get(row["case_id"])
        if case is None:
            continue
        probabilities = probabilities_from_published(case, row)
        if probabilities is not None:
            predictions.append(make_prediction(case, probabilities))
    assert predictions, f"{published} answers none of the selected cases"
    answered = {p.case_id for p in predictions}
    missing = sorted(set(cases) - answered)
    if missing and not allow_partial:
        raise click.ClickException(
            f"{published} does not answer {len(missing)} of {len(cases)} "
            f"selected cases (first: {missing[:3]}), so its scores would "
            "cover an easier or different subset than MAX; pass "
            "--allow-partial to score the overlap anyway"
        )
    scores = summarize_by_slice(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.jsonl", predictions)
    (output_dir / "scores.json").write_text(
        json.dumps(
            {
                "published": str(published),
                "complete": not missing,
                "answered": len(predictions),
                "selected": len(cases),
                "scores": scores,
            },
            indent=2,
        )
    )
    click.echo(format_table(scores))
    click.echo(f"scored {len(predictions)} of {len(cases)} selected cases")


if __name__ == "__main__":
    main()
