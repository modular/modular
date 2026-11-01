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

"""Adds Brier and ECE to a Decision Index run.

The Decision Index kit (``python -m decision_index run --engine http ...``)
scores accuracy-style native metrics but has no calibration metric. Its
``results.jsonl`` keeps every answer's probabilities, so this joins them with
the rows' gold answers and scores each benchmark with :mod:`metrics`.

Only ``choice`` and ``noul`` questions with a single gold answer are scored;
requests the server refused (status other than ``ok``) are counted and left out.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
from metrics import format_table, summarize
from predictions import Prediction


def _probabilities(
    question: dict[str, Any], answer: dict[str, Any]
) -> list[float] | None:
    if question["type"] == "noul":
        yes = float(answer["noul"])
        return [1.0 - yes, yes]
    if question["type"] == "choice":
        return [
            float(answer["probabilities"][name])
            for name in question["criteria"]
        ]
    return None


def _gold_index(question: dict[str, Any], gold: Any) -> int | None:
    if question["type"] == "noul":
        return int(bool(gold)) if isinstance(gold, bool) else None
    if question["type"] == "choice" and gold in question["criteria"]:
        return list(question["criteria"]).index(gold)
    return None


def collect_predictions(
    rows: dict[str, dict[str, Any]], results: list[dict[str, Any]]
) -> tuple[dict[str, list[Prediction]], dict[str, int]]:
    """Per-benchmark predictions, and how many requests were not answered."""
    by_benchmark: dict[str, list[Prediction]] = defaultdict(list)
    not_answered: dict[str, int] = defaultdict(int)
    for result in results:
        row = rows.get(result["group_id"])
        if row is None:
            continue
        if result["status"] != "ok":
            not_answered[result["dataset"]] += 1
            continue
        for qid, question in result["payload"]["questions"].items():
            gold = _gold_index(question, row["expected"].get(qid))
            probabilities = _probabilities(
                question, result["response"]["answers"][qid]
            )
            if gold is None or probabilities is None:
                continue
            by_benchmark[result["dataset"]].append(
                Prediction(
                    case_id=f"{result['group_id']}/{qid}",
                    family_id=result["group_id"],
                    slice=result["dataset"],
                    task_type=question["type"],
                    perturbation="none",
                    option_order=list(range(len(probabilities))),
                    gold_index=gold,
                    probabilities=probabilities,
                )
            )
    return by_benchmark, not_answered


@click.command()
@click.option(
    "--rows",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="The sampled rows file the run used (.jsonl.gz).",
)
@click.option(
    "--results", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(rows: Path, results: Path, output: Path | None) -> None:
    with gzip.open(rows, "rt") as source:
        by_group = {row["id"]: row for row in map(json.loads, source)}
    # Iterate the file, not ``splitlines()``: JSON strings may hold U+2028.
    with results.open() as source:
        records = [json.loads(line) for line in source if line.strip()]
    predictions, not_answered = collect_predictions(by_group, records)
    scores = {
        name: summarize(group) for name, group in sorted(predictions.items())
    }
    everything = [p for group in predictions.values() for p in group]
    scores["all"] = summarize(everything)
    click.echo(format_table(scores))
    click.echo(f"not answered (refused or errored): {dict(not_answered)}")
    if output is not None:
        output.write_text(
            json.dumps(
                {"scores": scores, "not_answered": not_answered}, indent=2
            )
        )


if __name__ == "__main__":
    main()
