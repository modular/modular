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

"""JevBench cases as ``/v1/systemone`` requests, and their answers as vectors.

A JevBench case row already is a System One request: a state plus one question
with ``instructions`` and ``criteria``. This module loads the rows, picks the
ones the served model can answer, and turns each answer back into one
probability per option so every system is scored the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predictions import Prediction

QUESTION_ID = "decision"
"""The key JevBench gives its single question."""

MAX_DECIDER_OPTIONS = 10
"""Largest option count the decider prompt format serves (letters A to J)."""

# Slices that need an image or whose state the dataset withholds for licence
# reasons; neither can be sent to a text-only model.
_UNSUPPORTED_SLICE_MARKERS = ("image", "vqa_rad")


def load_cases(
    cases_dir: Path,
    slices: list[str] | None = None,
    max_options: int = MAX_DECIDER_OPTIONS,
    limit_per_slice: int | None = None,
) -> list[dict[str, Any]]:
    """Loads the cases a text-only model with ``max_options`` can answer.

    Args:
        cases_dir: The dataset's ``cases`` directory of ``<slice>.jsonl``.
        slices: Slice names to keep, or ``None`` for every supported slice.
        max_options: Cases with more options than this are skipped.
        limit_per_slice: Keeps whole families (a case and its reordered twin)
            until a slice has at least this many cases.

    Returns:
        The kept case rows, grouped by slice in file order.
    """
    kept: list[dict[str, Any]] = []
    for path in sorted(cases_dir.glob("*.jsonl")):
        name = path.stem
        if slices is not None and name not in slices:
            continue
        if slices is None and any(
            m in name for m in _UNSUPPORTED_SLICE_MARKERS
        ):
            continue
        # Iterate the file, not ``splitlines()``: states may hold U+2028.
        with path.open() as source:
            rows = [json.loads(line) for line in source if line.strip()]
        rows = [
            row
            for row in rows
            if "state" in row and row["n_options"] <= max_options
        ]
        if limit_per_slice is not None:
            rows = _first_families(rows, limit_per_slice)
        kept.extend(rows)
    assert kept, f"no usable cases under {cases_dir} for slices {slices}"
    return kept


def _first_families(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    families: list[str] = []
    for row in rows:
        if row["family_id"] not in families:
            families.append(row["family_id"])
    keep: set[str] = set()
    count = 0
    for family in families:
        if count >= limit:
            break
        keep.add(family)
        count += sum(row["family_id"] == family for row in rows)
    return [row for row in rows if row["family_id"] in keep]


def option_count(case: dict[str, Any]) -> int:
    """How many options the case's question offers."""
    return 2 if case["task_type"] == "noul" else int(case["n_options"])


def gold_index(case: dict[str, Any]) -> int:
    """Position of the gold answer among the case's options."""
    gold = case["gold"]
    kind = case["task_type"]
    if kind == "noul":
        return int(bool(gold))
    if kind == "score":
        return int(gold)
    return list(case["question"]["criteria"]).index(gold)


def systemone_request(case: dict[str, Any], model: str) -> dict[str, Any]:
    """The ``/v1/systemone`` body that asks ``case``'s question."""
    return {
        "model": model,
        "state": case["state"],
        "questions": {QUESTION_ID: case["question"]},
    }


def probabilities_from_answer(
    case: dict[str, Any], answer: dict[str, Any]
) -> list[float]:
    """One probability per option, in the case's option order."""
    kind = case["task_type"]
    if kind == "noul":
        yes = float(answer["noul"])
        return [1.0 - yes, yes]
    named = answer["probabilities"]
    if kind == "score":
        return [float(named[str(i)]) for i in range(option_count(case))]
    return [float(named[name]) for name in case["question"]["criteria"]]


def probabilities_from_published(
    case: dict[str, Any], row: dict[str, Any]
) -> list[float] | None:
    """Per-option probabilities from a published JevBench prediction row.

    Returns ``None`` for a row the system failed to answer.
    """
    if row.get("error") or row.get("pred") is None:
        return None
    if case["task_type"] == "noul":
        yes = float(row["p_true"])
        return [1.0 - yes, yes]
    return probabilities_from_answer(case, row)


def make_prediction(
    case: dict[str, Any], probabilities: list[float]
) -> Prediction:
    """Pairs ``case``'s gold and ordering with a model's probabilities."""
    assert len(probabilities) == option_count(case), (
        f"{case['case_id']}: {len(probabilities)} probabilities for "
        f"{option_count(case)} options"
    )
    return Prediction(
        case_id=case["case_id"],
        family_id=case["family_id"],
        slice=case["slice"],
        task_type=case["task_type"],
        perturbation=case["perturbation"],
        option_order=list(case["option_order"]),
        gold_index=gold_index(case),
        probabilities=probabilities,
    )
