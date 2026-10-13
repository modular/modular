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

"""Accuracy, calibration and order-sensitivity metrics for decision models.

Conventions match the published JevBench scores: Brier is the sum over options
of squared error (averaged over cases), ECE uses 10 equal-width bins on the top
probability, and accuracy is the top option against the gold option.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from predictions import Prediction

ECE_BINS = 10
_NLL_CLIP = 1e-12


def brier(prediction: Prediction) -> float:
    """Sum over options of squared error against the one-hot gold."""
    return sum(
        (p - (1.0 if i == prediction.gold_index else 0.0)) ** 2
        for i, p in enumerate(prediction.probabilities)
    )


def expected_calibration_error(
    predictions: Sequence[Prediction], bins: int = ECE_BINS
) -> float:
    """Gap between top-probability confidence and accuracy, over equal bins."""
    assert predictions, "ECE needs at least one prediction"
    totals = [0] * bins
    confidence = [0.0] * bins
    correct = [0.0] * bins
    for prediction in predictions:
        top = max(prediction.probabilities)
        # Bin edges are open on the left: (lo, hi].
        index = min(max(math.ceil(top * bins) - 1, 0), bins - 1)
        totals[index] += 1
        confidence[index] += top
        correct[index] += prediction.predicted_index == prediction.gold_index
    return sum(
        abs(confidence[i] - correct[i]) / len(predictions)
        for i in range(bins)
        if totals[i]
    )


def order_flip_rate(
    predictions: Sequence[Prediction],
) -> dict[str, float] | None:
    """How often reordering the options changes which option is picked.

    Pairs each ``none`` case with its ``permuted`` twin (same family) and maps
    both picks back to the original option positions.

    Returns:
        ``families`` (pairs found) and ``flipped`` (fraction that disagree),
        or ``None`` when there are no pairs.
    """
    by_family: dict[str, dict[str, Prediction]] = defaultdict(dict)
    for prediction in predictions:
        by_family[prediction.family_id][prediction.perturbation] = prediction
    pairs = [
        (twins["none"], twins["permuted"])
        for twins in by_family.values()
        if "none" in twins and "permuted" in twins
    ]
    if not pairs:
        return None
    flipped = sum(
        plain.option_order[plain.predicted_index]
        != permuted.option_order[permuted.predicted_index]
        for plain, permuted in pairs
    )
    return {"families": len(pairs), "flipped": flipped / len(pairs)}


def summarize(predictions: Sequence[Prediction]) -> dict[str, Any]:
    """Scores one group of predictions."""
    assert predictions, "nothing to summarize"
    count = len(predictions)
    negative_log_likelihood = sum(
        -math.log(max(p.probabilities[p.gold_index], _NLL_CLIP))
        for p in predictions
    )
    summary: dict[str, Any] = {
        "n": count,
        "accuracy": sum(p.predicted_index == p.gold_index for p in predictions)
        / count,
        "brier": sum(brier(p) for p in predictions) / count,
        "nll": negative_log_likelihood / count,
        "ece": expected_calibration_error(predictions),
        "chance": sum(1.0 / len(p.probabilities) for p in predictions) / count,
    }
    flips = order_flip_rate(predictions)
    if flips is not None:
        summary["order_flip_rate"] = flips["flipped"]
        summary["order_families"] = flips["families"]
    return summary


def summarize_by_slice(
    predictions: Sequence[Prediction],
) -> dict[str, dict[str, Any]]:
    """Scores each slice separately, plus an ``all`` row over every case."""
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.slice].append(prediction)
    scores = {name: summarize(group) for name, group in sorted(grouped.items())}
    scores["all"] = summarize(predictions)
    return scores


def format_table(scores: dict[str, dict[str, Any]]) -> str:
    """Renders :func:`summarize_by_slice` output as a Markdown table."""
    lines = [
        "| slice | n | accuracy | chance | brier | ece | nll | order flips |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in scores.items():
        flips = row.get("order_flip_rate")
        lines.append(
            f"| {name} | {row['n']} | {row['accuracy']:.3f} | "
            f"{row['chance']:.3f} | {row['brier']:.3f} | {row['ece']:.3f} | "
            f"{row['nll']:.3f} | {'' if flips is None else f'{flips:.3f}'} |"
        )
    return "\n".join(lines)
