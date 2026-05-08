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

"""Lightweight percentile-metric types.

Carved out of :mod:`max.benchmark.benchmark_shared.metrics` so consumers
that just need the *type definitions* (e.g. dashboards deserialising
result rows from BigQuery) can import them without pulling in the
benchmark runner's heavy dependency tree (``max.serve``,
``max.diagnostics``, transformers, huggingface-hub, openai, etc.).

The full ``metrics`` module re-exports everything defined here, so
existing ``from max.benchmark.benchmark_shared.metrics import …``
imports keep working unchanged.

Imports here are kept to stdlib + ``pydantic``-free so the
``:percentile_metrics`` bazel target has a minimal dependency surface.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


def _is_finite_and_positive(value: float) -> bool:
    """Check that a numeric value is finite and positive."""
    return math.isfinite(value) and value > 0


ConfidenceLevel = Literal["high", "medium", "low", "insufficient_data"]


@dataclass
class ConfidenceInfo:
    """Confidence interval metadata for a metric.

    Attributes:
        ci_lower: Lower bound of the confidence interval (scaled units).
        ci_upper: Upper bound of the confidence interval (scaled units).
        ci_relative_width: Width of the CI as a fraction of the mean.
        confidence: Classification based on ci_relative_width.
        sample_size: Number of data points used to compute the CI.
    """

    ci_lower: float
    ci_upper: float
    ci_relative_width: float
    confidence: ConfidenceLevel
    sample_size: int


class Metrics(ABC):
    """Base class for all benchmark metric containers."""

    @abstractmethod
    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate metric values are meaningful (not 0, NaN, inf, or negative).

        Returns:
            A ``(success, errors)`` tuple where *success* is ``True`` when all
            checks pass and *errors* is a list of human-readable descriptions
            of any failed checks.
        """
        ...


@dataclass
class PercentileMetrics(Metrics):
    """Container for percentile-based metrics."""

    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    unit: str | None = None
    confidence_info: ConfidenceInfo | None = None

    def __str__(self) -> str:
        """Return a formatted string representation of the metrics in table format."""
        lines = []
        lines.append("{:<40} {:<10.2f}".format("Mean:", self.mean))
        lines.append("{:<40} {:<10.2f}".format("Std:", self.std))
        lines.append("{:<40} {:<10.2f}".format("P50:", self.p50))
        lines.append("{:<40} {:<10.2f}".format("P90:", self.p90))
        lines.append("{:<40} {:<10.2f}".format("P95:", self.p95))
        lines.append("{:<40} {:<10.2f}".format("P99:", self.p99))
        return "\n".join(lines)

    def format_with_prefix(self, prefix: str, unit: str | None = None) -> str:
        """Return formatted metrics with a custom prefix for labels."""
        # Use passed unit, or fall back to self.unit
        effective_unit = unit or self.unit
        unit_suffix = f" ({effective_unit})" if effective_unit else ""
        metrics_data = [
            ("Mean", self.mean),
            ("Std", self.std),
            ("P50", self.p50),
            ("P90", self.p90),
            ("P95", self.p95),
            ("P99", self.p99),
        ]
        return "\n".join(
            "{:<40} {:<10.2f}".format(f"{label} {prefix}{unit_suffix}:", value)
            for label, value in metrics_data
        )

    def to_flat_dict(self, name: str) -> dict[str, float]:
        """Flatten percentile stats into ``{"mean_{name}": v, ...}``.

        Note: emits ``median_{name}`` (not ``p50_{name}``) for the 50th
        percentile to preserve the legacy BigQuery column naming consumed by
        the SweepUploader path and benchmark-visibility dashboards. The
        dataclass field is named ``p50``; the legacy column name is kept
        here as the public flattening contract.
        """
        return {
            f"mean_{name}": self.mean,
            f"std_{name}": self.std,
            f"median_{name}": self.p50,
            f"p90_{name}": self.p90,
            f"p95_{name}": self.p95,
            f"p99_{name}": self.p99,
        }

    def confidence_to_flat_dict(self, name: str) -> dict[str, object]:
        """Flatten confidence-interval metadata into ``{"{name}_confidence": v, ...}``."""
        ci = self.confidence_info
        if ci is None:
            return {}
        return {
            f"{name}_ci_lower": ci.ci_lower,
            f"{name}_ci_upper": ci.ci_upper,
            f"{name}_ci_relative_width": ci.ci_relative_width,
            f"{name}_confidence": ci.confidence,
            f"{name}_sample_size": ci.sample_size,
        }

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate that the mean is finite and positive."""
        if not _is_finite_and_positive(self.mean):
            return False, [f"Invalid mean: {self.mean}"]
        return True, []
