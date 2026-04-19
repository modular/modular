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

"""Correctness metrics and reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from tabulate import tabulate


@dataclass
class CorrectnessResult:
    """Result of comparing two arrays for correctness."""

    mae: float
    max_abs_err: float
    cos_dist: float
    allclose: bool
    passed: bool
    label: str = ""


def compute_mae(
    a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]
) -> float:
    """Mean absolute error between two arrays."""
    return float(np.mean(np.abs(a - b)))


def compute_max_abs_error(
    a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]
) -> float:
    """Maximum absolute error between two arrays."""
    return float(np.max(np.abs(a - b)))


def compute_cosine_distance(
    a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]
) -> float:
    """Cosine distance (1 - cosine_similarity) between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    cos_sim = np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12
    )
    return float(1.0 - cos_sim)


def compare_outputs(
    actual: npt.NDArray[np.floating[Any]],
    expected: npt.NDArray[np.floating[Any]],
    atol: float = 1e-2,
    rtol: float = 1e-2,
    cos_threshold: float = 0.001,
) -> CorrectnessResult:
    """Compare two arrays and return correctness metrics.

    Args:
        actual: The output to test (e.g. MAX output).
        expected: The reference output (e.g. torch output).
        atol: Absolute tolerance for allclose check.
        rtol: Relative tolerance for allclose check.
        cos_threshold: Maximum cosine distance for pass.

    Returns:
        A CorrectnessResult with metrics and pass/fail status.
    """
    mae = compute_mae(actual, expected)
    max_abs_err = compute_max_abs_error(actual, expected)
    cos_dist = compute_cosine_distance(actual, expected)
    allclose = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return CorrectnessResult(
        mae=mae,
        max_abs_err=max_abs_err,
        cos_dist=cos_dist,
        allclose=allclose,
        passed=allclose and cos_dist < cos_threshold,
    )


def print_correctness_report(results: list[CorrectnessResult]) -> None:
    """Print a formatted correctness report."""
    headers = ["Shape", "MAE", "Max Abs Err", "Cos Dist", "AllClose", "Pass"]
    rows = []
    for r in results:
        rows.append(
            [
                r.label,
                f"{r.mae:.6f}",
                f"{r.max_abs_err:.6f}",
                f"{r.cos_dist:.8f}",
                str(r.allclose),
                "PASS" if r.passed else "FAIL",
            ]
        )

    print("\n  Correctness Report\n")
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="simple_outline",
            colalign=("left", "right", "right", "right", "center", "center"),
        )
    )
    print()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"  {passed}/{total} shapes passed.\n")
