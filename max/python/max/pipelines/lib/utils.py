# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Utility functions for MAX pipelines."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("max.pipelines")


class CompilationTimer:
    """Timer for logging graph build and compilation phases.

    Starts timing on initialization. Call ``mark_build_complete()`` after
    graph building, then ``done()`` after compilation to log all timings.

    Args:
        name: The name to use in log messages (e.g., "model", "vision model").

    Example:
        >>> timer = CompilationTimer("model")
        >>> graph = self._build_graph(self.weights, self.adapter)
        >>> timer.mark_build_complete()
        >>> model = session.load(graph, weights_registry=self.state_dict)
        >>> timer.done()
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._build_end_time: float = 0.0
        logger.info(f"Building and compiling {self.name}...")
        self._start_time = time.perf_counter()

    def mark_build_complete(self) -> None:
        """Mark the end of the build phase and log build time."""
        self._build_end_time = time.perf_counter()
        logger.info(
            f"Building {self.name} graph took "
            f"{self._build_end_time - self._start_time:.6f} seconds"
        )

    def done(self) -> None:
        """Log compile and total times. Call after compilation is complete."""
        end_time = time.perf_counter()
        if self._build_end_time > 0:
            logger.info(
                f"Compiling {self.name} took "
                f"{end_time - self._build_end_time:.6f} seconds"
            )
        logger.info(
            f"Building and compiling {self.name} took "
            f"{end_time - self._start_time:.6f} seconds"
        )


def upper_bounded_default(upper_bound: int, default: int | None) -> int:
    """
    Given an upper bound and an optional default value, returns a final value
    that cannot exceed the upper bound.

    Args:
        default: The default value to use, or None to use the upper bound.
        upper_bound: The upper bound to use.

    Raises:
        ValueError: If the provided default value exceeds the upper bound.

    Returns:
        The final value.
    """
    if default is None:
        return upper_bound
    elif default > upper_bound:
        raise ValueError(
            f"default value provided ({default}) exceeds the upper bound ({upper_bound})"
        )
    return default
