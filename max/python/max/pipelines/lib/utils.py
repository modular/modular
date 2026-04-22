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
"""Utility functions for MAX pipelines."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Generator, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from max.graph.weights import WeightData, Weights, WeightsAdapter
from transformers import AutoConfig

# Break circular import by importing PipelineConfig under TYPE_CHECKING.
if TYPE_CHECKING:
    from .config import PipelineConfig

logger = logging.getLogger("max.pipelines")

K = TypeVar("K")
V = TypeVar("V")


class BoundedCache(OrderedDict[K, V]):
    """An LRU-evicting cache backed by :class:`OrderedDict`.

    When the cache exceeds ``maxsize`` entries, the least-recently-used
    entry is evicted.  This is intended for GPU-resident tensors where
    unbounded caching can lead to OOM.

    Args:
        maxsize: Maximum number of entries to retain.
    """

    def __init__(self, maxsize: int = 32) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key: K) -> V:
        # Move to end on access so LRU ordering is maintained.
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def compute_data_parallel_splits(
    replica_batches: Sequence[Sequence[Any]],
) -> npt.NDArray[np.int64]:
    """Constructs splits for the data parallel execution.

    Args:
        replica_batches: A list of batches, each containing a sequence of contexts
        that are on the same replica.

    Returns:
        Buffer: An int64 tensor with shape (self.num_replicas + 1) that
        contains the number of requests on each device:
        [0, num_requests_on_replica_0, num_requests_on_replica_1, ...]
        or None if there is only one replica.
    """
    dp = len(replica_batches)
    splits = np.zeros(dp + 1, dtype=np.int64)
    for replica_idx, replica_batch in enumerate(replica_batches):
        splits[replica_idx + 1] += len(replica_batch)
    splits_summed = np.cumsum(splits)

    return splits_summed


class CompilationTimer:
    """Timer for logging graph build and compilation phases.

    Use as a context manager. Starts timing on entry. Call
    ``mark_build_complete()`` after graph building; timings are logged on exit.

    Args:
        name: The name to use in log messages (e.g., "model", "vision model").

    Example:
        >>> with CompilationTimer("model") as timer:
        ...     graph = self._build_graph(self.weights, self.adapter)
        ...     timer.mark_build_complete()
        ...     model = session.load(graph, weights_registry=self.state_dict)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._start_time: float | None = None
        self._build_end_time: float | None = None
        self._ctx: (
            contextlib.AbstractContextManager[CompilationTimer] | None
        ) = None

    def mark_build_complete(self) -> None:
        """Marks the end of the build phase and logs build time."""
        assert self._start_time is not None
        self._build_end_time = time.perf_counter()
        logger.info(
            f"Building {self.name} graph took "
            f"{self._build_end_time - self._start_time:.1f} seconds"
        )

    @contextlib.contextmanager
    def _run_timer(self) -> Generator[CompilationTimer, None, None]:
        self._start_time = time.perf_counter()
        self._build_end_time = None
        logger.info(f"Building and compiling {self.name}...")
        finish_event = threading.Event()
        reminder_thread = threading.Thread(
            target=self._reminder_thread_func,
            args=(finish_event,),
            daemon=True,
        )
        reminder_thread.start()
        try:
            yield self
        finally:
            end_time = time.perf_counter()
            finish_event.set()
            reminder_thread.join()
            if self._build_end_time is not None:
                logger.info(
                    f"Compiling {self.name} took "
                    f"{end_time - self._build_end_time:.1f} seconds"
                )
            logger.info(
                f"Building and compiling {self.name} took "
                f"{end_time - self._start_time:.1f} seconds"
            )
            self._start_time = None
            self._build_end_time = None

    def _reminder_thread_func(self, finish_event: threading.Event) -> None:
        assert self._start_time is not None
        while not finish_event.wait(timeout=60):
            if self._build_end_time is None:
                current_activity = "building"
                activity_start_time = self._start_time
            else:
                current_activity = "compiling"
                activity_start_time = self._build_end_time
            elapsed = time.perf_counter() - activity_start_time
            logger.info(
                f"Still {current_activity} {self.name} ({elapsed:.1f}s elapsed)"
            )

    def __enter__(self) -> CompilationTimer:
        if self._ctx is not None:
            raise RuntimeError(
                f"CompilationTimer({self.name!r}) is already active"
            )
        self._ctx = self._run_timer()
        return self._ctx.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._ctx is not None
        try:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._ctx = None


def upper_bounded_default(upper_bound: int, default: int | None) -> int:
    """Returns a value not exceeding the upper bound.

    Given an upper bound and an optional default value, returns the default
    if it is within bound, otherwise the upper bound (or raises if default
    exceeds the bound).

    Args:
        upper_bound: The upper bound to use.
        default: The default value to use, or None to use the upper bound.

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


def parse_state_dict_from_weights(
    pipeline_config: PipelineConfig,
    weights: Weights,
    adapter: WeightsAdapter | None = None,
    hf_config: AutoConfig | None = None,
) -> dict[str, WeightData]:
    """Parse the state dict from the weights, using the adapter if provided."""
    if adapter:
        if hf_config is None:
            hf_config = pipeline_config.model.huggingface_config
        return adapter(
            dict(weights.items()),
            huggingface_config=hf_config,
            pipeline_config=pipeline_config,
        )
    return {key: value.data() for key, value in weights.items()}
