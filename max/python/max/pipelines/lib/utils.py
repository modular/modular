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

import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt

# Re-exported for back-compat with callers that import CompilationTimer from
# max.pipelines.lib.utils. The canonical home is max.experimental.nn so that
# Module.compile() can use it without max.experimental depending on
# max.pipelines.
from max.experimental.nn._compilation_timer import (  # noqa: F401
    CompilationTimer,
)
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
