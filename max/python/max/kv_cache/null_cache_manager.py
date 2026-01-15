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
"""Null KV cache manager for compile-only mode.

This module provides a no-op KV cache manager that is used during compile-only
mode when running with virtual devices. It avoids GPU memory allocation while
still providing the necessary interface for graph construction.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import Tensor
from max.interfaces import RequestID, TextGenerationContext
from max.nn.kv_cache import KVCacheParams, RaggedKVCacheInputs
from max.nn.kv_cache.metrics import KVCacheMetrics

logger = logging.getLogger("max.pipelines")


class NullKVCacheManager:
    """A no-op KV cache manager for compile-only mode.

    This manager is used when compiling models with virtual devices and does not
    allocate any GPU memory. It provides dummy implementations of the KV cache
    interface to allow graph construction and compilation without requiring
    physical GPU hardware or actual memory allocation.

    This is particularly useful for cross-compilation scenarios where you want to
    compile models for GPU execution on a machine without a physical GPU present.
    """

    def __init__(
        self,
        params: KVCacheParams,
    ) -> None:
        """Initializes the null KV cache manager.

        Args:
            params: The KV cache parameters for the pipeline.
            session: The inference session for graph operations.
        """
        self.params = params
        self._metrics = KVCacheMetrics()
        self._request_to_replica_idx: dict[RequestID, int] = {}

        logger.info("Using NullKVCacheManager for compile-only mode")

    def get_replica(self, request_id: RequestID) -> int:
        """Gets the replica index for a request context.

        Args:
            request_id: The request ID to get the replica for.

        Returns:
            Always returns 0, as the null cache manager operates in single-replica mode.
        """
        return 0

    def get_or_recommend_replica(self, context: TextGenerationContext) -> int:
        """Gets or recommends a replica index for a request context.

        Args:
            context: The text generation context containing the request.

        Returns:
            Always returns 0, as the null cache manager operates in single-replica mode.
        """
        return 0

    def get_replica_request_count(self, replica_idx: int) -> int:
        """Get the number of active requests for a replica.

        Args:
            replica_idx: The replica index to query

        Returns:
            Always returns 0 for null cache manager (compile-only mode)
        """
        return 0

    def get_data_parallel_splits(
        self, batch: Sequence[TextGenerationContext]
    ) -> Sequence[Sequence[int]]:
        """Get data parallel splits for a batch.

        Args:
            batch: Batch of contexts

        Returns:
            Single split containing all batch indices
        """
        return [list(range(len(batch)))]

    def alloc(
        self,
        data: TextGenerationContext,
        num_steps: int = 1,
    ) -> None:
        """Allocates blocks for a request to run for N steps."""
        self._request_to_replica_idx[data.request_id] = 0

    def get_runtime_inputs(
        self, batch: Sequence[TextGenerationContext], num_steps: int = 1
    ) -> list[RaggedKVCacheInputs]:
        """Fetch KV cache blocks (returns dummy tensors).

        Args:
            batch: Batch of contexts
            num_steps: Number of steps to fetch

        Returns:
            List containing a single RaggedKVCacheInputs with dummy tensors

        Note:
            Tensors are kept on host since this is only used in compile-only mode
            with virtual devices that don't support device operations.
        """
        # Create dummy tensors for compilation (kept on host for virtual devices)
        dummy_blocks = Tensor.from_numpy(np.zeros((1,), dtype=np.int32))
        dummy_cache_lengths = Tensor.from_numpy(
            np.zeros((len(batch),), dtype=np.int32)
        )
        dummy_lookup_table = Tensor.from_numpy(
            np.zeros((len(batch), 1), dtype=np.int32)
        )
        dummy_max_lengths = Tensor.from_numpy(
            np.zeros((len(batch),), dtype=np.int32)
        )

        return [
            RaggedKVCacheInputs(
                blocks=dummy_blocks,
                cache_lengths=dummy_cache_lengths,
                lookup_table=dummy_lookup_table,
                max_lengths=dummy_max_lengths,
            )
        ]

    def release(self, request_id: RequestID) -> None:
        """Release cache blocks (no-op for null manager).

        Args:
            request_id: Request ID to release
        """
        self._request_to_replica_idx.pop(request_id, None)

    def claim(
        self, request_id: RequestID, replica_idx: int | None = None
    ) -> None:
        """Externally claim cache blocks (no-op for null manager).

        Args:
            request_id: Request ID
            replica_idx: Replica index (defaults to 0 if None)
        """
        self._request_to_replica_idx[request_id] = (
            replica_idx if replica_idx is not None else 0
        )

    def step(self, batch: Sequence[TextGenerationContext]) -> None:
        """Step the cache manager (no-op for null manager).

        Args:
            batch: Batch of contexts
        """
        pass

    def contains(self, request_id: RequestID) -> bool:
        """Check if a request is in the cache.

        Args:
            request_id: Request ID to check

        Returns:
            True if request is tracked, False otherwise
        """
        return request_id in self._request_to_replica_idx

    @property
    def num_free_blocks(self) -> int:
        """Get number of free blocks.

        Returns:
            Dummy value of 1000
        """
        return 1000  # dummy value

    @property
    def metrics(self) -> KVCacheMetrics:
        """Get cache metrics.

        Returns:
            Current metrics
        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self._metrics = KVCacheMetrics()

    def increment_cache_lengths(
        self,
        kv_cache_inputs: Sequence[RaggedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> Sequence[RaggedKVCacheInputs]:
        """Increment cache lengths (no-op for null manager).

        Args:
            kv_cache_inputs: Current cache state tuples
            prev_model_inputs: Previous model inputs

        Returns:
            Unchanged cache inputs (no-op implementation)
        """
        return kv_cache_inputs

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache (no-op for null manager)."""
        pass

    @property
    def free_blocks_pct(self) -> float:
        """Get percentage of free blocks.

        Returns:
            Always returns 1.0 (100%)
        """
        return 1.0

    @property
    def used_blocks_pct(self) -> float:
        """Get percentage of used blocks.

        Returns:
            Always returns 0.0 (0%)
        """
        return 0.0

    @property
    def host_committed_block_pct(self) -> float:
        """Get percentage of host committed blocks.

        Returns:
            Always returns 0.0 (0%)
        """
        return 0.0

    @property
    def total_num_host_pages(self) -> int:
        """Get total number of host pages.

        Returns:
            Always returns 0
        """
        return 0

    def get_req_blocks(self, request_id: RequestID) -> list[int]:
        """Get blocks for a request.

        Args:
            request_id: Request ID

        Returns:
            Empty list (no blocks allocated)
        """
        return []
