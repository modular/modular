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
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
from max.interfaces import RequestID, TextGenerationContext
from max.kv_cache.paged_cache import PagedCacheInputSymbols
from max.nn.kv_cache.cache_params import KVCacheParams
from max.nn.kv_cache.manager import RaggedKVCacheInputs
from max.nn.kv_cache.metrics import KVCacheMetrics

logger = logging.getLogger("max.pipelines")


class NullKVCacheManager:
    """A no-op KV cache manager for compile-only mode.

    This manager is used when compiling for virtual devices and does not
    allocate any GPU memory. It provides dummy implementations of the
    KV cache interface to allow graph construction without actual memory
    allocation.
    """

    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: Sequence[Device],
        session: InferenceSession,
        available_cache_memory: int,
        page_size: int = 128,
    ) -> None:
        """Initialize the null KV cache manager.

        Args:
            params: KV cache parameters
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            num_layers: Number of model layers
            devices: List of devices
            session: Inference session
            available_cache_memory: Available cache memory
            page_size: Page size in tokens
        """
        self.params = params
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.devices = devices
        self.session = session
        self.available_cache_memory = available_cache_memory
        self.page_size = page_size
        self._metrics = KVCacheMetrics()
        self._request_to_replica_idx: dict[RequestID, int] = {}

        logger.info("Using NullKVCacheManager for compile-only mode")

    def get_replica(self, context: TextGenerationContext) -> int:
        """Get the replica index for a context.

        Args:
            context: Text generation context

        Returns:
            Always returns 0 (single replica)
        """
        return 0

    def get_or_recommend_replica(self, context: TextGenerationContext) -> int:
        """Get or recommend a replica index for a context.

        Args:
            context: Text generation context

        Returns:
            Always returns 0 (single replica)
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

    def maybe_reserve(
        self,
        data: TextGenerationContext,
        num_steps: int = 1,
    ) -> bool:
        """Reserve cache blocks (no-op for null manager).

        Args:
            data: Text generation context
            num_steps: Number of steps to reserve

        Returns:
            Always returns True
        """
        self._request_to_replica_idx[data.request_id] = 0
        return True

    def fetch(
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

    def input_symbols(
        self,
        devices: Sequence[Device] | None = None,
        num_layers: int | None = None,
    ) -> Sequence[PagedCacheInputSymbols]:
        """Get input symbols for graph construction.

        Args:
            devices: Devices to use (defaults to self.devices)
            num_layers: Number of layers (defaults to self.num_layers)

        Returns:
            Sequence of PagedCacheInputSymbols for graph construction
        """
        if devices is None:
            devices = self.devices
        if num_layers is None:
            num_layers = self.num_layers

        # Create symbolic tensor types for graph construction
        # Block shape: [total_num_pages, kv_dim, num_layers, page_size, n_kv_heads_per_device, head_dim]
        kv_dim = 2 if not self.params.is_mla else 1

        result = []
        for device in devices:
            kv_blocks = BufferType(
                dtype=self.params.dtype,
                shape=[
                    "total_num_pages",  # dynamic parameter
                    kv_dim,  # K and V (or just 1 for MLA)
                    num_layers,
                    self.page_size,
                    self.params.n_kv_heads_per_device,
                    self.params.head_dim,
                ],
                device=DeviceRef(device.label, device.id),
            )

            cache_lengths = TensorType(
                DType.uint32,
                shape=["batch_size"],
                device=DeviceRef(device.label, device.id),
            )

            lookup_table = TensorType(
                DType.uint32,
                shape=["batch_size", "max_num_pages"],
                device=DeviceRef(device.label, device.id),
            )

            max_lengths = TensorType(
                DType.uint32,
                shape=["batch_size", "num_steps"],
                device=DeviceRef(device.label, device.id),
            )

            result.append(
                PagedCacheInputSymbols(
                    kv_blocks=kv_blocks,
                    cache_lengths=cache_lengths,
                    lookup_table=lookup_table,
                    max_lengths=max_lengths,
                )
            )

        return result

    def release(self, request_id: RequestID) -> None:
        """Release cache blocks (no-op for null manager).

        Args:
            request_id: Request ID to release
        """
        self._request_to_replica_idx.pop(request_id, None)

    def external_claim(
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

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """Estimate memory size (returns 0 for null manager).

        Args:
            params: KV cache parameters
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            num_layers: Number of layers
            available_cache_memory: Available cache memory
            devices: List of devices
            **kwargs: Additional arguments

        Returns:
            Always returns 0 (no memory used)
        """
        return 0

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """Infer optimal batch size (returns 1 for null manager).

        Args:
            params: KV cache parameters
            max_seq_len: Maximum sequence length
            num_layers: Number of layers
            available_cache_memory: Available cache memory
            devices: List of devices
            **kwargs: Additional arguments

        Returns:
            Always returns 1
        """
        return 1

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
