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

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.interfaces import RequestID, TextGenerationContext

from ..cache_params import KVCacheParams
from ..data_parallelism_utils import split_input_row_offsets, split_into_groups
from ..manager import RaggedKVCacheInputs
from .paged_cache import (
    KVCacheMetrics,
    PagedCacheInputSymbols,
    PagedKVCacheManager,
)

logger = logging.getLogger("max.pipelines")


class MultiPagedKVCacheManager(PagedKVCacheManager):
    """Enhanced PagedKVCacheManager with support for data parallelism.

    This class extends the existing PagedKVCacheManager to use MultiBlockManager,
    enabling efficient data parallelism across multiple kv cache replicas.
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
        enable_runtime_checks: bool = False,
    ) -> None:
        """Initialize the multi-device paged KV cache manager.

        Args:
            params: KV cache parameters including data parallelism settings
            max_batch_size: The maximum number of active requests that the
                manager should support. Note that this is the global maximum
                batch size across all devices, so when data parallelism is
                enabled, this would be split across all replicas of the cache.
            max_seq_len: Maximum sequence length
            num_layers: Number of model layers
            devices: The devices to use for the KV cache manager.  If data
                parallelism is enabled, the devices will be split into
                ``params.data_parallel_degree`` groups.
            session: Inference session
            available_cache_memory: Total cache memory across all devices
            page_size: Page size in tokens
            enable_runtime_checks: Whether to enable runtime checks
        """
        self.params = params
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        if params.data_parallel_degree <= 1:
            raise ValueError(
                "MultiPagedKVCacheManager requires data parallelism to be enabled"
            )

        if (
            params.enable_prefix_caching
            or params.enable_kvcache_swapping_to_host
        ):
            raise ValueError(
                "Prefix caching is not supported in MultiPagedKVCacheManager"
            )

        max_batch_size_per_replica = (
            max_batch_size // params.data_parallel_degree
        )
        cache_memory_per_replica = (
            available_cache_memory // params.data_parallel_degree
        )

        # The effective total number of pages is .
        num_replicas = params.data_parallel_degree
        assert len(devices) % num_replicas == 0, (
            "Number of devices must be divisible by number of replicas"
        )
        self.devices = devices
        self.devices_per_replica = split_into_groups(devices, num_replicas)

        self._replica_managers: list[PagedKVCacheManager] = []
        for devices in self.devices_per_replica:
            self._replica_managers.append(
                PagedKVCacheManager(
                    params=params,
                    max_batch_size=max_batch_size_per_replica,
                    max_seq_len=max_seq_len,
                    num_layers=num_layers,
                    devices=devices,
                    session=session,
                    available_cache_memory=cache_memory_per_replica,
                    page_size=page_size,
                    enable_runtime_checks=enable_runtime_checks,
                )
            )

        first_replica = self._replica_managers[0]
        self.page_size = first_replica.page_size
        self.enable_prefix_caching = first_replica.enable_prefix_caching
        self.enable_kvcache_swapping_to_host = (
            first_replica.enable_kvcache_swapping_to_host
        )
        self.total_num_pages = sum(
            manager.total_num_pages for manager in self._replica_managers
        )

        # Track requests to replicas.
        self._request_to_replica_idx: dict[RequestID, int] = {}
        self._request_count_per_replica: list[int] = [0] * num_replicas

        # Store session for model loading
        self.session = session

        # Initialize the ragged increment cache lengths model
        self.increment_cache_lengths_model = session.load(
            self._create_ragged_increment_cache_lengths_graph()
        )

    def get_replica(self, context: TextGenerationContext) -> int:
        return self._request_to_replica_idx[context.request_id]

    def get_or_recommend_replica(self, context: TextGenerationContext) -> int:
        """Return idx of the replica that should be used for the given request."""
        if context.request_id in self._request_to_replica_idx:
            return self._request_to_replica_idx[context.request_id]

        # Choose the replica with the fewest requests.
        replica_idx = min(
            range(len(self._request_count_per_replica)),
            key=self._request_count_per_replica.__getitem__,
        )
        return replica_idx

    def get_data_parallel_splits(
        self, context_batch: Sequence[TextGenerationContext]
    ) -> Tensor:
        """Constructs splits for the data parallel execution.

        Args:
            context_batch: Sequence of requests. This must already be ordered
                by replica index (so contexts that are on the same replica
                are adjacent in the batch, and the replica must be in order).

        returns:
            An int64 Tensor with shape (len(self.devices) + 1) that contains the
            number of requests on each device:
                [0, num_requests_on_replica_0, num_requests_on_replica_1, ...]
        """
        splits = np.zeros(len(self.devices) + 1, dtype=np.int64)
        for ctx in context_batch:
            replica_index = self._request_to_replica_idx[ctx.request_id]
            splits[replica_index + 1] += 1
        splits = np.cumsum(splits)  # type: ignore

        return Tensor.from_numpy(splits)

    def maybe_reserve(
        self,
        data: TextGenerationContext,
        num_steps: int = 1,
    ) -> bool:
        assert data.request_id in self._request_to_replica_idx, (
            f"Request ID {data.request_id} must already be assigned to a "
            "replica before prefetching"
        )
        replica_idx = self._request_to_replica_idx[data.request_id]
        return self._replica_managers[replica_idx].maybe_reserve(
            data, num_steps
        )

    def fetch(
        self, batch: Sequence[TextGenerationContext], num_steps: int = 1
    ) -> list[RaggedKVCacheInputs]:
        """Fetch KV cache blocks for a batch of requests.

        Args:
            batch: Batch of requests
            num_steps: Number of steps to fetch
        """

        batch_by_replica: list[list[TextGenerationContext]] = [
            [] for _ in range(len(self.devices_per_replica))
        ]

        for ctx in batch:
            replica_idx = self._request_to_replica_idx[ctx.request_id]
            batch_by_replica[replica_idx].append(ctx)

        ret_list: list[RaggedKVCacheInputs] = []
        for replica_idx, ctxs in enumerate(batch_by_replica):
            ret_list.extend(
                self._replica_managers[replica_idx].fetch(ctxs, num_steps)
            )
        return ret_list

    def input_symbols(
        self,
        devices: Sequence[Device] | None = None,
        num_layers: int | None = None,
    ) -> Sequence[PagedCacheInputSymbols]:
        input_symbols: list[PagedCacheInputSymbols] = []
        for i, devices in enumerate(self.devices_per_replica):
            symbols = self._replica_managers[i]._input_symbols(
                devices, num_layers, dynamic_dim_prefix=f"replica_{i}_"
            )
            input_symbols.extend(symbols)
        return input_symbols

    def release(self, request_id: RequestID) -> None:
        replica_idx = self._request_to_replica_idx.pop(request_id)
        self._request_count_per_replica[replica_idx] -= 1
        self._replica_managers[replica_idx].release(request_id)

    def external_claim(
        self, request_id: RequestID, replica_idx: int = -1
    ) -> None:
        """Reserve a sequence ID for the given request ID."""
        if replica_idx == -1:
            raise ValueError(
                "replica_idx must be specified for MultiPagedKVCacheManager"
            )
        if request_id in self._request_to_replica_idx:
            raise ValueError(
                f"Request ID {request_id} is already claimed for replica {self._request_to_replica_idx[request_id]}"
            )
        self._replica_managers[replica_idx].external_claim(request_id)
        self._request_to_replica_idx[request_id] = replica_idx
        self._request_count_per_replica[replica_idx] += 1

    def step(self, batch: Sequence[TextGenerationContext]) -> None:
        for ctx in batch:
            replica_idx = self._request_to_replica_idx[ctx.request_id]
            self._replica_managers[replica_idx].step([ctx])

    def contains(self, request_id: RequestID) -> bool:
        return request_id in self._request_to_replica_idx

    @property
    def num_free_blocks(self) -> int:
        """Get the set of free blocks."""
        return sum(
            [manager.num_free_blocks for manager in self._replica_managers],
            start=0,
        )

    @property
    def metrics(self) -> KVCacheMetrics:
        return sum(
            (manager.metrics for manager in self._replica_managers),
            start=KVCacheMetrics(),
        )

    def reset_metrics(self) -> None:
        for manager in self._replica_managers:
            manager.reset_metrics()

    def _create_ragged_increment_cache_lengths_graph(self) -> Graph:
        input_symbols = self.input_symbols()
        cache_lengths_types = [
            input_symbols[i][1] for i in range(len(self.devices))
        ]

        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef(self.devices[0].label, self.devices[0].id),
        )

        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=[self.params.data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        )

        with Graph(
            "update_cache_lengths",
            input_types=[
                input_row_offsets_type,
                data_parallel_splits_type,
                *cache_lengths_types,
            ],
        ) as graph:
            inp_row_offset, data_parallel_splits, *cache_lengths = (
                inp.tensor for inp in graph.inputs
            )
            split_offsets = split_input_row_offsets(
                self.params.data_parallel_degree,
                inp_row_offset,
                data_parallel_splits,
            )
            outputs = []
            start_idx = 0
            for replica_idx in range(self.params.data_parallel_degree):
                devices = self.devices_per_replica[replica_idx]

                for i, device in enumerate(devices):
                    row_offset = split_offsets[replica_idx].to(
                        DeviceRef.from_device(device)
                    )
                    cache_length = cache_lengths[start_idx + i]
                    assert isinstance(cache_length, TensorValue)
                    right_slice = row_offset[1:].rebind(cache_length.shape)
                    left_slice = row_offset[: row_offset.shape[0] - 1].rebind(
                        cache_length.shape
                    )
                    increment_amount = right_slice - left_slice
                    outputs.append(cache_length + increment_amount)
                start_idx += len(devices)
            graph.output(*outputs)

        return graph

    def increment_cache_lengths(
        self,
        kv_cache_inputs: list[RaggedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> list[RaggedKVCacheInputs]:
        """Prepares cache inputs for the next token in multistep execution.

        **Updated to handle replicas**

        Updates the cache lengths for the next inference step without requiring device
        synchronization or memory copies. This is crucial for maintaining performance
        during multi-token generation.

        Args:
            kv_cache_inputs: Current cache state tuples (blocks, lengths, lookup, max_lengths)
            prev_model_inputs: Previous model inputs including row offsets

        Returns:
            Updated cache input tuples with incremented lengths.
        """
        blocks = [kv_cache_inputs[i].blocks for i in range(len(self.devices))]
        cache_lengths = [
            kv_cache_inputs[i].cache_lengths for i in range(len(self.devices))
        ]
        lookup_table = [
            kv_cache_inputs[i].lookup_table for i in range(len(self.devices))
        ]

        assert hasattr(prev_model_inputs, "data_parallel_splits")

        # Update the cache_lengths of our batch by the previous sequence length.
        # Handle both single tensor and list of tensors for compatibility
        if isinstance(prev_model_inputs.input_row_offsets, list):
            # InternVL case: use the first tensor (row offsets are identical across devices)
            row_offsets = prev_model_inputs.input_row_offsets[0]
        else:
            # Standard case: single tensor
            row_offsets = prev_model_inputs.input_row_offsets
        row_offsets = row_offsets.to(self.devices[0])

        updated_cache_lengths = self.increment_cache_lengths_model.execute(
            row_offsets, prev_model_inputs.data_parallel_splits, *cache_lengths
        )

        start_idx = 0
        for devices in self.devices_per_replica:
            # max_lengths is ho st allocated and the same across each replica.
            max_lengths = kv_cache_inputs[start_idx].max_lengths

            # Advance to the next step of the max_lengths tensor.
            updated_max_lengths = max_lengths[1:, :]

            # Return our updated batch.
            for i in range(len(devices)):
                updated_cache_length = updated_cache_lengths[start_idx + i]
                assert isinstance(updated_cache_length, Tensor)
                kv_cache_inputs[start_idx + i] = RaggedKVCacheInputs(
                    blocks=blocks[start_idx + i],
                    cache_lengths=updated_cache_length,
                    lookup_table=lookup_table[start_idx + i],
                    max_lengths=updated_max_lengths,
                )
            start_idx += len(devices)
        return kv_cache_inputs

    def reset_prefix_cache(self) -> None:
        for manager in self._replica_managers:
            manager.reset_prefix_cache()
