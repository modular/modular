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

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, ops
from max.interfaces import RequestID, TextGenerationContext
from max.nn.legacy.comm import Signals
from max.nn.legacy.kv_cache import KVCacheParams, RaggedKVCacheInputs
from max.nn.legacy.kv_cache.data_parallelism_utils import (
    split_input_row_offsets,
    split_into_groups,
)
from max.nn.legacy.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

from .tp_cache_manager import _TPPagedKVCacheManager

logger = logging.getLogger("max.pipelines")


class PagedKVCacheManager:
    """Paged KVCache manager with data and tensor parallelism support.

    .. code-block:: python

        # Allocate metadata for requests in batch
        kv_manager.claim(ctx1.request_id, replica_idx=0)
        kv_manager.claim(ctx2.request_id, replica_idx=1)

        # Allocate blocks for these requests
        kv_manager.alloc(ctx1, replica_idx=0, num_steps=10)
        kv_manager.alloc(ctx2, replica_idx=1, num_steps=10)

        # Get KVCache inputs to feed to graph
        kv_cache_inputs = kv_manager.get_runtime_inputs(
            [[ctx1, ctx2]], num_steps=10
        )

        # Run model...
        # Update requests with newly generated tokens
        ctx1.update(42)
        ctx2.update(42)

        # Commit newly written blocks to prefix cache
        kv_manager.step([[ctx1, ctx2]])

        # Release metadata and KV blocks for these requests
        kv_manager.release(ctx1.request_id, replica_idx=0)
        kv_manager.release(ctx2.request_id, replica_idx=1)
    """

    def __init__(
        self,
        params: KVCacheParams,
        session: InferenceSession,
        total_num_pages: int,
        total_num_host_pages: int = 0,
        enable_runtime_checks: bool = False,
    ) -> None:
        """Initialize the multi-device paged KV cache manager.

        Args:
            params: KV cache parameters including data parallelism settings
            session: The MAX Engine inference session
            total_num_pages: The total number of pages to allocate
            total_num_host_pages: The total number of host pages to allocate
            enable_runtime_checks: Whether to enable runtime checks
        """
        self.params = params
        self.devices = [d.to_device() for d in params.devices]

        self.num_replicas = params.data_parallel_degree
        assert len(self.devices) % self.num_replicas == 0, (
            "Number of devices must be divisible by number of replicas"
        )
        self.devices_per_replica = split_into_groups(
            self.devices, self.num_replicas
        )

        self._replica_managers: list[_TPPagedKVCacheManager] = []
        dp_1_params = params.copy_as_dp_1()
        for devices in self.devices_per_replica:
            self._replica_managers.append(
                _TPPagedKVCacheManager(
                    params=dp_1_params,
                    total_num_pages=total_num_pages,
                    total_num_host_pages=total_num_host_pages,
                    devices=devices,
                    session=session,
                    enable_runtime_checks=enable_runtime_checks,
                )
            )

        first_replica = self._replica_managers[0]
        self.page_size = first_replica.page_size
        self.enable_prefix_caching = first_replica.enable_prefix_caching
        self.enable_kvcache_swapping_to_host = (
            first_replica.enable_kvcache_swapping_to_host
        )

        # Store session for model loading
        self.session = session

        # Enable broadcast for row_offset transfers when DP=1 with multiple devices.
        # - DP=1 check: DP>1 requires scatter semantics (different data per replica),
        #   not yet supported. See SERVOPT-970 for DP>1 broadcast support.
        # - len(devices)>1 check: single-device models don't provide signal_buffers
        #   in their ModelInputs. The broadcast kernel is a no-op for single device,
        #   but we need signal_buffers as graph inputs to call it.
        self._use_broadcast = (
            self.params.data_parallel_degree == 1 and len(self.devices) > 1
        )

        # Initialize the ragged increment cache lengths model
        self.increment_cache_lengths_model = session.load(
            self._create_ragged_increment_cache_lengths_graph()
        )

    def get_pct_used_blocks_after_allocation(
        self, ctx: TextGenerationContext, replica_idx: int, num_steps: int = 1
    ) -> float:
        """Get the percentage of blocks used after allocating for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_steps: Number of additional steps to allocate blocks for. Defaults to 1.

        Returns:
            The percentage of total blocks used after allocating for the request.
        """
        return self._replica_managers[
            replica_idx
        ].get_pct_used_blocks_after_allocation(ctx, num_steps)

    def alloc(
        self,
        data: TextGenerationContext,
        replica_idx: int,
        num_steps: int = 1,
    ) -> None:
        """Allocates blocks for a request to run for N steps.

        This method allocates blocks needed by a request to run for N steps.
        When prefix caching is enabled, some of the allocated blocks may be
        retrieved from the prefix cache.

        Args:
            data: The text generation context for the request. The request ID
                must already be assigned to a replica via `claim`.
            num_steps: The number of steps to reserve blocks for. Default: 1.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
        return self._replica_managers[replica_idx].alloc(data, num_steps)

    def get_runtime_inputs(
        self,
        batches: Sequence[Sequence[TextGenerationContext]],
        num_steps: int = 1,
    ) -> list[RaggedKVCacheInputs]:
        """Get the graph inputs for per-replica batches of requests.

        This method will raise a RuntimeError if any request has insufficient blocks
        already allocated to it to run for the given number of steps.

        Args:
            batches: Per-replica batches of requests
            num_steps: Number of steps to run for
        """
        ret_list: list[RaggedKVCacheInputs] = []
        for replica, ctxs in zip(self._replica_managers, batches, strict=True):
            ret_list.extend(replica.get_runtime_inputs(ctxs, num_steps))
        return ret_list

    def release(self, request_id: RequestID, replica_idx: int) -> None:
        self._replica_managers[replica_idx].release(request_id)

    def claim(self, request_id: RequestID, replica_idx: int) -> None:
        """Reserve a sequence ID for the given request ID."""
        self._replica_managers[replica_idx].claim(request_id)

    def step(self, batches: Sequence[Sequence[TextGenerationContext]]) -> None:
        """Commit new tokens into the prefix cache for per-replica batches."""
        for replica, ctxs in zip(self._replica_managers, batches, strict=True):
            replica.step(ctxs)

    def contains(self, request_id: RequestID, replica_idx: int) -> bool:
        return self._replica_managers[replica_idx].contains(request_id)

    def reset_metrics(self) -> None:
        for manager in self._replica_managers:
            manager.reset_metrics()

    def _create_ragged_increment_cache_lengths_graph(self) -> Graph:
        input_symbols = self.params.get_symbolic_inputs()
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

        # Build input types list
        input_types: list[TensorType | BufferType] = [
            input_row_offsets_type,
            data_parallel_splits_type,
            *cache_lengths_types,
        ]

        # Add signal buffer types when using broadcast
        signal_buffer_types: list[BufferType] = []
        if self._use_broadcast:
            device_refs = [DeviceRef(d.label, d.id) for d in self.devices]
            signals = Signals(devices=device_refs)
            signal_buffer_types = signals.input_types()
            input_types.extend(signal_buffer_types)

        with Graph(
            "update_cache_lengths",
            input_types=input_types,
        ) as graph:
            # Unpack inputs
            num_fixed_inputs = 2 + len(
                self.devices
            )  # row_offsets + splits + cache_lengths
            inp_row_offset, data_parallel_splits, *cache_lengths = [
                inp.tensor for inp in graph.inputs[:num_fixed_inputs]
            ]

            # Unpack signal buffers if using broadcast
            signal_buffers = None
            if self._use_broadcast:
                signal_buffers = [
                    inp.buffer for inp in graph.inputs[num_fixed_inputs:]
                ]

            split_offsets = split_input_row_offsets(
                self.params.data_parallel_degree,
                inp_row_offset,
                data_parallel_splits,
            )
            outputs = []
            start_idx = 0
            for replica_idx in range(self.params.data_parallel_degree):
                devices = self.devices_per_replica[replica_idx]

                # Use broadcast to transfer row_offset to all devices in parallel.
                # Currently only enabled for DP=1 (single replica with all devices).
                if self._use_broadcast:
                    assert signal_buffers is not None
                    row_offsets_per_device = ops.distributed_broadcast(
                        split_offsets[replica_idx], signal_buffers
                    )
                    for i in range(len(devices)):
                        row_offset = row_offsets_per_device[i]
                        cache_length = cache_lengths[start_idx + i]
                        assert isinstance(cache_length, TensorValue)
                        right_slice = row_offset[1:].rebind(cache_length.shape)
                        left_slice = row_offset[
                            : row_offset.shape[0] - 1
                        ].rebind(cache_length.shape)
                        increment_amount = right_slice - left_slice
                        outputs.append(cache_length + increment_amount)
                else:
                    # Fall back to sequential .to(device) transfers for DP>1.
                    # TODO(SERVOPT-970): Replace with scatter+broadcast for DP>1.
                    for i, device in enumerate(devices):
                        row_offset = split_offsets[replica_idx].to(
                            DeviceRef.from_device(device)
                        )
                        cache_length = cache_lengths[start_idx + i]
                        assert isinstance(cache_length, TensorValue)
                        right_slice = row_offset[1:].rebind(cache_length.shape)
                        left_slice = row_offset[
                            : row_offset.shape[0] - 1
                        ].rebind(cache_length.shape)
                        increment_amount = right_slice - left_slice
                        outputs.append(cache_length + increment_amount)
                start_idx += len(devices)
            graph.output(*outputs)

        return graph

    @traced
    def increment_cache_lengths(
        self,
        kv_cache_inputs: Sequence[RaggedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> Sequence[RaggedKVCacheInputs]:
        """Prepares cache inputs for the next token in multistep execution.

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

        if self.params.data_parallel_degree > 1:
            data_parallel_splits = prev_model_inputs.data_parallel_splits
        else:
            batch_size = cache_lengths[0].shape[0]
            data_parallel_splits = Buffer.from_numpy(
                np.array([0, batch_size], dtype=np.int64)
            )

        # Update the cache_lengths of our batch by the previous sequence length.
        # Handle both single tensor and list of tensors for compatibility
        if isinstance(prev_model_inputs.input_row_offsets, list):
            # InternVL case: use the first tensor (row offsets are identical across devices)
            row_offsets = prev_model_inputs.input_row_offsets[0]
        else:
            # Standard case: single tensor
            row_offsets = prev_model_inputs.input_row_offsets
        row_offsets = row_offsets.to(self.devices[0])

        # Build execution args, including signal buffers when using broadcast
        exec_args: list[Buffer] = [
            row_offsets,
            data_parallel_splits,
            *cache_lengths,
        ]
        if self._use_broadcast:
            if not hasattr(prev_model_inputs, "signal_buffers"):
                raise ValueError(
                    "signal_buffers required in model inputs when broadcast is "
                    "enabled (data_parallel_degree=1 with multiple devices)"
                )
            exec_args.extend(prev_model_inputs.signal_buffers)

        updated_cache_lengths = self.increment_cache_lengths_model.execute(
            *exec_args
        )

        start_idx = 0
        for devices in self.devices_per_replica:
            # max_lengths is ho st allocated and the same across each replica.
            max_lengths = kv_cache_inputs[start_idx].max_lengths

            # Advance to the next step of the max_lengths tensor.
            updated_max_lengths = max_lengths[1:, :]

            # Return our updated batch.
            assert isinstance(kv_cache_inputs, list)
            for i in range(len(devices)):
                updated_cache_length = updated_cache_lengths[start_idx + i]
                assert isinstance(updated_cache_length, Buffer)
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
        # We just hard-code a default of 512 for paged attention.
        # The worst case scenario if this is too high is that we'll evict
        # requests at an elevated rate. We print warnings in that case so users
        # are aware of what needs to be tweaked/changed.
        return 512

    def get_metrics(self, replica_idx: int) -> KVCacheMetrics:
        return self._replica_managers[replica_idx].metrics

    def get_req_blocks(
        self, request_id: RequestID, replica_idx: int
    ) -> list[int]:
        return self._replica_managers[replica_idx].block_manager.get_req_blocks(
            request_id
        )

    def get_num_pages(self, replica_idx: int) -> int:
        return self._replica_managers[replica_idx].num_pages

    def get_num_used_pages(self, replica_idx: int) -> int:
        return self._replica_managers[replica_idx].num_used_pages

    def get_num_host_pages(self, replica_idx: int) -> int:
        return self._replica_managers[replica_idx].num_host_pages

    def get_num_used_host_pages(self, replica_idx: int) -> int:
        return self._replica_managers[replica_idx].num_used_host_pages

    def get_device_tensors(self, replica_idx: int) -> list[Buffer]:
        return self._replica_managers[replica_idx].device_tensors
