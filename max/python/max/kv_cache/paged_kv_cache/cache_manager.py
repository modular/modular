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

import dataclasses
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.interfaces import RequestID, TextGenerationContext
from max.nn.legacy.kv_cache import KVCacheParams, RaggedKVCacheInputs
from max.nn.legacy.kv_cache.cache_params import KVCacheParamInterface
from max.nn.legacy.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.legacy.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

from .increment_cache_lengths import IncrementCacheLengthsProcessor
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
        for replica_idx, devices in enumerate(self.devices_per_replica):
            replica_params = dp_1_params
            # Give each replica its own disk cache subdirectory to avoid
            # file collisions, double disk-usage accounting, and metadata
            # corruption when DP > 1.
            if (
                self.num_replicas > 1
                and dp_1_params.disk_offload_dir is not None
            ):
                replica_params = dataclasses.replace(
                    dp_1_params,
                    disk_offload_dir=os.path.join(
                        dp_1_params.disk_offload_dir,
                        f"replica_{replica_idx}",
                    ),
                )
            self._replica_managers.append(
                _TPPagedKVCacheManager(
                    params=replica_params,
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

        # Initialize the ragged increment cache lengths model
        self.increment_cache_lengths_processor = IncrementCacheLengthsProcessor(
            session=session,
            params=self.params,
            devices=self.devices,
        )

    def get_pct_used_blocks_after_allocation(
        self, ctx: TextGenerationContext, replica_idx: int, num_steps: int = 1
    ) -> float:
        """Gets the percentage of blocks used after allocating for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            replica_idx: Index of the replica to query.
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
                must already be assigned to a replica via ``claim``.
            replica_idx: Index of the replica to allocate on.
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
        """Gets the graph inputs for per-replica batches of requests.

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
        """Releases blocks for the request on the given replica."""
        self._replica_managers[replica_idx].release(request_id)

    def claim(self, request_id: RequestID, replica_idx: int) -> None:
        """Reserves a sequence ID for the given request ID."""
        self._replica_managers[replica_idx].claim(request_id)

    @contextmanager
    def reserve(
        self,
        contexts: Sequence[TextGenerationContext],
        *,
        replica_idx: int,
        num_steps: int = 1,
    ) -> Iterator[None]:
        """Claims, allocates, and releases contexts within a scope.

        This helper is for ephemeral flows (for example, warmup capture) where
        request IDs should be released when leaving the scope.
        """
        claimed_request_ids: list[RequestID] = []
        try:
            for context in contexts:
                if self.contains(context.request_id, replica_idx=replica_idx):
                    raise ValueError(
                        "reserve() requires unclaimed request IDs, but "
                        f"{context.request_id!r} is already claimed on "
                        f"replica {replica_idx}."
                    )
                self.claim(context.request_id, replica_idx=replica_idx)
                claimed_request_ids.append(context.request_id)
                self.alloc(
                    context,
                    replica_idx=replica_idx,
                    num_steps=num_steps,
                )
            yield
        finally:
            for request_id in claimed_request_ids:
                self.release(request_id, replica_idx=replica_idx)

    def step(self, batches: Sequence[Sequence[TextGenerationContext]]) -> None:
        """Commits new tokens into the prefix cache for per-replica batches."""
        for replica, ctxs in zip(self._replica_managers, batches, strict=True):
            replica.step(ctxs)

    def contains(self, request_id: RequestID, replica_idx: int) -> bool:
        """Returns whether the request is present on the given replica."""
        return self._replica_managers[replica_idx].contains(request_id)

    def reset_metrics(self) -> None:
        """Resets metrics for all replica managers."""
        for manager in self._replica_managers:
            manager.reset_metrics()

    @traced
    def increment_cache_lengths(
        self,
        kv_cache_inputs: Sequence[RaggedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> Sequence[RaggedKVCacheInputs]:
        """Increments cache lengths for the given inputs and returns updated inputs."""
        return self.increment_cache_lengths_processor.execute(
            kv_cache_inputs,
            prev_model_inputs,
        )

    def reset_prefix_cache(self) -> None:
        """Resets the prefix cache for all replica managers."""
        for manager in self._replica_managers:
            manager.reset_prefix_cache()

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParamInterface,
        max_seq_len: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """Infers a default optimal batch size for paged attention (``512``)."""
        # We just hard-code a default of 512 for paged attention.
        # The worst case scenario if this is too high is that we'll evict
        # requests at an elevated rate. We print warnings in that case so users
        # are aware of what needs to be tweaked/changed.
        return 512

    def get_metrics(self, replica_idx: int) -> KVCacheMetrics:
        """Returns metrics for the given replica."""
        return self._replica_managers[replica_idx].metrics

    def get_req_blocks(
        self, request_id: RequestID, replica_idx: int
    ) -> list[int]:
        """Returns block IDs for the request on the given replica."""
        return self._replica_managers[replica_idx].block_manager.get_req_blocks(
            request_id
        )

    def get_num_pages(self, replica_idx: int) -> int:
        """Returns total number of pages for the replica."""
        return self._replica_managers[replica_idx].num_pages

    def get_num_used_pages(self, replica_idx: int) -> int:
        """Returns number of used pages for the replica."""
        return self._replica_managers[replica_idx].num_used_pages

    def get_num_host_pages(self, replica_idx: int) -> int:
        """Returns number of host pages for the replica."""
        return self._replica_managers[replica_idx].num_host_pages

    def get_num_used_host_pages(self, replica_idx: int) -> int:
        """Returns number of used host pages for the replica."""
        return self._replica_managers[replica_idx].num_used_host_pages

    def get_device_tensors(self, replica_idx: int) -> list[Buffer]:
        """Returns device tensors for the replica."""
        return self._replica_managers[replica_idx].device_tensors
