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

"""The graph inputs a slab-backed KV cache hands each forward.

:class:`KVGraphInputs` turns the blocks each request's forward reads into
the cache's graph inputs: the cache lengths, each leaf's staged lookup
tables, the dispatch metadata, and the row-addressed views bound once. The
Jenga manager and the null cache both build their inputs through it, so a
forward sees the same inputs whichever cache it runs against.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheParamInterface,
)
from max.nn.kv_cache.cache_params import (
    KVCacheAssignments,
    KVCacheBufferInterface,
    max_pages_per_request,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.pipelines.context import TextContext
from max.pipelines.graph_input_stager import (
    GraphInputStager,
    GraphInputStaging,
    InputDescriptor,
)
from max.support.math import ceildiv

from ._common import (
    KV_CACHE_LENGTHS,
    cache_valid_length_for_context,
    prompt_tokens_for_context,
)
from ._mixed_step_split import lengths_and_record

__all__ = ["KVGraphInputs", "Plans"]

Plans = Mapping[str, Sequence[Sequence[int]]]
"""Per leaf, the blocks each request's forward reads, in row order."""


class KVGraphInputs:
    """Builds a slab-backed cache's graph inputs from the blocks a forward reads.

    Owns the staging buffers, declared at the widest a forward can ask so a
    graph input never moves, and the row-addressed views bound once.
    """

    def __init__(
        self,
        *,
        params: KVCacheParamInterface,
        slabs: Sequence[Buffer],
        kv_buffers: Sequence[KVCacheBufferInterface],
        max_batch_size: int,
        max_seq_len: int,
    ) -> None:
        self.params = params
        self.kv_buffers = kv_buffers
        self.max_batch_size = max_batch_size
        dp = params.data_parallel_degree
        devices = [d.to_device() for d in params.devices]
        self.devices_per_replica = split_into_groups(devices, dp)
        self.leaves = params.leaves()
        # LUT columns index logical sequence pages, not physical page IDs.
        # Tiny scale pages can give a slab millions of physical pages, but
        # a request still only addresses its sequence length plus slack.
        max_num_blocks = max_pages_per_request(params, max_seq_len)
        # A row-addressed view onto the slab is a graph input the cache
        # already owns: bound once, never staged, and never rewritten by a
        # forward. Per replica, keyed the way its leaves read it.
        self.bound: list[Mapping[str, list[Buffer]]] = [
            params.slab_to_bound_views(replica_slabs)
            for replica_slabs in split_into_groups(slabs, dp)
        ]
        # One instance across every leaf, shard and replica. Leaf ids are
        # already unique, a replica's shards are the destinations of one
        # name, and the replica scopes the name because two replicas can be
        # handed the same device.
        #
        # Declared at the widest a forward can ask, because a graph input
        # that moved would have to be rebound. Batch-major, so a step's
        # slice of one is contiguous.
        described: list[InputDescriptor] = []
        for replica_idx, replica_devices in enumerate(self.devices_per_replica):
            described.append(
                InputDescriptor(
                    name=f"{replica_idx}/{KV_CACHE_LENGTHS}",
                    dtype=DType.uint32,
                    max_shape=(max_batch_size,),
                    destinations=replica_devices,
                )
            )
            # Metadata the kernel reads on the device is a transfer like
            # any other, so it rides the forward's.
            described.extend(
                InputDescriptor(
                    name=f"{replica_idx}/{name}",
                    dtype=spec.dtype,
                    max_shape=spec.shape,
                    destinations=replica_devices,
                )
                for name, spec in params.staged_dispatch_metadata().items()
            )
            for leaf in self.leaves.values():
                described.extend(
                    InputDescriptor(
                        name=f"{replica_idx}/{key}",
                        dtype=dtype,
                        max_shape=shape,
                        destinations=replica_devices,
                    )
                    for key, (shape, dtype) in leaf.staged_input_shapes(
                        max_batch_size, max_num_blocks
                    ).items()
                )
        self._stager = GraphInputStager(described)

    def build(
        self,
        batches: Sequence[Sequence[TextContext]],
        assign: Callable[
            [GraphInputStaging, int, Sequence[TextContext]], KVCacheAssignments
        ],
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Stages every replica's assignments into one forward's inputs.

        Args:
            batches: Each replica's requests, in row order.
            assign: Builds one replica's assignments, normally by calling
                :meth:`assign` with the blocks its cache holds.
        """
        if len(batches) != self.params.data_parallel_degree:
            raise ValueError(
                f"Number of batches must match number of replicas. Expected {self.params.data_parallel_degree}, got {len(batches)}"
            )
        # A row-addressed view is a graph input, and no request means no
        # rows to bind it with.
        if any(self.bound) and not any(batches):
            raise ValueError("runtime_inputs called with an empty batch")
        # One scope per forward: every replica stages into it, and leaving
        # it sends the lot.
        with self._stager.stage() as staging:
            assignments = [
                assign(staging, replica_idx, ctxs)
                for replica_idx, ctxs in enumerate(batches)
            ]
            # Built inside the scope because the dispatch metadata is staged
            # down there: its transfer folds into this forward's rather than
            # being one of its own.
            return self.params.build_runtime_inputs(
                assignments, self.kv_buffers, staging=staging
            )

    def assign(
        self,
        staging: GraphInputStaging,
        *,
        replica_idx: int,
        batch: Sequence[TextContext],
        num_blocks: Sequence[int],
        plans: Plans,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheAssignments:
        """Stages one replica's inputs for the blocks its forward reads.

        Args:
            staging: The forward's staging scope.
            replica_idx: The replica the batch runs on.
            batch: The requests, in row order.
            num_blocks: How far into each request's row the forward reaches.
            plans: Per leaf, the blocks each request's forward reads.
            max_cache_length: The capture-time bound on the cache length
                the lookup tables are sized for, if any.
            batch_characteristics: The upper-bound shape a captured graph
                was recorded at, if the forward replays one.
        """
        required_num_blocks = max(num_blocks, default=0)
        if max_cache_length is None:
            lut_num_blocks = required_num_blocks
        else:
            if max_cache_length < 1:
                raise ValueError("max_cache_length must be positive")
            lut_num_blocks = ceildiv(max_cache_length, self.params.page_size)
            if lut_num_blocks < required_num_blocks:
                raise ValueError(
                    "capture max_cache_length cannot be smaller than the "
                    "request-required runtime cache length: "
                    f"{lut_num_blocks} < {required_num_blocks} pages."
                )

        batch_size = len(batch)
        if batch_size > self.max_batch_size:
            raise ValueError(
                "Runtime batch size exceeds preallocated KV runtime "
                f"buffer capacity: {batch_size} > {self.max_batch_size}."
            )

        replica_devices = self.devices_per_replica[replica_idx]
        cache_lengths_host, cache_lengths_by_device = staging.get(
            f"{replica_idx}/{KV_CACHE_LENGTHS}", (batch_size,)
        )
        cache_lengths_np = cache_lengths_host.to_numpy()
        cache_lengths_np.fill(0)

        staged: dict[str, tuple[Buffer, ...]] = {}
        for leaf_id, leaf in self.leaves.items():
            into: dict[str, np.ndarray] = {}
            for key, (shape, _) in leaf.staged_input_shapes(
                batch_size, lut_num_blocks
            ).items():
                # Narrowed to the shape the group writes, whatever it is.
                host, staged[key] = staging.get(f"{replica_idx}/{key}", shape)
                into[key] = host.to_numpy()
            leaf.write_staged_inputs(plans[leaf_id], into)

        # Update cache_lengths and max prompt / cache lengths.
        max_prompt_len = 0
        absolute_max_cached_len = 0
        widths: list[int] = []
        valid_lengths: list[int] = []
        for batch_idx, ctx in enumerate(batch):
            # Get the existing cache length for this sequence.
            cache_length = ctx.tokens.processed_length + len(
                ctx.spec_decoding_state.maybe_accepted_draft_tokens
            )
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far. The shared helpers
            # keep this in lockstep with the graph-capture replay path's
            # upper-bound characteristics.
            widths.append(prompt_tokens_for_context(ctx))
            valid_lengths.append(
                cache_valid_length_for_context(
                    ctx, self.params.num_draft_tokens
                )
            )
            max_prompt_len = max(max_prompt_len, widths[-1])
            absolute_max_cached_len = max(
                absolute_max_cached_len, valid_lengths[-1]
            )

        # Choose the shape used to prepare attention dispatch metadata.
        # When ``batch_characteristics`` is provided (e.g. graph-capture
        # replay), the dispatch key is resolved once from those (aligned,
        # upper-bound) values so it matches a captured graph; otherwise
        # the real per-replica values are used. LUT / cache_lengths always
        # use the real values; only the dispatch metadata and
        # ``max_prompt_length`` / ``max_cache_length`` follow
        # ``dispatch_*``.
        if batch_characteristics is not None:
            bc = batch_characteristics
            if (
                batch_size > bc.batch_size
                or max_prompt_len > bc.max_prompt_length
                or absolute_max_cached_len > bc.max_cache_valid_length
            ):
                raise ValueError(
                    f"Real batch size ({batch_size}) exceeds the requested dispatch batch size ({bc.batch_size})."
                )
            batch_size = bc.batch_size
            max_prompt_len = bc.max_prompt_length
            absolute_max_cached_len = bc.max_cache_valid_length

        max_prompt_length_host, max_cache_length_host, mixed_step_split = (
            lengths_and_record(
                widths,
                valid_lengths,
                self.params.num_draft_tokens,
                max_prompt_len,
                absolute_max_cached_len,
            )
        )
        staged_by_device: list[dict[str, Buffer]] = [
            {key: value[device_idx] for key, value in staged.items()}
            for device_idx in range(len(replica_devices))
        ]
        # Inputs bound once ride along unnarrowed and uncopied.
        for key, views in self.bound[replica_idx].items():
            for device_idx, view in enumerate(views):
                staged_by_device[device_idx][key] = view

        return KVCacheAssignments(
            cache_lengths_by_device=list(cache_lengths_by_device),
            staged_by_device=staged_by_device,
            max_prompt_length=max_prompt_length_host,
            max_cache_length=max_cache_length_host,
            batch_characteristics=BatchCharacteristics(
                batch_size=batch_size,
                max_prompt_length=max_prompt_len,
                max_cache_valid_length=absolute_max_cached_len,
            ),
            mixed_step_split=mixed_step_split,
        )
