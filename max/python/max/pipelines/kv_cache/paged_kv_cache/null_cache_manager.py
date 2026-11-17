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

"""A KV cache that holds only the null page.

A pipeline runs its warmups against :class:`NullKVCache` before it knows how
much memory its real cache may take: every page of every request is the
null page, so any batch at any length fits in one huge page per device. The
graph inputs come from :class:`~.kv_graph_inputs.KVGraphInputs`, the code
the real cache builds its own with, so the forwards a warmup runs are the
forwards serving runs, reading garbage.
"""

from __future__ import annotations

from collections.abc import Sequence

from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
)
from max.nn.kv_cache.cache_params import (
    KVCacheAssignments,
    KVCacheBufferInterface,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.graph_input_stager import GraphInputStaging
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    ByteCount,
    CompletedTransfer,
    KVTransfer,
)
from max.pipelines.modeling.types import RequestID
from max.support.math import ceildiv

from ._common import PrefixCacheHits, _compute_seq_len
from .cache_manager_interface import PagedKVCacheManagerInterface
from .jenga_cache_manager import JengaKVCacheManager
from .kv_graph_inputs import KVGraphInputs

_NULL_BID = 0
"""Every leaf's null page: huge block 0's first little block."""

_GEOMETRY_BUDGET = 1 << 40
"""Any budget large enough to plan a geometry; only its huge page is used."""


class NullKVCache(PagedKVCacheManagerInterface):
    """A KV cache of one huge page per device, the null page every row reads.

    Holds no blocks, no prefix cache, and no connector. Every request it
    admits maps each of its pages to the null page; a recurrent or scratch
    leaf's one block is the null page too. A forward against it reads and
    writes only that page, so its results are garbage, but it allocates the
    workspaces a forward of that shape allocates against the real cache.
    """

    def __init__(
        self,
        *,
        params: KVCacheParamInterface,
        max_batch_size: int,
        max_seq_len: int,
        max_num_input_tokens: int | None = None,
    ) -> None:
        self.params = params
        self._max_seq_len = max_seq_len
        self._max_num_input_tokens = max_num_input_tokens
        geometry = JengaKVCacheManager._plan_geometry(params, _GEOMETRY_BUDGET)
        devices = [d.to_device() for d in params.devices]
        self._slabs = [
            Buffer.zeros(
                shape=(1, geometry.huge_page_bytes),
                dtype=DType.uint8,
                device=d,
            )
            for d in devices
        ]
        self.slab_bytes = geometry.huge_page_bytes
        """Bytes of the slab on each device."""
        kv_buffers = [
            params.slab_to_buffer_views(bs, geometry.padded_sizes)
            for bs in split_into_groups(
                self._slabs, params.data_parallel_degree
            )
        ]
        self._inputs = KVGraphInputs(
            params=params,
            slabs=self._slabs,
            kv_buffers=kv_buffers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        self._single_block = {
            leaf_id: leaf.group_id.is_recurrent() or leaf.group_id.is_scratch()
            for leaf_id, leaf in params.leaves().items()
        }
        self._requests: dict[RequestID, int] = {}

    def _num_blocks(self, ctx: TextContext) -> int:
        seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens=self.params.num_draft_tokens,
            num_draft_tokens_per_step=self.params.num_draft_tokens_per_step,
            max_num_input_tokens=self._max_num_input_tokens,
        )
        return ceildiv(seq_len, self.params.page_size)

    def runtime_inputs(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Gets the graph inputs, every page of every row the null page."""

        def assign(
            staging: GraphInputStaging,
            replica_idx: int,
            batch: Sequence[TextContext],
        ) -> KVCacheAssignments:
            num_blocks = [self._num_blocks(ctx) for ctx in batch]
            plans = {
                leaf_id: [
                    [_NULL_BID] * (1 if single else n) for n in num_blocks
                ]
                for leaf_id, single in self._single_block.items()
            }
            return self._inputs.assign(
                staging,
                replica_idx=replica_idx,
                batch=batch,
                num_blocks=num_blocks,
                plans=plans,
                max_cache_length=max_cache_length,
                batch_characteristics=batch_characteristics,
            )

        return self._inputs.build(batches, assign)

    def runtime_inputs_for_leaf(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> tuple[KVCacheInputsPerDevice[Buffer, Buffer], ...]:
        """Returns :meth:`runtime_inputs` narrowed to a single leaf cache."""
        inputs = self.runtime_inputs(
            batches,
            max_cache_length=max_cache_length,
            batch_characteristics=batch_characteristics,
        )
        return tuple(tree.leaves(inputs, leaf=KVCacheInputsPerDevice))

    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Admits the request on ``replica_idx``."""
        self._requests[ctx.request_id] = replica_idx

    def alloc(
        self, ctx: TextContext, *, defer_lookup: bool = False
    ) -> KVTransfer:
        """Allocates nothing: every page is the null page."""
        return CompletedTransfer()

    def alloc_dummy(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Admits the request; its pages are the null page."""
        self.claim(ctx, replica_idx)

    def step(self, ctx: TextContext) -> None:
        """Records nothing: no page holds anything to keep."""

    def release(self, ctx: TextContext) -> None:
        """Forgets the request."""
        self._requests.pop(ctx.request_id, None)

    def contains(self, ctx: TextContext) -> bool:
        """Whether the request was admitted and not released."""
        return ctx.request_id in self._requests

    def get_prefix_cache_hit_counts(
        self, ctx: TextContext
    ) -> list[PrefixCacheHits]:
        """Returns no hits: nothing is cached."""
        return []

    def get_req_blocks_per_leaf(self, ctx: TextContext) -> dict[str, list[int]]:
        """Returns the null page for every page of every leaf."""
        n = self._num_blocks(ctx)
        return {
            leaf_id: [_NULL_BID] * (1 if single else n)
            for leaf_id, single in self._single_block.items()
        }

    def num_req_blocks(self, ctx: TextContext) -> int:
        """Returns how many pages the request's next forward reaches."""
        return self._num_blocks(ctx)

    def block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns one block, the null page, never used."""
        return BlockCount(free=1, total=1)

    def host_byte_count(self) -> ByteCount:
        """Returns no host tier."""
        return ByteCount(free=0, total=0)

    def disk_byte_count(self) -> ByteCount:
        """Returns no disk tier."""
        return ByteCount(free=0, total=0)

    def poll_transfers(self) -> None:
        """No-op: nothing is ever in flight."""

    def pending_transfers_exist(self, replica_idx: int = 0) -> bool:
        """Returns False: nothing is ever in flight."""
        return False

    def get_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns empty metrics."""
        return KVCacheMetrics()

    def take_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns empty metrics."""
        return KVCacheMetrics()

    def reset_prefix_cache(self) -> None:
        """No-op: nothing is cached."""

    def shutdown(self) -> None:
        """No-op: there is no connector."""

    def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:
        """Returns the replica's views over the null slab."""
        return self._inputs.kv_buffers[replica_idx]

    @property
    def effective_max_seq_length(self) -> int | None:
        """Returns the max length: any length fits in the null page."""
        return self._max_seq_len
