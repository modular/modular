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

"""Common interface shared by paged KV cache manager implementations.

TODO(bez): temporary scaffolding for the Jenga cutover. Once
``JengaKVCacheManager`` replaces ``PagedKVCacheManager`` outright, delete
this interface, drop the ``use_jenga_cache`` flag, and inline everything
back into a single concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from max.driver import Buffer
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
)
from max.nn.kv_cache.cache_params import KVCacheBufferInterface
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    ByteCount,
    KVTransfer,
)

from ._common import PrefixCacheHits


class PagedKVCacheManagerInterface(ABC):
    """Interface for the KVCache manager.

    This is largely a temporarily class to help phase in the JengaKVCacheManager.
    We want to delete the legacy PagedKVCacheManager and rename JengaKVCacheManager
    to PagedKVCacheManager soon.
    """

    params: KVCacheParamInterface

    @abstractmethod
    def get_prefix_cache_hit_counts(
        self, ctx: TextContext
    ) -> list[PrefixCacheHits]:
        """Counts each replica's contiguous cached prefix for a request."""

    @abstractmethod
    def alloc(
        self, ctx: TextContext, *, defer_lookup: bool = False
    ) -> KVTransfer:
        """Allocates blocks for a request.

        ``defer_lookup`` says the caller holds the request out of the batch
        while the returned transfer is incomplete and calls ``alloc`` again
        once it completes. Only then may a connector lookup run off the
        calling thread, since the request then holds no pages until that
        next call. A caller that reads the request's blocks right after this
        returns leaves it off.
        """

    def starts_lookup(self, ctx: TextContext) -> bool:
        """Returns whether a deferred ``alloc`` would start a connector lookup.

        Takes nothing, so a caller that bounds how many lookups it has out can
        hold the request back before one is started. A manager that never
        defers a lookup reads ``False``.
        """
        return False

    def start_lookup(self, ctx: TextContext) -> KVTransfer:
        """Starts the lookup a deferred ``alloc`` would start, taking no pages.

        For a request that has to wait its turn anyway, so the lookup runs
        while it waits. Asked only when ``starts_lookup`` reads ``True``, so
        a manager that never defers a lookup is never asked.
        """
        raise NotImplementedError(
            f"{type(self).__name__} never starts a deferred lookup"
        )

    def awaits_lookup(self, ctx: TextContext) -> bool:
        """Returns whether the request's last ``alloc`` left a lookup out.

        Read after a deferred ``alloc``, so a caller counts only the lookups
        that actually hold a request out, re-asks included. A manager that
        never defers a lookup reads ``False``.
        """
        return False

    def hand_back_lookup(self, ctx: TextContext) -> bool:
        """Hands back the request's lookup without loading from it.

        For a request whose answered lookup would otherwise wait in a queue
        past its lease. Its next deferred ``alloc`` starts a fresh one. A
        manager may keep the lookup instead, for one that already gave up, a
        request handed back too often, or an answer still fresh enough to
        load. A manager that never defers a lookup has nothing to hand back.

        Returns ``False`` when the manager kept an answer a later call may
        still hand back, and ``True`` when there is nothing left to hand
        back.
        """
        return True

    @abstractmethod
    def runtime_inputs(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Gets the graph inputs for per-replica batches of requests."""

    @abstractmethod
    def runtime_inputs_for_leaf(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> tuple[KVCacheInputsPerDevice[Buffer, Buffer], ...]:
        """Returns :meth:`runtime_inputs` narrowed to a single leaf cache."""

    @abstractmethod
    def alloc_dummy(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Claims a dummy request and maps it to the replica's null block."""

    @abstractmethod
    def block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns the device KV cache block occupancy for the given replica."""

    def pressure_pct(self, replica_idx: int = 0) -> float:
        """Returns how close the replica is to refusing work, in ``[0, 100]``.

        Defaults to block occupancy. A manager whose blocks are coarser than
        what a request allocates overrides this.
        """
        return self.block_count(replica_idx).used_pct

    @abstractmethod
    def host_byte_count(self) -> ByteCount:
        """Returns the host KV tier occupancy in bytes, shared by every replica."""

    @abstractmethod
    def disk_byte_count(self) -> ByteCount:
        """Returns the disk KV tier occupancy in bytes, shared by every replica."""

    @abstractmethod
    def release(self, ctx: TextContext) -> None:
        """Releases the blocks the request holds on the replica it was claimed on."""

    @abstractmethod
    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Pins a request to one replica, which owns it until it is released."""

    @abstractmethod
    def step(self, ctx: TextContext) -> None:
        """Commits the request's newly written tokens into the prefix cache."""

    @abstractmethod
    def poll_transfers(self) -> None:
        """Drains completed async KV transfers (onloads and offloads)."""

    @abstractmethod
    def pending_transfers_exist(self, replica_idx: int = 0) -> bool:
        """Returns whether any async KV transfer is in flight on the replica."""

    @abstractmethod
    def contains(self, ctx: TextContext) -> bool:
        """Returns whether the request is claimed on any replica."""

    @abstractmethod
    def take_metrics_aggregated(self) -> KVCacheMetrics:
        """Reads and clears aggregated metrics across all replicas."""

    @abstractmethod
    def reset_prefix_cache(self) -> None:
        """Resets the device prefix caches and every connector's tiers."""

    @abstractmethod
    def shutdown(self) -> None:
        """Releases the KV connector's external resources."""

    @abstractmethod
    def get_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns aggregated metrics across all replicas."""

    @abstractmethod
    def get_req_blocks_per_leaf(self, ctx: TextContext) -> dict[str, list[int]]:
        """Returns the block IDs the request holds, keyed by leaf id.

        Covers the replica the request was claimed on, with one entry per leaf
        that :meth:`KVCacheParamInterface.leaves` names. A manager whose leaves
        share one block space repeats the same row under every leaf.
        """

    @abstractmethod
    def num_req_blocks(self, ctx: TextContext) -> int:
        """Returns how many pages long the request is on its claimed replica."""

    @abstractmethod
    def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:
        """Returns the replica's KV buffer (single leaf or tree)."""

    @property
    def chunk_alignment_tokens(self) -> int:
        """Returns the token multiple a context-encoding chunk ends on.

        Zero when no group of this cache needs the alignment.
        """
        return 0

    @property
    @abstractmethod
    def effective_max_seq_length(self) -> int | None:
        """Returns the effective maximum sequence length that can be served by the block manager."""
