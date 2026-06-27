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

"""Connector protocol for external KV cache tiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from max.nn.kv_cache.cache_params import KVHashAlgo
from max.nn.kv_cache.metrics import KVCacheMetrics


def to_block_hash_bytes(h: int | bytes) -> bytes:
    """Coerces a block hash to the canonical bytes form for connector calls.

    Block hashes flow through the prefix-caching layer as ``int | bytes``
    because each algo defines its own natural Python type at hash production
    (ahash64-family produces ``int``, SHA-256 produces ``bytes``). Below the
    ``KVConnector`` boundary every connector sees a single type. This shim
    is the only int->bytes coercion site and lives next to the Protocol it
    serves.

    Args:
        h: A block hash. ``int`` is encoded as 8 big-endian signed bytes,
            which covers the negative range produced by the ``sha256_64``
            truncation path. ``bytes`` must already be in canonical 8- or
            32-byte form and is returned unchanged.

    Returns:
        The 8-byte (ahash64-family) or 32-byte (SHA-256) canonical encoding.

    Raises:
        ValueError: If ``h`` is ``bytes`` with a length other than 8 or 32.
    """
    if isinstance(h, bytes):
        if len(h) not in (8, 32):
            raise ValueError(
                f"block hash bytes must be length 8 or 32, got {len(h)}"
            )
        return h
    return h.to_bytes(8, "big", signed=True)


@runtime_checkable
class KVConnector(Protocol):
    """Protocol for KV cache connectors managing external (non-device) tiers.

    The manager owns device tensors, block allocation, and device-side prefix
    cache. Connectors handle external tier operations (e.g., host memory)
    via load/offload methods.

    All block hashes crossing this Protocol are in canonical bytes form:
    8 big-endian bytes for ahash64-family algos (including ``sha256_64``),
    32 bytes for full SHA-256 digests. ``parent_seq_hash`` is ``None`` to
    denote the root of the chain; otherwise it is in the same bytes form
    as each element of ``block_hashes``. The ``BlockManager`` is
    responsible for any int->bytes coercion (see ``to_block_hash_bytes``);
    connectors never see Python ``int`` hashes.

    Required call ordering per inference step:
      1. connector.load()            # post loads on the main stream
      2. connector.wait_for_loads()  # block until loads have landed
      3. connector.offload()         # kick off this step's offloads
      4. [model executes]
      5. connector.wait_for_offloads()  # drain offloads posted this step
    """

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        ...

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        """Load data from external cache into device blocks.

        Args:
            device_block_ids: Device block IDs to load data into.
            block_hashes: Hashes to load data for, in canonical bytes form
                (8 big-endian bytes for ahash64-family, 32 bytes for
                SHA-256).
            replica_idx: DP replica whose device buffers receive the loaded
                blocks. The external tier itself is replica-agnostic (keyed by
                hash); this only selects the H2D destination.

        Returns:
            Number of blocks loaded from external cache.
        """
        ...

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> None:
        """Offload the device blocks to the external cache.

        The blocks form one ordered sequence whose first block chains onto
        ``parent_seq_hash`` (``None`` denotes the root of the chain).
        Connectors that key blocks purely by hash (host/disk tiers) ignore
        ``parent_seq_hash``; the dKV connector uses it to chain the
        sequence server-side.

        Args:
            block_ids: Device block IDs to offload, in prefix order.
            block_hashes: Hashes for the blocks being offloaded, in prefix
                order. Canonical bytes form (8 big-endian bytes for
                ahash64-family, 32 bytes for SHA-256).
            parent_seq_hash: Hash of the block preceding ``block_hashes[0]``
                in the prefix in the same bytes form as ``block_hashes``,
                or ``None`` if this run begins at the root.
            replica_idx: DP replica whose device buffers source the offloaded
                blocks. The external tier itself is replica-agnostic.
        """
        ...

    def wait_for_loads(self) -> None:
        """Block until all posted loads have landed in device memory.

        Called before the forward pass. Connectors whose loads are ordered on
        the device stream (host/disk tiers) need no work here; the dKV
        connector blocks on its off-stream NIXL READs. No-op by default.
        """
        return None

    def wait_for_offloads(self) -> None:
        """Drain offloads posted since the last call.

        Called after the forward pass. No-op by default.
        """
        return None

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        ...

    # Optional properties with default implementations
    @property
    def num_host_blocks(self) -> int:
        """Number of host blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_used_host_blocks(self) -> int:
        """Number of used host blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_disk_blocks(self) -> int:
        """Number of disk blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        """Number of used disk blocks. Returns 0 if not applicable."""
        return 0

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache. No-op by default."""
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector. Returns empty metrics by default."""
        return KVCacheMetrics()

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        """Set of hash algos this connector accepts in ``load``/``offload``.

        The default ``frozenset({"ahash64"})`` keeps legacy connectors
        written before SHA-256 support landed working under the original
        hashing algo. Connectors that accept 32-byte SHA-256 hashes must
        override this to advertise ``frozenset({"ahash64", "sha256"})``
        (or an SHA-256-only set).
        """
        return frozenset({"ahash64"})
