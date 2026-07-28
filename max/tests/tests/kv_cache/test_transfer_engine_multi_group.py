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

"""CPU unit tests for KVTransferEngine's per-group NIXL registration.

Covers two things, using CPU buffers only (no NIXL/GPU objects required):

- ``_unit_shard_groups``: how typed ``to_memory()`` units are grouped into
  shape-homogeneous NIXL groups (one per child x kind).
- ``KVTransferEngine.__init__`` structural-validation guards, which raise
  before any NIXL registration.
"""

from __future__ import annotations

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import (
    KVCacheBuffer,
    KVCacheMemory,
    MultiKVCacheBuffer,
    ReplicatedKVCacheMemory,
)


def _cpu_buf(
    num_pages: int,
    elts_per_page: int,
    dtype: DType = DType.bfloat16,
) -> Buffer:
    """Allocate a 2-D CPU buffer in the original (non-uint8) dtype."""
    return Buffer(shape=(num_pages, elts_per_page), dtype=dtype, device=CPU())


def _kv(
    elts_per_page: int,
    *,
    tp: int = 2,
    num_pages: int = 8,
    replicated: bool = False,
    dtype: DType = DType.bfloat16,
    scale_elts: int | None = None,
) -> KVCacheBuffer:
    """A single-cache ``KVCacheBuffer`` with ``tp`` TP shards.

    Pass ``scale_elts`` to attach a float32 scales tensor (quantized cache).
    """
    return KVCacheBuffer(
        values=[_cpu_buf(num_pages, elts_per_page, dtype) for _ in range(tp)],
        scales=(
            [_cpu_buf(num_pages, scale_elts, DType.float32) for _ in range(tp)]
            if scale_elts is not None
            else None
        ),
        replicates_kv_across_tp=replicated,
    )


def _hetero_multi(*, num_pages: int = 8, tp: int = 2) -> MultiKVCacheBuffer:
    """Multi-child cache whose children differ in bytes_per_page.

    A 61-"layer" target next to a 1-"layer" draft: the two children have
    different per-page byte sizes and must land in separate NIXL groups. This
    is the case that motivated per-child grouping.
    """
    return MultiKVCacheBuffer(
        children={
            "target": _kv(61 * 8, tp=tp, num_pages=num_pages),
            "draft": _kv(8, tp=tp, num_pages=num_pages),
        }
    )


# ---------------------------------------------------------------------------
# _unit_shard_groups: units -> shape-homogeneous NIXL groups
# ---------------------------------------------------------------------------


def test_unit_shard_groups_sharded() -> None:
    """Non-replicated single cache: one group holding all TP shards."""
    from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
        _unit_shard_groups,
    )

    groups = _unit_shard_groups(_kv(64).to_memory())
    assert len(groups) == 1
    assert len(groups[0]) == 2  # 2 TP shards


def test_unit_shard_groups_replicated() -> None:
    """Replicated cache: one group whose shard list is [buffer, *peers]."""
    from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
        _unit_shard_groups,
    )

    groups = _unit_shard_groups(_kv(64, tp=3, replicated=True).to_memory())
    assert len(groups) == 1
    assert len(groups[0]) == 3  # buffer + 2 peers


def test_unit_shard_groups_quantized() -> None:
    """Quantized cache: values and scales become separate NIXL groups."""
    from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
        _unit_shard_groups,
    )

    groups = _unit_shard_groups(
        _kv(64, dtype=DType.uint8, scale_elts=4).to_memory()
    )
    assert len(groups) == 2  # values group + scales group
    assert len(groups[0]) == 2 and len(groups[1]) == 2


def test_unit_shard_groups_multi_child_heterogeneous() -> None:
    """Multi-child cache with different shapes: one group per child."""
    from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
        _unit_shard_groups,
    )

    groups = _unit_shard_groups(_hetero_multi().to_memory())
    assert len(groups) == 2  # one group per child
    assert len(groups[0]) == 2 and len(groups[1]) == 2
    # Different per-page byte sizes -> kept in separate groups.
    assert groups[0][0].shape[1] != groups[1][0].shape[1]


# ---------------------------------------------------------------------------
# KVTransferEngine.__init__ structural guards (raise before NIXL)
# ---------------------------------------------------------------------------


def _mem(elts_per_page: int = 64, *, num_pages: int = 4) -> KVCacheMemory:
    """A sharded (non-replicated) transport memory unit, uint8 view."""
    return KVCacheMemory(buffer=_cpu_buf(num_pages, elts_per_page, DType.uint8))


def test_mixed_replication_within_engine_detected() -> None:
    """Units that disagree on replication kind are rejected."""
    from max.pipelines.kv_cache import KVTransferEngine

    replicated = ReplicatedKVCacheMemory(
        buffer=_cpu_buf(4, 64, DType.uint8),
        peers=[_cpu_buf(4, 64, DType.uint8)],
    )

    with pytest.raises(ValueError, match="same replication kind"):
        KVTransferEngine("engine", [[replicated, _mem()]], total_num_pages=4)


def test_replicas_with_different_group_counts_raises() -> None:
    """Replicas that produce different NIXL group counts are rejected."""
    from max.pipelines.kv_cache import KVTransferEngine

    # Replica 0: two differently-shaped units -> 2 groups; replica 1: 1 group.
    replica0 = [_mem(64), _mem(16)]
    replica1 = [_mem(64)]

    with pytest.raises(ValueError, match="consistent buffer structure"):
        KVTransferEngine("engine", [replica0, replica1], total_num_pages=4)


# ---------------------------------------------------------------------------
# Per-group descriptor arithmetic (the SERVOPT-1456 stride-mismatch guard)
# ---------------------------------------------------------------------------


def test_build_group_descriptors_uses_own_stride_per_group() -> None:
    """Each group is addressed with its OWN base + bytes_per_page.

    Guards the descriptor math behind SERVOPT-1456: page ``i`` of group ``g``
    lands at ``base[g] + i*bpp[g]`` with size ``bpp[g]``. A group with a
    different ``bytes_per_page`` (e.g. a 1-layer draft next to a 61-layer
    target) can never inherit another group's stride.
    """
    from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
        _build_group_descriptors,
    )

    base_addrs = [0x1000, 0x5000]  # target group, draft group
    bytes_per_group = [800, 16]  # deliberately different per-page sizes
    page_idxs = [0, 2, 5]
    device_id = 3

    descs = _build_group_descriptors(
        base_addrs, bytes_per_group, page_idxs, device_id
    )

    # Group-major, page-inner ordering; one descriptor per (group, page).
    assert len(descs) == len(bytes_per_group) * len(page_idxs)

    # Group 0 (target): base 0x1000, stride 800.
    for i, k in enumerate(page_idxs):
        addr, size, dev = descs[i]
        assert addr == 0x1000 + k * 800
        assert size == 800
        assert dev == device_id

    # Group 1 (draft): base 0x5000, stride 16 -- its own bpp, not the target's.
    off = len(page_idxs)
    for i, k in enumerate(page_idxs):
        addr, size, _ = descs[off + i]
        assert addr == 0x5000 + k * 16
        assert size == 16


# ---------------------------------------------------------------------------
# _validate_tensor_shape: cross-shard shape-mismatch rejection
# ---------------------------------------------------------------------------


def test_validate_tensor_shape_rejects_mismatched_shards() -> None:
    """Shards of one group must share a shape (subsumes elt-count + dtype).

    ``to_memory()`` emits 2-D uint8 views (``[total_num_pages,
    bytes_per_page]``), so shape-equality is the single invariant a NIXL group
    needs: a differing page count or per-page stride across shards is rejected
    outright rather than silently producing a mismatched transfer descriptor.
    """
    from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
        _validate_tensor_shape,
    )

    good = _cpu_buf(17, 24, DType.uint8)
    # Same per-page stride, different page count.
    fewer_pages = _cpu_buf(9, 24, DType.uint8)
    with pytest.raises(ValueError, match="same shape"):
        _validate_tensor_shape([good, fewer_pages])

    # Same page count, different per-page stride.
    wider = _cpu_buf(17, 48, DType.uint8)
    with pytest.raises(ValueError, match="same shape"):
        _validate_tensor_shape([good, wider])


# ---------------------------------------------------------------------------
# from_paged_kv_cache memory build (guards the re-flatten regression)
# ---------------------------------------------------------------------------


def test_per_replica_memory_keeps_children_separate() -> None:
    """from_paged maps each replica buffer through ``to_memory()``, keeping a
    MultiKVCacheBuffer's children as separate typed groups.

    Guards against regressing to a flattened ``all_buffers`` layout (the
    multi-cache heterogeneous-shape crash). CPU-only: exercises the build step
    without constructing an engine (which needs NIXL).
    """
    from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
        _per_replica_memory,
        _unit_shard_groups,
    )

    # dp = 2 replicas.
    memory = _per_replica_memory(
        [_hetero_multi(num_pages=4), _hetero_multi(num_pages=4)]
    )

    assert len(memory) == 2
    # Typed to_memory() units, not raw Buffers from a flattened all_buffers.
    assert all(isinstance(u, KVCacheMemory) for u in memory[0])
    # Children stay in separate groups with different bytes_per_page.
    groups = _unit_shard_groups(memory[0])
    assert len(groups) == 2
    assert groups[0][0].shape[1] != groups[1][0].shape[1]
