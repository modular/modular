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

"""Unit tests for registering KV buffers one CUDA allocation at a time.

A VMM-backed buffer spans many CUDA allocations, and UCX handles one allocation
per registered region, so the transfer engine registers each piece separately
and cuts any page that crosses a piece boundary on either side.
"""

from __future__ import annotations

from collections.abc import Callable

import msgspec
import pytest
from max.pipelines.kv_cache.paged_kv_cache._cuda_alloc import (
    allocation_splits,
)
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    TensorAgentMetadata,
    _build_group_descriptors,
    _build_paired_group_descriptors,
)


def _granules(
    start: int, size: int, count: int
) -> Callable[[int], tuple[int, int]]:
    """Returns an address_range lookup for ``count`` adjacent allocations."""

    def address_range(addr: int) -> tuple[int, int]:
        index = (addr - start) // size
        assert 0 <= index < count, f"{addr:#x} outside the fake allocations"
        return start + index * size, size

    return address_range


def test_allocation_splits_single_allocation() -> None:
    assert allocation_splits(0x1100, 0x200, _granules(0x1000, 0x1000, 1)) == []


def test_allocation_splits_buffer_starting_mid_granule() -> None:
    # Buffer [0x1800, 0x3c00) over granules at 0x1000, 0x2000, 0x3000.
    splits = allocation_splits(0x1800, 0x2400, _granules(0x1000, 0x1000, 3))
    assert splits == [0x800, 0x1800]


def test_allocation_splits_ending_on_a_boundary_adds_no_split() -> None:
    assert allocation_splits(0x1000, 0x2000, _granules(0x1000, 0x1000, 2)) == [
        0x1000
    ]


def test_allocation_splits_rejects_inconsistent_range() -> None:
    with pytest.raises(RuntimeError, match="does not contain"):
        allocation_splits(0x5000, 0x100, lambda addr: (0x1000, 0x100))


def test_paired_descriptors_without_splits_match_per_side_builder() -> None:
    bytes_per_group = [800, 16]
    src = ([0x1000, 0x9000], [[1, 3, 4], [0, 2]], 0)
    dst = ([0x20000, 0x30000], [[0, 2, 5], [7, 1]], 1)

    src_descs, dst_descs = _build_paired_group_descriptors(
        bytes_per_group,
        src[0],
        [],
        src[1],
        src[2],
        dst[0],
        [[], []],
        dst[1],
        dst[2],
    )

    assert src_descs == _build_group_descriptors(
        src[0], bytes_per_group, src[1], src[2]
    )
    assert dst_descs == _build_group_descriptors(
        dst[0], bytes_per_group, dst[1], dst[2]
    )


def test_paired_descriptors_cut_pages_at_either_sides_boundary() -> None:
    bpp = 100
    # Source region boundary at offset 250 (inside page 2); destination
    # boundary at offset 130 (inside page 1).
    src_descs, dst_descs = _build_paired_group_descriptors(
        [bpp],
        [0x1000],
        [[250]],
        [[2, 0]],
        0,
        [0x8000],
        [[130]],
        [[1, 3]],
        1,
    )

    assert src_descs == [
        # Page 2 -> page 1: cut at src offset 50 and dst offset 30.
        (0x1000 + 200, 30, 0),
        (0x1000 + 230, 20, 0),
        (0x1000 + 250, 50, 0),
        # Page 0 -> page 3: no boundary inside either page.
        (0x1000 + 0, 100, 0),
    ]
    assert dst_descs == [
        (0x8000 + 100, 30, 1),
        (0x8000 + 130, 20, 1),
        (0x8000 + 150, 50, 1),
        (0x8000 + 300, 100, 1),
    ]
    for (_, src_len, _), (_, dst_len, _) in zip(
        src_descs, dst_descs, strict=True
    ):
        assert src_len == dst_len


def test_paired_descriptors_boundary_on_page_edge_does_not_cut() -> None:
    src_descs, dst_descs = _build_paired_group_descriptors(
        [100], [0], [[200]], [[1, 2]], 0, [0x1000], [[]], [[1, 2]], 0
    )
    assert src_descs == [(100, 100, 0), (200, 100, 0)]
    assert dst_descs == [(0x1000 + 100, 100, 0), (0x1000 + 200, 100, 0)]


def test_metadata_without_region_splits_decodes_from_older_peers() -> None:
    old = msgspec.json.encode(
        {
            "type": "TensorAgentMetadata",
            "agent_name": "a",
            "metadata": "",
            "base_addrs": [0x1000],
            "device_id": 0,
        }
    )
    meta = msgspec.json.decode(old, type=TensorAgentMetadata)
    assert meta.region_splits == []

    new = TensorAgentMetadata(
        agent_name="a",
        metadata=b"",
        base_addrs=[0x1000],
        device_id=0,
        region_splits=[[0x800]],
    )
    assert (
        msgspec.json.decode(msgspec.json.encode(new), type=TensorAgentMetadata)
        == new
    )
