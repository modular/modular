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

"""Finds the CUDA allocation boundaries inside a device buffer.

With VMM enabled, MAX's memory manager backs one buffer with many separately
created granules, each its own CUDA allocation. UCX registers and exports a
single CUDA allocation per memory region: its cuda_ipc transport maps only the
allocation that contains the region's start, and its InfiniBand transport
cannot register a range that spans allocations. A buffer that crosses granules
must therefore be registered with NIXL one allocation at a time.
"""

from __future__ import annotations

from collections.abc import Callable


def allocation_splits(
    base_addr: int,
    num_bytes: int,
    address_range: Callable[[int], tuple[int, int]],
) -> list[int]:
    """Returns the offsets in a buffer where a new allocation begins.

    Walks ``[base_addr, base_addr + num_bytes)`` one allocation at a time.
    Offsets are relative to ``base_addr``, sorted, and strictly between 0 and
    ``num_bytes``; an empty list means the buffer lies in one allocation.

    Args:
        base_addr: Start of the buffer.
        num_bytes: Length of the buffer in bytes.
        address_range: Returns ``(allocation_base, allocation_size)`` for the
            allocation containing an address, such as
            ``Device._allocation_range``.
    """
    end = base_addr + num_bytes
    splits: list[int] = []
    addr = base_addr
    while addr < end:
        alloc_base, alloc_size = address_range(addr)
        alloc_end = alloc_base + alloc_size
        if not alloc_base <= addr < alloc_end:
            raise RuntimeError(
                f"Allocation [{alloc_base:#x}, {alloc_end:#x}) reported for "
                f"address {addr:#x} does not contain it"
            )
        if alloc_end < end:
            splits.append(alloc_end - base_addr)
        addr = alloc_end
    return splits
