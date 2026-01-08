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
"""Tensor Memory (TMEM) abstractions for SM100 Blackwell GPUs.

TMEM is dedicated memory for MMA accumulators, separate from registers and
shared memory. This module provides type-safe abstractions:

- TmemAllocation: Manages TMEM lifecycle (alloc/dealloc)
- TmemStage: Represents a pipeline stage for accumulator buffering
- TmemAddress: Simple address wrapper for TMEM load operations
"""

from gpu import syncwarp
from gpu.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)

from linalg.structuring import SMemArrayType, SMemPtr


@register_passable("trivial")
struct TmemAllocation[
    cta_group: Int,
    max_cols: Int = 512,
]:
    """Handle to allocated Tensor Memory.

    Lifecycle: allocate() → use → release_lock() → wait → deallocate()

    Parameters:
        cta_group: Cooperating CTAs (1 or 2).
        max_cols: TMEM columns (512 for SM100).
    """

    comptime SmemAddrStorage = SMemArrayType[UInt32, 1]

    var addr: UInt32
    var _smem_ptr: SMemPtr[UInt32]  # Kept for cross-warp sharing

    fn __init__(out self, addr: UInt32, smem_ptr: SMemPtr[UInt32]):
        self.addr = addr
        self._smem_ptr = smem_ptr

    @staticmethod
    fn allocate(smem_addr: Self.SmemAddrStorage) -> Self:
        """Allocate TMEM (MMA warp). Address stored in smem for epilogue."""
        tcgen05_alloc[Self.cta_group](smem_addr.ptr, Self.max_cols)
        syncwarp()
        return Self(smem_addr.ptr[0], smem_addr.ptr)

    @staticmethod
    fn from_shared(smem_addr: Self.SmemAddrStorage) -> Self:
        """Get handle from existing allocation (epilogue warp)."""
        return Self(smem_addr.ptr[0], smem_addr.ptr)

    fn release_lock(self):
        """Release allocation lock before waiting for epilogue."""
        tcgen05_release_allocation_lock[Self.cta_group]()

    fn deallocate(self):
        """Free TMEM after epilogue completion."""
        tcgen05_dealloc[Self.cta_group](self.addr, Self.max_cols)


# TMEM Address Encoding (SM100 Blackwell)
# =========================================
# TMEM addresses encode row and column offsets in a packed format:
#
#   Address = [row_offset : 16 bits] [column_offset : 16 bits]
#
# SM100 MMA accumulators span 32 rows × N columns per tile:
#   - Upper fragment: rows 0-15  (accessed at base address)
#   - Lower fragment: rows 16-31 (accessed at base + TMEM_LOWER_ROW_OFFSET)
#
# The value 16 << 16 encodes "row 16, column 0" as the starting offset
# for the lower half of the accumulator.
comptime TMEM_LOWER_ROW_OFFSET: UInt32 = 16 << 16


@register_passable("trivial")
struct TmemAddress:
    """Simple TMEM address wrapper for load/store operations.

    Encapsulates TMEM address encoding for accumulator fragment access.
    SM100 MMA accumulators are organized as 32 rows, split into:
      - Upper fragment (rows 0-15): accessed via upper_addr()
      - Lower fragment (rows 16-31): accessed via lower_addr()

    The lower fragment address adds TMEM_LOWER_ROW_OFFSET (16 << 16) to
    encode the row offset in the upper 16 bits of the address.

    Usage:
        var tmem = TmemAddress(base_offset)

        # Load operations
        var upper = tmem.load_upper[dtype, size]()
        var lower = tmem.load_lower[dtype, size]()
        TmemAddress.wait_load()

        # Store operations
        tmem.store_upper[dtype, size](upper_frag)
        tmem.store_lower[dtype, size](lower_frag)
        TmemAddress.wait_store()

        # Low-level address access for custom operations
        raw_upper = tmem.upper_addr()
        raw_lower = tmem.lower_addr()
    """

    var addr: UInt32

    @always_inline
    fn __init__(out self, addr: UInt32):
        self.addr = addr

    @always_inline
    fn __add__(self, offset: UInt32) -> Self:
        """Create new TmemAddress with column offset added."""
        return Self(self.addr + offset)

    @always_inline
    fn __add__(self, offset: Int) -> Self:
        """Create new TmemAddress with column offset added."""
        return Self(self.addr + UInt32(offset))

    @always_inline
    fn upper_addr(self) -> UInt32:
        """Raw address for upper fragment (rows 0-15)."""
        return self.addr

    @always_inline
    fn lower_addr(self) -> UInt32:
        """Raw address for lower fragment (rows 16-31)."""
        return self.addr + TMEM_LOWER_ROW_OFFSET

    @always_inline
    fn load_upper[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self) -> SIMD[dtype, width]:
        """Load upper accumulator fragment (rows 0-15)."""
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=width,
        ](self.upper_addr())

    @always_inline
    fn load_lower[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self) -> SIMD[dtype, width]:
        """Load lower accumulator fragment (rows 16-31)."""
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=width,
        ](self.lower_addr())

    @always_inline
    fn store_upper[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self, data: SIMD[dtype, width]):
        """Store upper accumulator fragment (rows 0-15)."""
        tcgen05_st[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            pack=False,
        ](self.upper_addr(), data)

    @always_inline
    fn store_lower[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self, data: SIMD[dtype, width]):
        """Store lower accumulator fragment (rows 16-31)."""
        tcgen05_st[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            pack=False,
        ](self.lower_addr(), data)

    @staticmethod
    @always_inline
    fn wait_store():
        """Wait for TMEM store operations to complete."""
        tcgen05_store_wait()

    @staticmethod
    @always_inline
    fn wait_load():
        """Wait for TMEM load operations to complete."""
        tcgen05_load_wait()


@register_passable("trivial")
struct TmemStage[
    num_stages: Int,
    stage_stride: Int,
    cta_group: Int,
]:
    """A pipeline stage within TMEM for accumulator buffering.

    Used by OutputTilePipeline to manage MMA→Epilogue synchronization.
    MMA writes to one stage while epilogue reads from another.

    Wraps TmemAddress with stage-specific offset calculation:
      - offset(): Column address for this stage (base + index * stride)
      - address(): TmemAddress for this stage (for load/store ops)

    Parameters:
        num_stages: Pipeline stages (typically 2-4).
        stage_stride: Columns per stage (512 / num_stages).
        cta_group: Cooperating CTAs (1 or 2).
    """

    var base_addr: UInt32
    var index: UInt32

    @always_inline
    fn __init__(out self, base_addr: UInt32, index: UInt32):
        self.base_addr = base_addr
        self.index = index

    @always_inline
    fn offset(self) -> UInt32:
        """TMEM column address for this stage."""
        return self.base_addr + self.index * UInt32(Self.stage_stride)

    @always_inline
    fn address(self) -> TmemAddress:
        """Get TmemAddress for this stage's offset."""
        return TmemAddress(self.offset())

    @always_inline
    fn load_upper[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> SIMD[dtype, frag_size]:
        """Load upper accumulator fragment (rows 0-15)."""
        return self.address().load_upper[
            dtype, frag_size, data_paths, bits, repeat
        ]()

    @always_inline
    fn load_lower[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> SIMD[dtype, frag_size]:
        """Load lower accumulator fragment (rows 16-31)."""
        return self.address().load_lower[
            dtype, frag_size, data_paths, bits, repeat
        ]()

    @staticmethod
    @always_inline
    fn wait_load():
        """Wait for TMEM load operations to complete."""
        TmemAddress.wait_load()
