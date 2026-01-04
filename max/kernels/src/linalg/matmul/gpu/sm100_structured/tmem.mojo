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
"""

from gpu import syncwarp
from gpu.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)

from ....structuring import SMemArrayType, SMemPtr


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


@register_passable("trivial")
struct TmemStage[
    num_stages: Int,
    stage_stride: Int,
    cta_group: Int,
]:
    """A pipeline stage within TMEM for accumulator buffering.

    MMA writes to one stage while epilogue reads from another.

    Parameters:
        num_stages: Pipeline stages (typically 2-4).
        stage_stride: Columns per stage.
        cta_group: Cooperating CTAs (1 or 2).
    """

    # Row offset for lower fragment (rows 16-31)
    comptime LOWER_ROW_OFFSET: UInt32 = 16 << 16

    var base_addr: UInt32
    var index: UInt32

    fn __init__(out self, base_addr: UInt32, index: UInt32):
        self.base_addr = base_addr
        self.index = index

    fn offset(self) -> UInt32:
        """TMEM address offset for MMA operations."""
        return self.base_addr + self.index * UInt32(Self.stage_stride)

    fn load_upper[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> SIMD[dtype, frag_size]:
        """Load upper accumulator fragment (rows 0-15)."""
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=frag_size,
        ](self.offset())

    fn load_lower[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> SIMD[dtype, frag_size]:
        """Load lower accumulator fragment (rows 16-31)."""
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=frag_size,
        ](self.offset() + Self.LOWER_ROW_OFFSET)

    fn wait_load(self):
        """Wait for TMEM load operations to complete."""
        tcgen05_load_wait()
