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
"""Pieces the SM100 3-D tensor-core conv kernels share.

Target family: Blackwell sm_100a, 1-SM UMMA, NDHWC activations and outputs.
`conv3d_col_umma` and `conv3d_tap_umma` each have their own schedule, but they
use the same mbarrier ring cursor, drain an accumulator through the same
staging layout, and read the same `[F, K]` filter that `pack_conv3d_filter`
builds.
"""

from std.math import ceildiv
from std.sys import size_of

from layout import (
    Idx,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.compute.arch.tcgen05 import tcgen05_ld, tcgen05_load_wait
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)


struct RingCursor[depth: Int](TrivialRegisterPassable):
    """A position in an mbarrier ring: the slot and the parity of its lap.

    A producer waits an empty barrier on `phase ^ 1`, so its first lap passes
    through; a consumer waits a full barrier on `phase`.

    Parameters:
        depth: Slots in the ring.
    """

    var stage: Int32
    """The slot. A 32-bit index keeps the cursor on the warp-uniform integer
    datapath; a 64-bit one costs a carry chain per advance."""
    var phase: UInt32

    @always_inline
    def __init__(out self):
        self.stage = 0
        self.phase = 0

    @always_inline
    def slot(self) -> Int:
        """The slot index."""
        return Int(self.stage)

    @always_inline
    def slot_after(self, ahead: Int32) -> Int:
        """The slot `ahead` positions on (less than a lap), without a modulo."""
        var s = self.stage + ahead
        if s >= Int32(Self.depth):
            s -= Int32(Self.depth)
        return Int(s)

    @always_inline
    def advance(mut self):
        """Moves to the next slot, flipping the phase at the wrap."""
        self.stage += 1
        if self.stage == Int32(Self.depth):
            self.stage = 0
            self.phase ^= 1


@always_inline
def stage_columns_swizzled[
    dtype: DType, pieces: Int
](
    taddr: UInt32,
    slots: MutPointer[Scalar[dtype], _, address_space=.SHARED],
    slot_elems: Int,
    row: Int,
    live: Bool,
):
    """Moves `32 * pieces` accumulator columns of this thread's lane from
    tensor memory to swizzled staging boxes of 32 columns each.

    Each box is one TMA store box: 32 output channels per row, in the layout
    the store's swizzle expects (SW64 at 16 bits, SW128 at 32), so a lane's
    16-byte chunks do not collide on shared-memory banks. Every thread of the
    four epilogue warps calls this together, because `tcgen05.ld` is
    warp-collective; a thread whose lane belongs to no output (`live` false)
    still takes part in the load and writes nothing.

    Parameters:
        dtype: Element type of the output.
        pieces: Boxes to fill, at `slot_elems` apart.

    Args:
        taddr: Tensor-memory address of the first column, lane field zero.
        slots: First box.
        slot_elems: Elements between consecutive boxes.
        row: The staging row this thread's lane writes.
        live: Whether the row holds an output.
    """
    comptime EB = size_of[dtype]()
    comptime assert EB == 2 or EB == 4, "16- or 32-bit output"
    comptime CH = 16 // EB
    """Output elements per 16-byte chunk."""
    comptime NCH = 32 // CH
    """Chunks per 32-column row."""
    comptime for pc in range(pieces):
        var regs = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=32,
            dtype=.float32,
            pack=False,
        ](taddr + UInt32(pc * 32))
        tcgen05_load_wait()
        if live:
            var dst = slots.unsafe_offset(pc * slot_elems)
            comptime for c in range(NCH):
                var v = SIMD[dtype, CH]()
                comptime for e in range(CH):
                    v[e] = regs[c * CH + e].cast[dtype]()
                # SW64 (16-bit): 64-byte rows, chunk ^ (row / 2 mod 4);
                # SW128 (32-bit): 128-byte rows, chunk ^ (row mod 8).
                var key = row & 7
                comptime if EB == 2:
                    key = (row >> 1) & 3
                dst.unsafe_offset(row * 32 + ((c ^ key) * CH)).unsafe_store[
                    alignment=16
                ](0, v)


comptime _TILE = 32
"""Edge of the transposing tile of `pack_filter_qrscf_kernel`."""
comptime _TILE_ROWS = 8
"""Rows of the tile one block iteration moves: block is `_TILE x _TILE_ROWS`."""


@__name(t"pack_conv3d_filter_fcqrs_{dtype}")
def pack_filter_fcqrs_kernel[
    dtype: DType,
    taps: Int,
    channels: Int,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst: MutPointer[Scalar[dtype], MutUntrackedOrigin],
):
    """Permutes a filter FCQRS `[F, C, Q*R*S]` to `[F, Q*R*S, C]`.

    One block per output channel `f`. Its slab of `C * T` elements is
    contiguous in both layouts, so the block reads it with 16-byte vectors into
    shared memory and writes it back transposed, again in 16-byte vectors; the
    transpose happens in shared memory.

    Parameters:
        dtype: Element type of the filter.
        taps: `Q * R * S`.
        channels: Input channels.
        SrcLayoutType: Layout of `src`.
        src_origin: Origin of `src`.
        SrcEngine: Storage engine of `src`.

    Args:
        src: Filter, FCQRS.
        dst: Packed filter, `[F, K]` with `K = Q * R * S * C` and `C`
            innermost.
    """
    comptime W = 16 // size_of[dtype]()
    comptime t_n = taps
    comptime c_n = channels
    comptime slab = t_n * c_n
    var tid = thread_idx.x
    var threads = block_dim.x
    var f = block_idx.x
    var tile = external_memory[
        Scalar[dtype], address_space=.SHARED, alignment=16
    ]()
    # Trigger first: the dependent kernel's prologue overlaps this one's work.
    wait_on_dependent_grids()
    launch_dependent_grids()
    var from_f = src.unsafe_ptr().unsafe_offset(f * slab)
    for i in range(tid * W, slab, threads * W):
        tile.unsafe_offset(i).unsafe_store[alignment=16](
            0, from_f.unsafe_load[width=W, alignment=16](i)
        )
    barrier()
    var packed = TileTensor(dst, row_major((grid_dim.x, Idx[slab])))
    for j0 in range(tid * W, slab, threads * W):
        var v = SIMD[dtype, W]()
        comptime for e in range(W):
            var t, c = divmod(j0 + e, c_n)
            v[e] = tile.unsafe_load(c * t_n + t)
        packed.store[width=W, alignment=16]((f, j0), v)


@__name(t"pack_conv3d_filter_qrscf_{dtype}")
def pack_filter_qrscf_kernel[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    reduction: Int32,
    outputs: Int32,
):
    """Permutes a filter QRSCF `[K, F]` to `[F, K]`: a tiled transpose, both
    sides coalesced.

    Args:
        src: Filter, QRSCF viewed as `[K, F]` with `K = Q * R * S * C`.
        dst: Packed filter, `[F, K]`.
        reduction: `K`.
        outputs: Output channels.
    """
    var k_n = Int(reduction)
    var f_n = Int(outputs)
    var packed = TileTensor(dst, row_major((f_n, k_n)))
    # One extra column spreads the transposing reads over the banks.
    var tile = tt_stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=16
    ](row_major[_TILE, _TILE + 1]())
    var tx = thread_idx.x
    var ty = thread_idx.y
    var f0 = block_idx.x * _TILE
    var k0 = block_idx.y * _TILE
    # Trigger first: the dependent kernel's prologue overlaps this one's work.
    wait_on_dependent_grids()
    launch_dependent_grids()
    comptime for j in range(0, _TILE, _TILE_ROWS):
        if f0 + tx < f_n and k0 + ty + j < k_n:
            tile[ty + j, tx] = src.unsafe_ptr().unsafe_load(
                (k0 + ty + j) * f_n + f0 + tx
            )
    barrier()
    comptime for j in range(0, _TILE, _TILE_ROWS):
        if k0 + tx < k_n and f0 + ty + j < f_n:
            packed[f0 + ty + j, k0 + tx] = tile[tx, ty + j]


def pack_conv3d_filter[
    filter_is_fcrs: Bool, taps: Int, channels: Int
](
    filter: TileTensor[address_space=.GENERIC, ...],
    outputs: Int,
    ctx: DeviceContext,
) raises -> DeviceBuffer[filter.dtype]:
    """Enqueues the permutation of a dense 3-D filter to `[F, K]`, `K =
    ((q * R + r) * S + s) * C + c`, and returns the packed buffer.

    The tensor-core kernels read the filter K-major, 8 contiguous reduction
    elements per lane, so the channel run must be the innermost one. The
    caller keeps the buffer alive until the kernel that reads it is enqueued.

    The pack is on the critical path of every call, so these are not
    `_fcrs_to_krsc_row` and `_transpose_qrscf_to_nk`, which they replaced in a
    trial: on the 8 MB mid_res filter that pair takes 10.3 us (FCQRS) and
    18.5 us (QRSCF, uncoalesced reads) against 6.2 us and 10.3 us here, and
    the QRSCF difference alone is most of the route's 10 us margin at that
    shape. The FCQRS kernel reads and writes 16-byte vectors and takes the
    extents at compile time; the QRSCF one tiles the transpose through shared
    memory.

    Parameters:
        filter_is_fcrs: The filter is FCQRS rather than QRSCF.
        taps: `Q * R * S`.
        channels: Input channels, a multiple of 16 bytes.

    Args:
        filter: Filter; dense, row-major and 16-byte aligned.
        outputs: Output channels.
        ctx: Device context for the launch.

    Returns:
        The packed filter, `outputs * taps * channels` elements.
    """
    comptime eb = size_of[filter.dtype]()
    var total = taps * channels * outputs
    var packed = ctx.enqueue_create_buffer[filter.dtype](total)
    comptime if filter_is_fcrs:
        comptime kernel = pack_filter_fcqrs_kernel[
            filter.dtype,
            taps,
            channels,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
        ]
        comptime slab_bytes = taps * channels * eb
        ctx.enqueue_function[kernel](
            filter.as_imm(),
            packed.unsafe_ptr(),
            grid_dim=outputs,
            block_dim=256,
            shared_mem_bytes=slab_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(slab_bytes)
            ),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    else:
        comptime kernel = pack_filter_qrscf_kernel[
            filter.dtype,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
        ]
        comptime reduction = taps * channels
        ctx.enqueue_function[kernel](
            filter.as_imm(),
            packed.unsafe_ptr(),
            Int32(reduction),
            Int32(outputs),
            grid_dim=(ceildiv(outputs, _TILE), ceildiv(reduction, _TILE)),
            block_dim=(_TILE, _TILE_ROWS),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    return packed^
