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
"""The MoE combine with the routed-rows norm folded in, in one launch.

For token t, slot k names its expert-output row `restore[t * top_k + k]`;
the rows below the routed count `offsets[last]` are this device's:

    out[t] = sum_k [row_k routed] * w[t, k]
                   * bf16(rms_norm(bf16(x_k * (1 + gamma))))

in float32, k in order, with `norm.routed_rows_rms_norm`'s roundings, then
stored as `out_dtype`: float32, or bf16 rounded once from the float32 sum.
Persistent CTAs take tokens cta, cta + G, ...: a load lane bulk-copies the
token's `top_k` rows into a ring slot, routed or not (every restore value
indexes inside the buffer, so the copies need not wait on the count), and
eight work warps, a thread per two 16-byte pieces, sum each row's squares,
meet, and write the combined row. A row past the count may hold NaN, so it
is dropped, not weighed by zero. The schedule and the arithmetic are the
Moka component's
(`Kernels/test/gpu/Moka/kernels/moe/routed_rows_finalize_plan.mojo`), whose
ops call the helpers below.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import ceildiv, rsqrt
from std.memory import bitcast
from std.sys import size_of
from std.utils import StaticTuple

from layout.tma_async import SharedMemBarrier
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.memory import (
    cp_async_bulk_shared_cluster_global,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.primitives import warp
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier, named_barrier

comptime HIDDEN = 4096
"""Muse Spark's expert width, the only width the kernel takes."""
comptime VEC = 8
"""BF16 channels per 16-byte access."""
comptime PIECES = 2
"""16-byte pieces of a row per work thread."""
comptime WORK_WARPS = HIDDEN // VEC // PIECES // WARP_SIZE
comptime WORKERS = WORK_WARPS * WARP_SIZE
comptime ROW_BYTES = HIDDEN * 2

comptime POISONS = 4
"""Planted faults, `poison` 1 through 4, each of which a test must catch:
1 adds the slots past the routed count, 2 skips the input scale, 3 skips the
last token, 4 truncates a bf16 output instead of rounding it."""

comptime In[T: DType] = Pointer[Scalar[T], ImmUntrackedOrigin]
comptime Out[T: DType] = Pointer[Scalar[T], MutUntrackedOrigin]
comptime Piece = SIMD[DType.bfloat16, VEC]
comptime Wide = SIMD[DType.float32, VEC]


def stages_for(top_k: Int) -> Int:
    """Ring slots for `top_k` rows a slot: three while they fit in 200 KB."""
    return 3 if 3 * top_k * ROW_BYTES <= 200 * 1024 else 2


@always_inline
def chan(p: Int, rt: Int) -> Int:
    """The first channel of work thread `rt`'s piece `p`: a warp's pieces
    are contiguous, so its 16-byte accesses coalesce."""
    return (p * WORKERS + rt) * VEC


@always_inline
def scaled[poison: Int = 0](x: Piece, scale: Piece) -> Piece:
    """A row piece times `1 + gamma`, rounded to bf16 as the graph does."""
    comptime if poison == 2:
        return x
    return x * scale


@always_inline
def square_sum(x: Array[Piece, PIECES]) -> Float32:
    """A thread's sum of squares over its pieces."""
    var acc = Wide(0)
    comptime for p in range(PIECES):
        var f = x[p].cast[DType.float32]()
        acc += f * f
    return acc.reduce_add()


@always_inline
def sum_at(half: Int, k: Int, top_k: Int, w: Int = 0) -> Int:
    """Where warp `w` keeps slot k's sum: by trip parity, then slot, then
    warp, so a slot's sums are one vector."""
    return (half * top_k + k) * WORK_WARPS + w


@always_inline
def narrow[
    out_dtype: DType, poison: Int = 0
](acc: Wide) -> SIMD[out_dtype, VEC]:
    """The float32 sum as the output's dtype, rounded to nearest even."""
    comptime if poison == 4 and out_dtype == DType.bfloat16:
        return bitcast[out_dtype, VEC](
            (bitcast[DType.uint32, VEC](acc) >> 16).cast[DType.uint16]()
        )
    return acc.cast[out_dtype]()


@always_inline
def combine[
    poison: Int = 0
](
    acc: Wide,
    x: Piece,
    total: Float32,
    eps: Float32,
    weight: Float32,
    routed: Bool,
) -> Wide:
    """`acc` plus slot k's normalized piece by its router weight; an unrouted
    slot adds nothing."""
    var inv = rsqrt(total / Float32(HIDDEN) + eps)
    var normed = (
        (x.cast[DType.float32]() * Wide(inv))
        .cast[DType.bfloat16]()
        .cast[DType.float32]()
    )
    var keep = SIMD[DType.bool, VEC](fill=routed or poison == 1)
    return acc + keep.select(normed * Wide(weight), Wide(0))


@fieldwise_init
struct RoutedRowsFinalizeArgs[out_dtype: DType = DType.float32](
    DevicePassable, ImplicitlyCopyable
):
    """The combine's operands, one kernel parameter. Rows are dense."""

    var down: In[DType.bfloat16]
    """Expert outputs `[tokens * top_k, HIDDEN]`, in expert order."""
    var gamma: In[DType.bfloat16]
    """The norm's input scale weight `[HIDDEN]`."""
    var offsets: In[DType.uint32]
    """Per-group row offsets; `offsets[last]` is the routed row count."""
    var restore: In[DType.uint32]
    """Each `token * top_k + k` slot's row in `down`."""
    var weight: In[DType.float32]
    """Each slot's router weight `[tokens, top_k]`."""
    var out: Out[Self.out_dtype]
    """One combined row per token `[tokens, HIDDEN]`."""
    var tokens: Int64
    var last: Int64
    var eps: Float32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "RoutedRowsFinalizeArgs"


comptime _RED = 0
comptime _BARS[top_k: Int] = 2 * WORK_WARPS * top_k * 4
comptime _SLOTS0[top_k: Int] = (_BARS[top_k] + 2 * 3 * 8 + 127) // 128 * 128


def smem_bytes[top_k: Int, stages: Int]() -> Int:
    """The scratch (the warps' sums, double buffered by trip parity), the
    full and empty barriers, then the ring slots, 128-byte aligned."""
    return _SLOTS0[top_k] + stages * top_k * ROW_BYTES


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(WORKERS + WARP_SIZE)
    )
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
def _routed_rows_finalize[
    top_k: Int, stages: Int, poison: Int, out_dtype: DType
](a: RoutedRowsFinalizeArgs[out_dtype]):
    var smem = external_memory[UInt8, address_space=.SHARED, alignment=128]()
    var red = smem.unsafe_offset(_RED).unsafe_bitcast[Float32]()
    var bars = smem.unsafe_offset(_BARS[top_k]).unsafe_bitcast[
        SharedMemBarrier
    ]()
    var slots = smem.unsafe_offset(_SLOTS0[top_k])
    if thread_idx.x == 0:
        comptime for s in range(stages):
            bars[unsafe_offset=s].init(1)
            bars[unsafe_offset=stages + s].init(Int32(WORKERS))
        fence_mbarrier_init()
    # The rows, the routing and the count are the grid before this one's.
    wait_on_dependent_grids()
    barrier()
    var tokens = Int(a.tokens) - Int(poison == 3)
    var grid = grid_dim.x
    var cta = block_idx.x
    var trips = ceildiv(tokens - cta, grid)

    if warp_id() == WORK_WARPS:
        if lane_id() != 0:
            return
        for i in range(trips):
            var slot = i % stages
            var full = bars.unsafe_offset(slot)
            bars[unsafe_offset=stages + slot].wait(
                UInt32((i // stages) & 1) ^ 1
            )
            # the work warps read the slot through the generic proxy, and
            # the copies write it through the async one
            fence_async_view_proxy()
            full[].expect_bytes(Int32(top_k * ROW_BYTES))
            var t = cta + i * grid
            var dst = slots.unsafe_offset(slot * top_k * ROW_BYTES)
            comptime for k in range(top_k):
                var row = Int(a.restore[unsafe_offset=t * top_k + k])
                cp_async_bulk_shared_cluster_global(
                    dst.unsafe_offset(k * ROW_BYTES),
                    a.down.unsafe_offset(row * HIDDEN),
                    Int32(ROW_BYTES),
                    full,
                )
        return

    var rt = thread_idx.x
    var wid = warp_id()
    var scale = Array[Piece, PIECES](fill=Piece(0))
    comptime for p in range(PIECES):
        scale[p] = 1 + a.gamma.unsafe_load[width=VEC, alignment=16](chan(p, rt))
    for i in range(trips):
        var slot = i % stages
        bars[unsafe_offset=slot].wait(UInt32((i // stages) & 1))
        var t = cta + i * grid
        var xs = slots.unsafe_offset(slot * top_k * ROW_BYTES).unsafe_bitcast[
            BFloat16
        ]()
        var half = i & 1
        comptime for k in range(top_k):
            var x = Array[Piece, PIECES](fill=Piece(0))
            comptime for p in range(PIECES):
                x[p] = scaled[poison](
                    xs.unsafe_load[width=VEC, alignment=16](
                        k * HIDDEN + chan(p, rt)
                    ),
                    scale[p],
                )
            var s = warp.sum(square_sum(x))
            if lane_id() == 0:
                red[unsafe_offset=sum_at(half, k, top_k, wid)] = s
        named_barrier[Int32(WORKERS)](1)

        var routed = a.offsets[unsafe_offset=Int(a.last)]
        var acc = Array[Wide, PIECES](fill=Wide(0))
        comptime for k in range(top_k):
            var total = red.unsafe_load[width=WORK_WARPS, alignment=32](
                sum_at(half, k, top_k)
            ).reduce_add()
            var on = a.restore[unsafe_offset=t * top_k + k] < routed
            var w = a.weight[unsafe_offset=t * top_k + k]
            comptime for p in range(PIECES):
                var x = scaled[poison](
                    xs.unsafe_load[width=VEC, alignment=16](
                        k * HIDDEN + chan(p, rt)
                    ),
                    scale[p],
                )
                acc[p] = combine[poison](acc[p], x, total, a.eps, w, on)
        var op = a.out.unsafe_offset(t * HIDDEN)
        comptime for p in range(PIECES):
            op.unsafe_store[alignment=VEC * size_of[out_dtype]()](
                chan(p, rt), narrow[out_dtype, poison](acc[p])
            )
        # each reader releases its own reads: an arrive for the whole warp
        # does not order the other lanes' reads before the async refill
        _ = bars[unsafe_offset=stages + slot].arrive()


def routed_rows_finalize[
    out_dtype: DType, //, top_k: Int, poison: Int = 0
](a: RoutedRowsFinalizeArgs[out_dtype], ctx: DeviceContext) raises:
    """Writes each token's combined row, float32 or bf16.

    `a` holds dense rows, 16-byte aligned, `[tokens * top_k, HIDDEN]` bf16
    expert outputs, and an output aligned to its 8-channel stores that
    overlaps no input. Every restore value must be below `tokens * top_k`.
    `poison` plants a fault for the tests.
    """
    comptime stages = stages_for(top_k)
    comptime smem = smem_bytes[top_k, stages]()
    var tokens = Int(a.tokens) - Int(poison == 3)
    if tokens <= 0:
        return
    ctx.enqueue_function[
        _routed_rows_finalize[top_k, stages, poison, out_dtype]
    ](
        a,
        grid_dim=min(Int(a.tokens), ctx.default_device_info.sm_count),
        block_dim=WORKERS + WARP_SIZE,
        shared_mem_bytes=smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
