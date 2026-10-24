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

from std.math import align_down, align_up, ceildiv, clamp, rsqrt
from std.math.uutils import umod, ufloordiv, uceildiv
from std.sys.info import align_of, simd_width_of, size_of

import max.gpu.primitives.warp as warp
from std.algorithm import vectorize
from max.algorithm.functional import (
    sync_parallelize,
)
from std.bit import log2_floor
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    thread_idx,
    block_dim,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import (
    syncwarp,
    barrier,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import is_cpu, is_gpu
from max.gpu.memory import external_memory
from std.sys.info import is_apple_gpu
from max.gpu.primitives import block
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from layout import (
    Coord,
    CoordLike,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tile_layout import Layout
from std.memory import ThinAllocation, dealloc, unsafe_stack_allocation
from std.memory.alloc import Layout as AllocLayout
from max.runtime.asyncrt import parallelism_level
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.reflection import reflect
from std.utils.coord import ComptimeInt, CoordLike
from std.utils.static_tuple import StaticTuple
from std.utils.numerics import get_accum_type, max_finite, min_finite
from comm.rms_norm_fp8 import rms_norm_fused_fp8
from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize

from .shapes import _get_start_indices_of_nth_subvolume_static

comptime _APPLE_STATIC_SHMEM_MAX_BYTES = 32 * 1024
"""Maximum number of bytes that can be used on Apple GPUs (32K)."""

comptime _APPLE_STATIC_SHMEM_MAX_COUNT[
    T: AnyType
] = _APPLE_STATIC_SHMEM_MAX_BYTES // size_of[T]()
"""Maximum number of elements of type T that can fit in Apple's
static shared memory which is 32k."""


@inline(.always)
def block_reduce[
    dtype: DType, max_warps_per_block: Int
](val: Scalar[dtype]) -> Scalar[dtype]:
    var m2_shared = unsafe_stack_allocation[
        max_warps_per_block, dtype, address_space=.SHARED
    ]()
    var m2_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()

    var warp_m2 = warp.sum(val)

    var warp_id = warp_id[broadcast=True]()
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0:
        var block_m2 = Scalar[dtype](0)

        # Only read lanes corresponding to active warps to avoid
        # reading uninitialized shared memory.
        if lane_idx < ufloordiv(block_dim.x, WARP_SIZE):
            block_m2 = m2_shared[lane_idx]

        # On some GPUs, the warp-level reduction implicitly requires all lanes
        # to participate in the reduction. Otherwise, we would get deadlocks.
        block_m2 = warp.lane_group_sum[num_lanes=max_warps_per_block](block_m2)

        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


@inline(.always)
def block_reduce_dual_sum[
    dtype: DType, max_warps_per_block: Int
](val0: Scalar[dtype], val1: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype]
]:
    """Combined block reduction for two sums using only 2 barriers."""
    var shared0 = unsafe_stack_allocation[
        max_warps_per_block, dtype, address_space=.SHARED
    ]()
    var shared1 = unsafe_stack_allocation[
        max_warps_per_block, dtype, address_space=.SHARED
    ]()
    var broadcast0 = unsafe_stack_allocation[1, dtype, address_space=.SHARED]()
    var broadcast1 = unsafe_stack_allocation[1, dtype, address_space=.SHARED]()

    var warp_sum0 = warp.sum(val0)
    var warp_sum1 = warp.sum(val1)

    var warp_id = warp_id()
    var lane_idx = lane_id()

    if lane_idx == 0:
        shared0[warp_id] = warp_sum0
        shared1[warp_id] = warp_sum1
    barrier()

    if warp_id == 0:
        var block_sum0 = Scalar[dtype](0)
        var block_sum1 = Scalar[dtype](0)

        if lane_idx < ufloordiv(block_dim.x, WARP_SIZE):
            block_sum0 = shared0[lane_idx]
            block_sum1 = shared1[lane_idx]

        block_sum0 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            block_sum0
        )
        block_sum1 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            block_sum1
        )

        if lane_idx == 0:
            broadcast0[0] = block_sum0
            broadcast1[0] = block_sum1
    barrier()
    return (broadcast0[0], broadcast1[0])


# using numerically stable Welford online algorithm to compute single pass mean and variance
def welford_update[
    dtype: DType, //
](
    val: Scalar[dtype],
    mut mean: Scalar[dtype],
    mut m2: Scalar[dtype],
    mut count: Scalar[dtype],
):
    count += 1
    var d1 = val - mean
    mean += d1 / count
    var d2 = val - mean
    m2 += d1 * d2


def welford_combine[
    dtype: DType, //
](
    mean: Scalar[dtype],
    m2: Scalar[dtype],
    count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    if count == 0:
        return
    var x_count = count + res_count
    var m = count / x_count
    var delta = mean - res_mean
    res_mean += delta * m
    res_m2 += m2 + delta * delta * res_count * m
    res_count = x_count


def welford_warp_reduce[
    dtype: DType, //
](
    thread_mean: Scalar[dtype],
    thread_m2: Scalar[dtype],
    thread_count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    res_mean = thread_mean
    res_m2 = thread_m2
    res_count = thread_count

    comptime limit = log2_floor(WARP_SIZE)

    comptime for mask in reversed(range(limit)):
        var mean = warp.shuffle_down(res_mean, UInt32(1 << mask))
        var m2 = warp.shuffle_down(res_m2, UInt32(1 << mask))
        var count = warp.shuffle_down(res_count, UInt32(1 << mask))
        welford_combine(mean, m2, count, res_mean, res_m2, res_count)


def welford_block_all_reduce[
    dtype: DType, //
](
    thread_mean: Scalar[dtype],
    thread_m2: Scalar[dtype],
    thread_count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    var mean_shared = unsafe_stack_allocation[
        WARP_SIZE, dtype, address_space=.SHARED
    ]()
    var m2_shared = unsafe_stack_allocation[
        WARP_SIZE, dtype, address_space=.SHARED
    ]()
    var count_shared = unsafe_stack_allocation[
        WARP_SIZE, dtype, address_space=.SHARED
    ]()
    var mean_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()
    var m2_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()
    var count_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()

    var warp_idx = warp_id()
    var lane_idx = lane_id()
    var warp_mean = Scalar[dtype]()
    var warp_m2 = Scalar[dtype]()
    var warp_count = Scalar[dtype]()
    welford_warp_reduce(
        thread_mean, thread_m2, thread_count, warp_mean, warp_m2, warp_count
    )
    barrier()

    if lane_idx == 0:
        mean_shared[warp_idx] = warp_mean
        m2_shared[warp_idx] = warp_m2
        count_shared[warp_idx] = warp_count
    barrier()

    if warp_idx == 0:
        if thread_idx.x < ufloordiv(block_dim.x, WARP_SIZE):
            warp_mean = mean_shared[lane_idx]
            warp_m2 = m2_shared[lane_idx]
            warp_count = count_shared[lane_idx]
        else:
            warp_mean = Scalar[dtype](0)
            warp_m2 = Scalar[dtype](0)
            warp_count = Scalar[dtype](0)
        syncwarp()
        var block_mean = Scalar[dtype](0)
        var block_m2 = Scalar[dtype](0)
        var block_count = Scalar[dtype](0)
        welford_warp_reduce(
            warp_mean, warp_m2, warp_count, block_mean, block_m2, block_count
        )
        if lane_idx == 0:
            mean_broadcast[0] = block_mean
            m2_broadcast[0] = block_m2
            count_broadcast[0] = block_count

    barrier()

    welford_combine(
        mean_broadcast[0],
        m2_broadcast[0],
        count_broadcast[0],
        res_mean,
        res_m2,
        res_count,
    )


@inline(.always)
def _rms_norm_warp_tiling_subkernel[
    dtype: DType,
    simd_width: SIMDLength,
    accum_type: DType,
    //,
    max_warps_per_block: Int,
    multiply_before_cast: Bool,
    rows_per_warp: Int = 1,
](
    row: Int,
    idx: Int,
    vec_data: SIMD[accum_type, simd_width],
    gamma_val: SIMD[dtype, simd_width],
    epsilon: Float32,
    weight_offset: Scalar[accum_type],
    num_cols: Int,
) -> SIMD[dtype, simd_width]:
    # To utilize simd vector load.
    var thread_m2: Scalar[accum_type] = (vec_data**2).reduce_add()

    var row_m2: Scalar[accum_type]
    comptime if rows_per_warp == 2:
        # Each half warp handles reduction for one row.
        row_m2 = warp.lane_group_sum[num_lanes=WARP_SIZE // 2](thread_m2)
    else:
        row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )

    var norm_factor = rsqrt(
        (row_m2 / Scalar[accum_type](num_cols)) + epsilon.cast[accum_type]()
    )
    var norm_val: SIMD[dtype, simd_width] = 0
    if idx < num_cols:
        comptime if multiply_before_cast:
            var gamma_accum = gamma_val.cast[accum_type]() + weight_offset
            norm_val = (vec_data * norm_factor * gamma_accum).cast[dtype]()
        else:
            norm_val = (vec_data * norm_factor).cast[dtype]() * (
                gamma_val + weight_offset.cast[dtype]()
            )

    return norm_val


@__name(
    t"rms_norm_gpu_warp_tiling_128_{dtype}_{epsilon.to_bits()}_{weight_offset.to_bits()}_{multiply_before_cast}"
)
def rms_norm_gpu_warp_tiling_128[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    Engine: TensorEngine,
    //,
    simd_width: Int,
    warps_per_block: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    num_rows: Int32,
    num_cols: Int32,
    input_fn: InputFnType,
    output_fn: OutputFnType,
):
    var _num_rows = Int(num_rows)
    var _num_cols = Int(num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1
    comptime half_warp_size = WARP_SIZE // 2
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var weight_offset_accum = comptime (weight_offset.cast[accum_type]())

    var vec_data = SIMD[accum_type, simd_width](0)
    var tid = thread_idx.x
    # Each warp handles 2 rows, so total rows per block is warps_per_block * 2
    var block_row = block_idx.x * warps_per_block * 2
    var warp_id = ufloordiv(tid, WARP_SIZE)
    var sub_warp_id = ufloordiv(umod(tid, WARP_SIZE), half_warp_size)
    # Each warp handles 2 rows, offset by the block's base row
    var row = block_row + warp_id * 2 + Int(sub_warp_id)
    var local_tid = umod(tid, half_warp_size)
    var idx = local_tid * simd_width

    with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
        var gamma_val = SIMD[dtype, simd_width](0)
        if row < _num_rows and idx < _num_cols:
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()
            # Prefetch gamma before reduction to overlap load with compute.
            gamma_val = gamma.load[width=simd_width, alignment=align]((idx,))

        var norm_val = _rms_norm_warp_tiling_subkernel[
            warps_per_block, multiply_before_cast, rows_per_warp=2
        ](
            row,
            idx,
            vec_data,
            gamma_val,
            epsilon,
            weight_offset_accum,
            _num_cols,
        )
        if row < _num_rows and idx < _num_cols:
            output_fn[simd_width, align](row, idx, norm_val)


# Barrier-free, SMEM-free warp-per-row RMSNorm: one warp owns a full row,
# `rows_per_block` warps per block, `warp.sum` reduction (no block barrier).
#
# `single_pass` selects the load strategy:
#   * single_pass=True: cache each lane's `chunks` exact-fit SIMD vectors in
#     registers across the reduction and normalize from registers (input read
#     ONCE). Requires exact fit (`chunks * WARP_SIZE * simd_width == num_cols`).
#     Fastest for narrow f32/bf16 rows of 1..4 vectors/lane (1.2-1.4x over
#     two-pass); beyond that the register cache spills.
#   * single_pass=False: two passes -- accumulate mean-of-squares, then reload
#     from L2 and normalize. Handles ragged tails and wider rows. `chunks` is
#     unused (pass any value, e.g. 1).
@__name(
    t"rms_norm_gpu_warp_per_row_{dtype}_{single_pass}_{chunks}_{epsilon.to_bits()}_{weight_offset.to_bits()}_{multiply_before_cast}"
)
def rms_norm_gpu_warp_per_row[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    Engine: TensorEngine,
    //,
    simd_width: Int,
    rows_per_block: Int,
    single_pass: Bool,
    chunks: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    num_rows: Int32,
    num_cols: Int32,
    input_fn: InputFnType,
    output_fn: OutputFnType,
):
    var _num_rows = Int(num_rows)
    var _num_cols = Int(num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum = comptime (epsilon.cast[accum_type]())
    var weight_offset_accum = comptime (weight_offset.cast[accum_type]())

    var tid = thread_idx.x
    var warp_in_block = ufloordiv(tid, WARP_SIZE)
    var lane = umod(tid, WARP_SIZE)
    var row = block_idx.x * rows_per_block + Int(warp_in_block)

    comptime stride = WARP_SIZE * simd_width

    comptime if single_pass:
        # Register-cached input chunks (accum precision), carried across the warp
        # reduction so the normalize pass needs no re-read.
        var vec_data = Array[SIMD[accum_type, simd_width], chunks](
            fill=SIMD[accum_type, simd_width](0)
        )

        with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
            # Single load pass: cache each chunk and accumulate mean-of-squares.
            var thread_m2 = Scalar[accum_type](0)
            if row < _num_rows:
                comptime for c in range(chunks):
                    var col = Int(lane) * simd_width + c * stride
                    vec_data[c] = input_fn[simd_width](row, col).cast[
                        accum_type
                    ]()
                    thread_m2 += (vec_data[c] ** 2).reduce_add()

            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](_num_cols)) + eps_accum
            )

            # Normalize from the cached registers (no reload).
            if row < _num_rows:
                comptime for c in range(chunks):
                    var col = Int(lane) * simd_width + c * stride
                    var gamma_val = gamma.load[
                        width=simd_width, alignment=align
                    ]((col,))
                    var norm_val: SIMD[dtype, simd_width]
                    comptime if multiply_before_cast:
                        var gamma_accum = (
                            gamma_val.cast[accum_type]() + weight_offset_accum
                        )
                        norm_val = (
                            vec_data[c] * norm_factor * gamma_accum
                        ).cast[dtype]()
                    else:
                        norm_val = (vec_data[c] * norm_factor).cast[dtype]() * (
                            gamma_val + comptime (weight_offset.cast[dtype]())
                        )
                    output_fn[simd_width, align](row, col, norm_val)
    else:
        with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
            # Pass 1: accumulate the per-thread mean-of-squares scalar only.
            var thread_m2 = Scalar[accum_type](0)
            if row < _num_rows:
                var col = Int(lane) * simd_width
                while col < _num_cols:
                    var v = input_fn[simd_width](row, col).cast[accum_type]()
                    thread_m2 += (v**2).reduce_add()
                    col += stride

            # Barrier-free, SMEM-free warp reduction (shuffle butterfly).
            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](_num_cols)) + eps_accum
            )

            # Pass 2: reload from L2 and normalize.
            if row < _num_rows:
                var col = Int(lane) * simd_width
                while col < _num_cols:
                    var v = input_fn[simd_width](row, col).cast[accum_type]()
                    var gamma_val = gamma.load[
                        width=simd_width, alignment=align
                    ]((col,))
                    var norm_val: SIMD[dtype, simd_width]
                    comptime if multiply_before_cast:
                        var gamma_accum = (
                            gamma_val.cast[accum_type]() + weight_offset_accum
                        )
                        norm_val = (v * norm_factor * gamma_accum).cast[dtype]()
                    else:
                        norm_val = (v * norm_factor).cast[dtype]() * (
                            gamma_val + comptime (weight_offset.cast[dtype]())
                        )
                    output_fn[simd_width, align](row, col, norm_val)
                    col += stride


# The warp-tiling kernel wants the row geometry as a typed `Coord` — its
# `ComptimeInt` dims are what strength-reduce the per-thread row translation —
# but it takes the shape as an explicit kernel argument, and a *bare* `Coord`
# argument corrupts the launch: the param pack lowers to one scalar device
# `.param` per dynamic leaf while the host packs the whole coord as a single
# aggregate slot, and that arity mismatch makes the driver read undefined
# parameter slots (MOCO-4307). Nesting the coord in a `DevicePassable` struct
# keeps both sides at one aggregate slot, which is how `TileTensor` already
# carries its `Layout`'s coords to a kernel;
# `max/mojo/test/gpu/host/test_coord_device_launch.mojo` covers this path.
#
# `num_cols` rides along deliberately: a *single*-field struct is transparently
# unwrapped to its field, which puts the bare param pack straight back on the
# kernel boundary and reintroduces the mismatch. Two fields keep the struct a
# struct, and the kernel needs both values anyway.
# TODO(MOCO-4307): Remove this type and use a regular Coord
@fieldwise_init
struct _RowSpec[element_types: TypeList[Trait=CoordLike, ...]](
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    var shape: Coord[*Self.element_types]
    var num_cols: Int32

    comptime device_type = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        comptime assert (
            reflect[Self].field_count() >= 2
        ), "_RowSpec must keep >= 2 fields or it is unwrapped to a bare Coord"
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(
            "_RowSpec[", Coord[*Self.element_types].get_type_name(), "]"
        )


# SM100 (B200) primary target; portable (only uses `block_reduce`, warp
# shuffle, and a per-thread grid-stride column loop — no arch intrinsics).
#
# Warp-tiling RMSNorm tuned to cut instruction overhead. Three structural
# choices vs the old single-chunk-per-thread form (which was instruction-issue
# bound on B200, not bandwidth bound -- high SM issue but low L2/HBM
# throughput):
#
#   1. The rank-N row -> base-coords translation
#      (`_get_start_indices_of_nth_subvolume_static`) is hoisted and run ONCE
#      per thread, then reused for every chunk's load AND store. The old 2D
#      wrappers ran that divmod chain twice per thread (once in `input_fn_2d`,
#      once in `output_fn_2d`); we now take the rank-N `input_fn`/`output_fn`
#      directly and only mutate `base[rank - 1]` per chunk.
#   2. Each thread processes `chunks_per_thread` independent vector chunks
#      (unrolled LDGs cached in registers), so the launch uses fewer threads
#      per block -> fewer warps -> a cheaper two-barrier `block_reduce`, and
#      the loads pipeline (ILP). Data stays in registers between the reduction
#      and the normalize pass (still a single global read).
#   3. When `exact_fit` (block_dim * simd_width * chunks_per_thread == cols),
#      every thread is fully active, so the per-chunk `col < num_cols` guards
#      (the ISETP/SEL bloat) are dropped at comptime. A guarded variant
#      (`exact_fit=False`) handles ragged tails.
@__name(
    t"rms_norm_gpu_warp_tiling_{dtype}_{chunks_per_thread}_{exact_fit}_{epsilon.to_bits()}_{weight_offset.to_bits()}_{multiply_before_cast}"
)
def rms_norm_gpu_warp_tiling[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    rank: Int,
    Engine: TensorEngine,
    shape_types: TypeList[Trait=CoordLike, ...],
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    chunks_per_thread: Int,
    exact_fit: Bool,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](Coord, SIMD[dtype, width]) -> None,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    row_spec: _RowSpec[shape_types],
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    input_fn: InputFnType,
    output_fn: OutputFnType,
):
    comptime assert (
        shape_types.length == rank
    ), "shape_types.length must be the same as rank"
    var _num_cols = Int(row_spec.num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum = comptime (epsilon.cast[accum_type]())
    var weight_offset_accum = comptime (weight_offset.cast[accum_type]())

    var tid = thread_idx.x
    var row = block_idx.x
    var bdim = block_dim.x

    # Hoist the rank-N row translation ONCE; reuse the base for load and store.
    var base = _get_start_indices_of_nth_subvolume_static(row, row_spec.shape)

    # Per-chunk register-cached input (in accum precision) and gamma weights,
    # carried across the reduction so the normalize pass needs no re-read.
    var vec_data = Array[SIMD[accum_type, simd_width], chunks_per_thread](
        fill=SIMD[accum_type, simd_width](0)
    )
    var gamma_val = Array[SIMD[dtype, simd_width], chunks_per_thread](
        fill=SIMD[dtype, simd_width](0)
    )

    with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
        var thread_m2 = Scalar[accum_type](0)

        comptime for c in range(chunks_per_thread):
            var col = (c * bdim + tid) * simd_width
            comptime if exact_fit:
                base[rank - 1] = col
                vec_data[c] = input_fn[simd_width](Coord(base)).cast[
                    accum_type
                ]()
                gamma_val[c] = gamma.load[width=simd_width, alignment=align](
                    (col,)
                )
                thread_m2 += (vec_data[c] ** 2).reduce_add()
            else:
                if col < _num_cols:
                    base[rank - 1] = col
                    vec_data[c] = input_fn[simd_width](Coord(base)).cast[
                        accum_type
                    ]()
                    gamma_val[c] = gamma.load[
                        width=simd_width, alignment=align
                    ]((col,))
                    thread_m2 += (vec_data[c] ** 2).reduce_add()

        var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / row_spec.num_cols.cast[accum_type]()) + eps_accum
        )

        comptime for c in range(chunks_per_thread):
            var col = (c * bdim + tid) * simd_width

            @inline(.always)
            def _normalize() {imm} -> SIMD[dtype, simd_width]:
                comptime if multiply_before_cast:
                    var gamma_accum = (
                        gamma_val[c].cast[accum_type]() + weight_offset_accum
                    )
                    return (vec_data[c] * norm_factor * gamma_accum).cast[
                        dtype
                    ]()
                else:
                    return (vec_data[c] * norm_factor).cast[dtype]() * (
                        gamma_val[c] + comptime (weight_offset.cast[dtype]())
                    )

            comptime if exact_fit:
                base[rank - 1] = col
                output_fn[simd_width, align](Coord(base), _normalize())
            else:
                if col < _num_cols:
                    base[rank - 1] = col
                    output_fn[simd_width, align](Coord(base), _normalize())


@inline(.always)
def _rms_norm_gpu_block_subkernel[
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    multiply_before_cast: Bool,
    InputFnType: ImplicitlyCopyable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
](
    input_fn: InputFnType,
    output_fn: OutputFnType,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var thread_m2 = Scalar[accum_type](0)
    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width
        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            thread_m2 += (vec_data**2).reduce_add()

    var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_m2
    )
    var norm_factor = rsqrt((row_m2 / Scalar[accum_type](num_cols)) + eps_accum)

    # Need a pass again to perform in place normalization.
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width

        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            var norm_val: SIMD[dtype, simd_width]
            var gamma_val = gamma.load[width=simd_width, alignment=align](
                (offset,)
            )

            if multiply_before_cast:
                var gamma_accum = (
                    gamma_val.cast[accum_type]() + weight_offset_accum
                )
                norm_val = (vec_data * norm_factor * gamma_accum).cast[dtype]()
            else:
                norm_val = (vec_data * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset.cast[dtype]()
                )

            output_fn[simd_width, align](row, offset, norm_val)


@__name(
    t"rms_norm_gpu_block_{dtype}_{epsilon.to_bits()}_{weight_offset.to_bits()}_{multiply_before_cast}"
)
def rms_norm_gpu_block[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    Engine: TensorEngine,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    gamma: TileTensor[dtype, LayoutType, origin, Engine=Engine],
    num_cols: Int32,
    input_fn: InputFnType,
    output_fn: OutputFnType,
):
    var _num_cols = Int(num_cols)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    with PDL[pdl_level == PDLLevel.OVERLAP_AT_BEGINNING]():
        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            multiply_before_cast,
        ](
            input_fn,
            output_fn,
            gamma,
            epsilon,
            comptime (weight_offset.cast[dtype]()),
            _num_cols,
        )


# `input_fn` and `output_fn` run on the device, so they must copy (`{var}`)
# what they capture. Callers may norm in place, with both callbacks capturing
# the same buffer: every element is read before the thread reading it writes it.
@__unsafe_nested_origins_read_only
def rms_norm_gpu[
    dtype: DType,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](Coord, SIMD[dtype, width]) -> None,
    //,
    rank: Int,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.OVERLAP_AT_BEGINNING,
](
    input_fn: InputFnType,
    output_fn: OutputFnType,
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    ctx: DeviceContext,
) raises:
    # `shape` arrives as a `Coord`, with statically-known outer dims encoded in
    # its type, and the lambdas are `Coord`-form all the way down to the
    # kernels.
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    if rank == 0:
        return

    # Derive the number of columns from the `gamma` input as this value may be
    # statically known.
    var cols = Int(gamma.dim[0]())

    if cols == 0:
        return

    var rows = Int(shape.product()) // cols

    # A rank owns an empty shard when rows < TP group size (bs=1 decode at TP4)
    # and `enqueue_function` rejects the zero `grid_dim` every launch derives.
    if rows == 0:
        return

    # The 2D wrappers translate each flattened `(row, col)` back to the original
    # n-D coordinate. The row -> n-D decomposition divides by the outer dims; on
    # the static-shape path those divisors are the `ComptimeInt` dims carried in
    # `type_of(shape)` (a `Coord`), so the per-row `divmod` strength-reduces to
    # magic-multiply + shift. Dynamic dims fall back to the coord's
    # runtime leaf, so this path is behavior-identical to the pre-migration
    # `_get_start_indices_of_nth_subvolume` form for non-static shapes.
    #
    # The wrappers copy-capture (`{var}`) `shape_layout` and the callbacks:
    # they reach the GPU kernels by value through `host_args`, and a captured
    # *local* `var` held by reference would be a host-stack pointer on device.
    # Without the copy the rank-N divmod reads garbage outer dims on device
    # (rank-2 is unaffected since its outer translation is trivial; rank>=3
    # produces wrong results / launch failures).
    #
    # The carrier is a `Layout`, not the `Coord` itself: a captured `Coord`
    # lowers to one scalar device `.param` per dynamic leaf while the host packs
    # it as a single capture slot, and that arity mismatch corrupts the launch
    # (MOCO-4307). A `row_major` `Layout` is the one carrier that keeps the
    # capture a single aggregate `.param`, so the typed `Coord` — static dims
    # and all — arrives intact via `shape_coord()`.
    var shape_layout = row_major(shape)

    @inline(.always)
    def output_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
        var shape_layout, var output_fn
    } -> None:
        var indices = _get_start_indices_of_nth_subvolume_static(
            row, shape_layout.shape_coord()
        )
        indices[rank - 1] = col
        output_fn[simd_width, alignment](Coord(indices), val)

    @inline(.always)
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {var shape_layout, var input_fn} -> SIMD[
        dtype, simd_width
    ]:
        var indices = _get_start_indices_of_nth_subvolume_static(
            row, shape_layout.shape_coord()
        )
        indices[rank - 1] = col
        return input_fn[simd_width](Coord(indices))

    comptime simd_width = simd_width_of[dtype, target=ctx.target]()
    comptime warp_size = ctx.default_device_info.warp_size
    comptime max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // warp_size
    )
    comptime sm_version = ctx.default_device_info.version
    comptime sm_count = ctx.default_device_info.sm_count
    comptime warp_per_row_rows_per_block = 8
    var warp_per_row_min_grid = 3 * sm_count
    var warp_per_row_region = (
        ceildiv(rows, warp_per_row_rows_per_block) >= warp_per_row_min_grid
    )
    # Conservative floor: below one block per SM, chunking can't raise
    # occupancy and only reassociates the mean-of-squares reduction (the real
    # crossover is higher and width-dependent; this sits safely below it).
    var enough_rows_to_chunk = rows >= sm_count

    var grid_dim = rows
    var block_dim = min(
        align_up(ceildiv(cols, simd_width), warp_size),
        warp_size * max_warps_per_block,
    )

    # Warp-tiling launch geometry for a given per-thread (`eff_simd`, `chunks`):
    # one block per row, `_wt_threads_per_block` threads (rounded to whole
    # warps, capped at the device max), each owning `eff_simd * chunks` columns.
    # Single source of truth for both the launcher and the warp-per-row gate
    # below.
    @inline(.always)
    def _wt_threads_per_block[eff_simd: Int, chunks: Int]() {imm} -> Int:
        var threads = ceildiv(ceildiv(cols, eff_simd), chunks)
        return min(
            align_up(threads, warp_size),
            warp_size * max_warps_per_block,
        )

    # `exact` means every thread is fully active (the block tiles the row with
    # no ragged tail), so the unguarded kernel can be used.
    @inline(.always)
    def _wt_exact[eff_simd: Int, chunks: Int]() {imm} -> Bool:
        return (
            _wt_threads_per_block[eff_simd, chunks]() * eff_simd * chunks
        ) == cols

    # Within the warp-tiling column range, warp-tiling (register-cached chunks)
    # beats the barrier-free warp-per-row kernel on SM100 (B200) whenever it
    # tiles the row exactly *and* the row is wide enough to amortize the
    # inter-warp block reduction: a measured 1.1-1.4x at 2048..8192 cols.
    # Warp-per-row wins in two cases, both of which must stay on it:
    #   1. ragged tail (e.g. 8192x2880, cols not a clean multiple of the
    #      per-thread tile -> wasted threads + per-chunk bounds guards), and
    #   2. narrow rows (cols <= 1024), where the one-warp-per-row kernel has no
    #      inter-warp barrier to pay and warp-tiling's block reduce dominates
    #      (measured: 8192x256 1.9x, 8192x512 1.7x, 8192x1024 1.3x slower under
    #      warp-tiling).
    # So prefer warp-tiling only for exact-fit rows past the narrow-row floor;
    # everything else keeps warp-per-row. Uses the native-width chunk count the
    # dispatch below would pick (1 up to one warp-row, 2 up to two, else 4) and
    # the shared geometry.
    #
    # This is only consulted to gate warp-per-row, which itself only runs for
    # `cols <= WARP_SIZE * simd_width * max_warps_per_block`. Within that range
    # the dispatch always launches at native `simd_width` with these same chunk
    # counts, so the gate matches the kernel that runs. The wider-row branch
    # below (`simd_width * 2`) lies entirely past the warp-per-row bound, so the
    # native-width value computed here is never read against it.
    #
    # `warp_tiling_min_cols` is the narrow-row floor: warp-tiling is preferred
    # only above it. 1024 = WARP_SIZE * simd_width * 4 (= one max-width
    # per-thread tile across a single warp) is the measured crossover on B200
    # bf16 -- the largest exact-fit width that still loses to warp-per-row.
    var warp_tiling_min_cols = warp_size * simd_width * 4
    var warp_tiling_exact_fit: Bool
    if cols <= (warp_size * simd_width):
        warp_tiling_exact_fit = _wt_exact[simd_width, 1]()
    elif cols <= (warp_size * simd_width * 2):
        warp_tiling_exact_fit = _wt_exact[simd_width, 2]()
    else:
        warp_tiling_exact_fit = _wt_exact[simd_width, 4]()
    var warp_tiling_exact = warp_tiling_exact_fit and (
        cols > warp_tiling_min_cols
    )

    # Launch the multi-chunk warp-tiling kernel. `exact_fit` (every thread
    # fully active, no ragged tail) is decided at runtime and selects the
    # unguarded instantiation.
    @inline(.always)
    def _launch_warp_tiling[eff_simd: Int, chunks: Int]() raises {imm}:
        var threads_per_block = _wt_threads_per_block[eff_simd, chunks]()
        var exact = _wt_exact[eff_simd, chunks]()

        @inline(.always)
        def _enqueue[exact_fit: Bool]() raises {imm}:
            comptime kernel = rms_norm_gpu_warp_tiling[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                Engine=gamma.Engine,
                rank=rank,
                shape_types=type_of(shape).element_types,
                eff_simd,
                max_warps_per_block,
                chunks,
                exact_fit,
                InputFnType,
                OutputFnType,
                epsilon=epsilon,
                weight_offset=weight_offset,
                multiply_before_cast=multiply_before_cast,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel](
                _RowSpec[type_of(shape).element_types](shape, Int32(cols)),
                gamma,
                host_args=(input_fn, output_fn),
                grid_dim=rows,
                block_dim=threads_per_block,
                attributes=pdl_launch_attributes(pdl_level),
            )

        if exact:
            _enqueue[True]()
        else:
            _enqueue[False]()

    # _rms_norm_input_alignment trusts gates like this one. Loosen or
    # remove it and update that function too.
    if cols % simd_width == 0:
        # When the number of columns are less enough that they can be placed in
        # registers we do warp tiling which is a single pass to do mean/var
        # computation and normalization.
        if cols <= 128 and dtype == .bfloat16:
            # Experimentally determined to be the best - tapers off at 2.
            comptime warps_per_block = 2
            # Each warp handles 2 rows, so total rows per block is warps_per_block * 2.
            block_dim = warps_per_block * warp_size
            grid_dim = ceildiv(rows, warps_per_block * 2)

            comptime kernel = rms_norm_gpu_warp_tiling_128[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                Engine=gamma.Engine,
                simd_width,
                warps_per_block,
                type_of(input_fn_2d),
                type_of(output_fn_2d),
                epsilon=epsilon,
                weight_offset=weight_offset,
                multiply_before_cast=multiply_before_cast,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel](
                gamma,
                Int32(rows),
                Int32(cols),
                host_args=(input_fn_2d, output_fn_2d),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
        elif (
            cols >= 128
            and cols <= (warp_size * simd_width * max_warps_per_block)
            and warp_per_row_region
            and not warp_tiling_exact
        ):
            comptime rows_per_block = warp_per_row_rows_per_block
            block_dim = rows_per_block * warp_size
            grid_dim = ceildiv(rows, rows_per_block)

            # Single-pass register-cached warp-per-row for narrow, exact-fit
            # rows. Keeps the 8-rows/block high-occupancy geometry of the
            # two-pass kernel but loads each lane's row chunks ONCE (cached in
            # registers across the `warp.sum`) instead of reloading + resquaring
            # in pass 2.
            comptime sp_stride = warp_size * simd_width
            var sp_chunks = cols // sp_stride
            if (
                dtype in (DType.float32, DType.bfloat16)
                and (cols % sp_stride == 0)
                and (sp_chunks >= 1)
                and (sp_chunks <= 4)
            ):
                comptime for cc in range(1, 5):
                    if sp_chunks == cc:
                        comptime kernel = rms_norm_gpu_warp_per_row[
                            mut=gamma.mut,
                            LayoutType=gamma.LayoutType,
                            origin=gamma.origin,
                            Engine=gamma.Engine,
                            simd_width,
                            rows_per_block,
                            True,
                            cc,
                            type_of(input_fn_2d),
                            type_of(output_fn_2d),
                            epsilon=epsilon,
                            weight_offset=weight_offset,
                            multiply_before_cast=multiply_before_cast,
                            pdl_level=pdl_level,
                        ]
                        ctx.enqueue_function[kernel](
                            gamma,
                            Int32(rows),
                            Int32(cols),
                            host_args=(input_fn_2d, output_fn_2d),
                            grid_dim=grid_dim,
                            block_dim=block_dim,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
            else:
                comptime kernel = rms_norm_gpu_warp_per_row[
                    mut=gamma.mut,
                    LayoutType=gamma.LayoutType,
                    origin=gamma.origin,
                    Engine=gamma.Engine,
                    simd_width,
                    rows_per_block,
                    False,
                    1,
                    type_of(input_fn_2d),
                    type_of(output_fn_2d),
                    epsilon=epsilon,
                    weight_offset=weight_offset,
                    multiply_before_cast=multiply_before_cast,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel](
                    gamma,
                    Int32(rows),
                    Int32(cols),
                    host_args=(input_fn_2d, output_fn_2d),
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(pdl_level),
                )
        elif cols <= (warp_size * simd_width * max_warps_per_block):
            # CDNA4 (MI355X): when there are enough rows to keep the GPU busy,
            # use a 2x-wider per-thread SIMD so each row's block needs half the
            # warps. This halves the inter-warp block reduction cost and
            # doubles blocks-per-CU, lifting achieved HBM bandwidth by ~15-30%
            # on prefill-sized shapes. It is gated on row count because the
            # smaller block lowers total occupancy when rows are few (a net loss
            # below ~8x the CU count).
            comptime sw_wide = simd_width * 2
            comptime widen_ok = sm_version == "CDNA4"
            var enough_rows = rows >= 8 * sm_count
            if widen_ok and enough_rows and cols % sw_wide == 0:
                _launch_warp_tiling[sw_wide, 1]()
            elif not enough_rows_to_chunk:
                _launch_warp_tiling[simd_width, 1]()
            else:
                # Narrow rows: a single full-width pass keeps the block small;
                # split into independent chunks (ILP + smaller block_reduce)
                # once there are enough columns to fill ~2+ chunks/thread.
                if cols <= (warp_size * simd_width):
                    _launch_warp_tiling[simd_width, 1]()
                elif cols <= (warp_size * simd_width * 2):
                    _launch_warp_tiling[simd_width, 2]()
                else:
                    _launch_warp_tiling[simd_width, 4]()
        elif (
            cols <= (warp_size * (simd_width * 2) * max_warps_per_block)
            and cols % (simd_width * 2) == 0
        ):
            # Wider rows: double the vector width and, for high-row launches,
            # split into chunks.
            if not enough_rows_to_chunk:
                _launch_warp_tiling[simd_width * 2, 1]()
            elif cols <= (warp_size * simd_width * 2 * 2):
                _launch_warp_tiling[simd_width * 2, 2]()
            else:
                _launch_warp_tiling[simd_width * 2, 4]()
        else:
            comptime kernel = rms_norm_gpu_block[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                Engine=gamma.Engine,
                simd_width,
                max_warps_per_block,
                type_of(input_fn_2d),
                type_of(output_fn_2d),
                epsilon=epsilon,
                weight_offset=weight_offset,
                multiply_before_cast=multiply_before_cast,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel](
                gamma,
                Int32(cols),
                host_args=(input_fn_2d, output_fn_2d),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
    else:
        comptime kernel = rms_norm_gpu_block[
            mut=gamma.mut,
            LayoutType=gamma.LayoutType,
            origin=gamma.origin,
            Engine=gamma.Engine,
            1,
            max_warps_per_block,
            type_of(input_fn_2d),
            type_of(output_fn_2d),
            epsilon=epsilon,
            weight_offset=weight_offset,
            multiply_before_cast=multiply_before_cast,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel](
            gamma,
            Int32(cols),
            host_args=(input_fn_2d, output_fn_2d),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def _sum_to_mean[
    dtype: DType, //
](sum_val: Scalar[dtype], n: Int) -> Scalar[dtype]:
    comptime if dtype.is_integral():
        return sum_val // Scalar[dtype](n)
    return sum_val / Scalar[dtype](n)


def rms_norm_cpu[
    dtype: DType,
    InputFnType: def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    //,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool,
](
    input_fn: InputFnType,
    output_fn: OutputFnType,
    gamma: TileTensor[mut=False, dtype, ...],
    out_shape: Coord,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1
    comptime assert out_shape.rank == 2, "out_shape must be rank 2"

    comptime simd_width = simd_width_of[dtype]()

    var num_rows = Int(out_shape[0])
    var num_cols = Int(out_shape[1])

    var simd_loop_end = align_down(num_cols, simd_width)
    comptime intermediate_type = get_accum_type[dtype]()

    # PyTorch converts the input to float32 before computing the RMS norm
    # https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L76
    for var row in range(num_rows):
        var sum_simd = SIMD[intermediate_type, simd_width]()
        for col in range(0, simd_loop_end, simd_width):
            sum_simd += (
                input_fn[simd_width](row, col).cast[intermediate_type]() ** 2
            )

        var sum_val = sum_simd.reduce_add()
        for col in range(simd_loop_end, num_cols):
            sum_val += input_fn[1](row, col).cast[intermediate_type]() ** 2

        var mean_val = _sum_to_mean(sum_val, num_cols)
        var norm_factor = rsqrt(
            mean_val + comptime (epsilon.cast[intermediate_type]())
        )

        def _normalize[
            simd_width: Int
        ](col: Int) {gamma, imm input_fn, imm output_fn, mut}:
            var input_val = input_fn[simd_width](row, col).cast[
                intermediate_type
            ]()
            var gamma_val = gamma.load[width=simd_width, alignment=1]((col,))
            var norm_val: SIMD[dtype, simd_width]

            if multiply_before_cast:
                var gamma_offset = gamma_val + comptime (
                    weight_offset.cast[dtype]()
                )
                norm_val = (input_val * norm_factor).cast[
                    dtype
                ]() * gamma_offset
            else:
                norm_val = (input_val * norm_factor).cast[dtype]() * (
                    gamma_val + comptime (weight_offset.cast[dtype]())
                )

            output_fn[simd_width, 1](row, col, norm_val)

        vectorize[simd_width](num_cols, _normalize)


def rms_norm_cpu[
    dtype: DType,
    InputFnType: def[width: Int](Coord) -> SIMD[dtype, width],
    OutputFnType: def[width: SIMDLength, alignment: Int](
        Coord, SIMD[dtype, width]
    ) -> None,
    //,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool,
](
    input_fn: InputFnType,
    output_fn: OutputFnType,
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    ctx: Optional[DeviceContext.CPU] = None,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime rank = shape.rank

    var last_dim = Int(shape[rank - 1])
    var prod_all_but_last_dim = Int(shape.product()) // last_dim

    var num_workers = min(parallelism_level(ctx), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    def task_func(
        thread_id: Int,
    ) {var chunk_size, var prod_all_but_last_dim, var last_dim, imm,}:
        var num_rows = min(
            chunk_size, prod_all_but_last_dim - thread_id * chunk_size
        )
        var row_idx = thread_id * chunk_size

        @inline(.always)
        def output_fn_2d[
            simd_width: SIMDLength, alignment: Int
        ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
            var row_idx, imm
        } -> None:
            # Translate a given 2D index back to the original n-D tensor.
            var indices = _get_start_indices_of_nth_subvolume_static(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            output_fn[simd_width, alignment](Coord(indices), val)

        @inline(.always)
        def input_fn_2d[
            simd_width: Int
        ](row: Int, col: Int) {var row_idx, imm} -> SIMD[dtype, simd_width]:
            # Translate a given 2D index back to the original n-D tensor.
            var indices = _get_start_indices_of_nth_subvolume_static(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            return input_fn[simd_width](Coord(indices))

        rms_norm_cpu[
            epsilon=epsilon,
            weight_offset=weight_offset,
            multiply_before_cast=multiply_before_cast,
        ](
            input_fn_2d,
            output_fn_2d,
            gamma,
            out_shape=Coord(num_rows, last_dim),
        )

    sync_parallelize(task_func, num_workers, ctx)


@inline(.always)
def _rms_norm_input_alignment[
    dtype: DType, width: Int, target: StaticString
]() -> Int:
    """The alignment an rms_norm input_fn can claim for a load of `width`.

    Sound only on GPU, whose dispatchers prove `cols % width == 0` before
    requesting `width > 1`. CPU always claims 1: its vectorized loop runs
    regardless of `num_cols % simd_width`.
    """
    comptime if is_cpu[target]():
        return 1
    else:
        return align_of[SIMD[dtype, width]]()


# The KV cache norms run in place: `input_0_fn` and `output_fn` may capture
# the same cache. Every element is read before the thread reading it writes it.
@__unsafe_nested_origins_read_only
@inline(.always)
def _rms_norm_impl[
    dtype: DType,
    rank: Int,
    Input0FnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](Coord, SIMD[dtype, width]) -> None,
    /,
    epsilon: Float32,
    weight_offset: Float32,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    input_0_fn: Input0FnType,
    output_fn: OutputFnType,
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    ctx: DeviceContext,
) raises:
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"

    var reduce_dim_size = Int(shape[rank - 1])

    # Note: we only support reduction along the last dimension
    if Int(gamma.dim[0]()) != reduce_dim_size:
        raise Error(
            "Gamma size "
            + String(gamma.dim[0]())
            + " does not match dimension of reduction "
            + String(reduce_dim_size)
            + "."
        )

    if Int(shape.product()) == 0:
        # Nothing to do.
        return

    @inline(.always)
    def input_fn_target[
        width: Int
    ](coords: Coord) {var input_0_fn} -> SIMD[dtype, width]:
        comptime align = _rms_norm_input_alignment[dtype, width, target]()
        return input_0_fn[width, align](coords)

    comptime if is_cpu[target]():
        rms_norm_cpu[
            epsilon=epsilon,
            weight_offset=weight_offset,
            multiply_before_cast=multiply_before_cast,
        ](
            input_fn_target,
            output_fn,
            shape,
            gamma,
            Optional(ctx.assert_cpu_target()),
        )
    elif is_gpu[target]():
        rms_norm_gpu[
            rank,
            epsilon=epsilon,
            weight_offset=weight_offset,
            multiply_before_cast=multiply_before_cast,
        ](
            input_fn_target,
            output_fn,
            shape,
            gamma,
            ctx,
        )
    else:
        comptime assert False, "unsupported target " + target


# ===----------------------------------------------------------------------=== #
# Fused Q/K RMSNorm apply: kernel + CPU/GPU entry points + dispatcher
# ===----------------------------------------------------------------------=== #
#
# SM100 (B200) primary target; portable to SM90 / CDNA4 / Apple (pure
# elementwise apply, no arch-specific intrinsics, no block reduction).
#
# Given the already-reduced per-row statistics `qk_var` of shape `[M, 2]`
# (col 0 = mean(q^2), col 1 = mean(k^2), float32), and per-column float32
# scales `gamma_q [Nq]` / `gamma_k [Nk]`, this applies the QK-RMSNorm scale to
# Q `[M, Nq]` and K `[M, Nk]` in a SINGLE launch:
#
#     rs_q = rsqrt(qk_var[m, 0] + epsilon)            # float32
#     q_out[m, c] = cast( (cast(q[m,c], f32) * rs_q) * gamma_q[c], out_dtype )
#     rs_k = rsqrt(qk_var[m, 1] + epsilon)
#     k_out[m, c] = cast( (cast(k[m,c], f32) * rs_k) * gamma_k[c], out_dtype )
#
# The grouping is `((x_f32 * rs) * gamma)` then cast (NOT `x_f32 * (rs*gamma)`)
# to bit-match the unfused graph it replaces. This fuses the ~7 tiny
# elementwise/View kernels of the QK-norm apply chain into one launch, keeping
# the grid tiny for small-M decode (grid = (rows, 2); e.g. M=16 -> 32 blocks).


@__name(t"apply_qk_rms_norm_gpu_block_{in_dtype}_{out_dtype}")
def apply_qk_rms_norm_gpu_block[
    in_dtype: DType,
    out_dtype: DType,
    q_out_mut: Bool,
    q_out_layout: TensorLayout,
    q_out_origin: Origin[mut=q_out_mut],
    q_out_engine: TensorEngine,
    k_out_mut: Bool,
    k_out_layout: TensorLayout,
    k_out_origin: Origin[mut=k_out_mut],
    k_out_engine: TensorEngine,
    gamma_q_mut: Bool,
    gamma_q_layout: TensorLayout,
    gamma_q_origin: Origin[mut=gamma_q_mut],
    gamma_q_engine: TensorEngine,
    gamma_k_mut: Bool,
    gamma_k_layout: TensorLayout,
    gamma_k_origin: Origin[mut=gamma_k_mut],
    gamma_k_engine: TensorEngine,
    var_mut: Bool,
    var_layout: TensorLayout,
    var_origin: Origin[mut=var_mut],
    var_engine: TensorEngine,
    q_layout: TensorLayout,
    q_origin: Origin,
    q_engine: TensorEngine,
    k_layout: TensorLayout,
    k_origin: Origin,
    k_engine: TensorEngine,
    //,
    simd_width: Int,
](
    q_out: TileTensor[
        out_dtype, q_out_layout, q_out_origin, Engine=q_out_engine
    ],
    k_out: TileTensor[
        out_dtype, k_out_layout, k_out_origin, Engine=k_out_engine
    ],
    gamma_q: TileTensor[
        .float32, gamma_q_layout, gamma_q_origin, Engine=gamma_q_engine
    ],
    gamma_k: TileTensor[
        .float32, gamma_k_layout, gamma_k_origin, Engine=gamma_k_engine
    ],
    qk_var: TileTensor[.float32, var_layout, var_origin, Engine=var_engine],
    q: TileTensor[in_dtype, q_layout, q_origin, Engine=q_engine],
    k: TileTensor[in_dtype, k_layout, k_origin, Engine=k_engine],
    epsilon: Float32,
    q_cols: Int32,
    k_cols: Int32,
) where (q_out_mut and k_out_mut):
    """Fused per-element QK-RMSNorm apply for Q and K in a single launch.

    The grid is 2D: `block_idx.x` selects the row and `block_idx.y` selects the
    operand (0 = Q, 1 = K). Each block owns one (row, operand) and threads
    grid-stride across that operand's columns, applying `((x * rs) * gamma)`.
    All operands (`q [M, Nq]`, `k [M, Nk]`, `gamma_q [Nq]`, `gamma_k [Nk]`,
    `qk_var [M, 2]`, and the outputs `q_out [M, Nq]` / `k_out [M, Nk]`) are
    loaded/stored directly from their `TileTensor`s, matching the in-file
    rms_norm `gamma.load[...]` idiom.
    """
    var _q_cols = Int(q_cols)
    var _k_cols = Int(k_cols)
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    comptime align = align_of[SIMD[.float32, simd_width]]()

    var tid = thread_idx.x
    var row = block_idx.x
    # block_idx.y is uniform across the block, so this branch never diverges.
    var is_k = block_idx.y == 1
    var num_cols = _k_cols if is_k else _q_cols

    with PDL():
        # rsqrt of the (already cross-rank reduced) per-row mean of squares.
        var rs = rsqrt(qk_var.load[width=1]((row, block_idx.y)) + epsilon)

        # Each block owns a single (row, operand); threads grid-stride the cols.
        for x in range(ceildiv(ceildiv(num_cols, simd_width), block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                if is_k:
                    var xf = k.load[width=simd_width]((row, offset)).cast[
                        .float32
                    ]()
                    var g = gamma_k.load[width=simd_width, alignment=align](
                        (offset,)
                    )
                    k_out.store(
                        (Int(row), offset),
                        ((xf * rs) * g).cast[out_dtype](),
                    )
                else:
                    var xf = q.load[width=simd_width]((row, offset)).cast[
                        .float32
                    ]()
                    var g = gamma_q.load[width=simd_width, alignment=align](
                        (offset,)
                    )
                    q_out.store(
                        (Int(row), offset),
                        ((xf * rs) * g).cast[out_dtype](),
                    )


def apply_qk_rms_norm_gpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, .float32, ...],
    gamma_k: TileTensor[mut=False, .float32, ...],
    qk_var: TileTensor[mut=False, .float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Float32,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Launches the fused Q/K RMSNorm apply: one launch, grid (rows, 2).

    `block_idx.y` selects Q (0) or K (1). Block dim is sized for the wider of
    the two operands; the narrower operand simply leaves trailing threads idle.
    """
    if rows == 0 or (q_cols == 0 and k_cols == 0):
        return

    comptime simd_width = simd_width_of[in_dtype, target=ctx.target]()
    comptime warp_size = ctx.default_device_info.warp_size
    comptime max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // warp_size
    )

    # 2D grid: x = row, y = operand (0 = Q, 1 = K). One block per (row, operand)
    # keeps the grid tiny for small-M decode (e.g. M=16 -> 32 blocks).
    var grid_dim = (rows, 2)
    var max_cols = max(q_cols, k_cols)

    if q_cols % simd_width == 0 and k_cols % simd_width == 0:
        # Vectorized loads/stores; size threads for the wider operand.
        var block_dim = min(
            align_up(ceildiv(max_cols, simd_width), warp_size),
            warp_size * max_warps_per_block,
        )
        comptime kernel = apply_qk_rms_norm_gpu_block[
            in_dtype=q.dtype,
            out_dtype=out_dtype,
            q_out_mut=q_out.mut,
            q_out_layout=q_out.LayoutType,
            q_out_origin=q_out.origin,
            q_out_engine=q_out.Engine,
            k_out_mut=k_out.mut,
            k_out_layout=k_out.LayoutType,
            k_out_origin=k_out.origin,
            k_out_engine=k_out.Engine,
            gamma_q_mut=gamma_q.mut,
            gamma_q_layout=gamma_q.LayoutType,
            gamma_q_origin=gamma_q.origin,
            gamma_q_engine=gamma_q.Engine,
            gamma_k_mut=gamma_k.mut,
            gamma_k_layout=gamma_k.LayoutType,
            gamma_k_origin=gamma_k.origin,
            gamma_k_engine=gamma_k.Engine,
            var_mut=qk_var.mut,
            var_layout=qk_var.LayoutType,
            var_origin=qk_var.origin,
            var_engine=qk_var.Engine,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            q_engine=q.Engine,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            k_engine=k.Engine,
            simd_width=simd_width,
        ]
        ctx.enqueue_function[kernel](
            q_out,
            k_out,
            gamma_q,
            gamma_k,
            qk_var,
            q,
            k,
            epsilon,
            Int32(q_cols),
            Int32(k_cols),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # General N (incl. non-multiple of vector width): scalar loads/stores.
        var block_dim = min(
            align_up(max_cols, warp_size),
            warp_size * max_warps_per_block,
        )
        comptime kernel = apply_qk_rms_norm_gpu_block[
            in_dtype=q.dtype,
            out_dtype=out_dtype,
            q_out_mut=q_out.mut,
            q_out_layout=q_out.LayoutType,
            q_out_origin=q_out.origin,
            q_out_engine=q_out.Engine,
            k_out_mut=k_out.mut,
            k_out_layout=k_out.LayoutType,
            k_out_origin=k_out.origin,
            k_out_engine=k_out.Engine,
            gamma_q_mut=gamma_q.mut,
            gamma_q_layout=gamma_q.LayoutType,
            gamma_q_origin=gamma_q.origin,
            gamma_q_engine=gamma_q.Engine,
            gamma_k_mut=gamma_k.mut,
            gamma_k_layout=gamma_k.LayoutType,
            gamma_k_origin=gamma_k.origin,
            gamma_k_engine=gamma_k.Engine,
            var_mut=qk_var.mut,
            var_layout=qk_var.LayoutType,
            var_origin=qk_var.origin,
            var_engine=qk_var.Engine,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            q_engine=q.Engine,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            k_engine=k.Engine,
            simd_width=1,
        ]
        ctx.enqueue_function[kernel](
            q_out,
            k_out,
            gamma_q,
            gamma_k,
            qk_var,
            q,
            k,
            epsilon,
            Int32(q_cols),
            Int32(k_cols),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def apply_qk_rms_norm_cpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, .float32, ...],
    gamma_k: TileTensor[mut=False, .float32, ...],
    qk_var: TileTensor[mut=False, .float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Float32,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
):
    """Applies Q/K RMS normalization across independent CPU SIMD lanes."""
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    comptime assert q_out.flat_rank == 2, "q_out must have rank 2"
    comptime assert k_out.flat_rank == 2, "k_out must have rank 2"
    comptime assert gamma_q.flat_rank == 1, "gamma_q must have rank 1"
    comptime assert gamma_k.flat_rank == 1, "gamma_k must have rank 1"
    comptime simd_width = simd_width_of[DType.float32]()
    var contiguous_q = (
        q.layout.stride_coord().flatten()[1].value() == 1
        and gamma_q.layout.stride_coord().flatten()[0].value() == 1
        and q_out.layout.stride_coord().flatten()[1].value() == 1
    )
    var contiguous_k = (
        k.layout.stride_coord().flatten()[1].value() == 1
        and gamma_k.layout.stride_coord().flatten()[0].value() == 1
        and k_out.layout.stride_coord().flatten()[1].value() == 1
    )
    for r in range(rows):
        var rs_q = rsqrt(qk_var.load[width=1]((r, 0)) + epsilon)

        def normalize_q[width: Int](c: Int) {q, gamma_q, q_out, rs_q, r}:
            var xf = q.load[width=width]((r, c)).cast[.float32]()
            var g = gamma_q.load[width=width]((c,))
            q_out.store((r, c), ((xf * rs_q) * g).cast[out_dtype]())

        if contiguous_q:
            vectorize[simd_width](q_cols, normalize_q)
        else:
            vectorize[1](q_cols, normalize_q)

        var rs_k = rsqrt(qk_var.load[width=1]((r, 1)) + epsilon)

        def normalize_k[width: Int](c: Int) {k, gamma_k, k_out, rs_k, r}:
            var xf = k.load[width=width]((r, c)).cast[.float32]()
            var g = gamma_k.load[width=width]((c,))
            k_out.store((r, c), ((xf * rs_k) * g).cast[out_dtype]())

        if contiguous_k:
            vectorize[simd_width](k_cols, normalize_k)
        else:
            vectorize[1](k_cols, normalize_k)


def apply_qk_rms_norm[
    in_dtype: DType,
    out_dtype: DType,
    //,
    target: StaticString = "cpu",
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, .float32, ...],
    gamma_k: TileTensor[mut=False, .float32, ...],
    qk_var: TileTensor[mut=False, .float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Float32,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Fused per-element QK-RMSNorm apply for two operands Q and K.

    Given the already cross-rank-reduced per-row statistics `qk_var [M, 2]`
    (col 0 = mean(q^2), col 1 = mean(k^2), float32) and per-column float32
    scales `gamma_q [Nq]` / `gamma_k [Nk]`, applies in a single launch:

    `q_out[m,c] = cast((cast(q[m,c], f32) * rsqrt(qk_var[m,0] + eps)) * gamma_q[c], out_dtype)`
    and likewise for K with column 1. The grouping `((x * rs) * gamma)` then
    cast matches the unfused graph this replaces for bit-accuracy. This fuses
    the QK-RMSNorm apply chain (~7 tiny elementwise/View kernels) into one
    launch, used for cross-head QK-RMSNorm under tensor parallelism.

    All operands (`q` / `k` activations, the outputs `q_out` / `k_out`, and the
    `gamma_q` / `gamma_k` / `qk_var` inputs) are passed directly as
    `TileTensor`s and loaded/stored in-kernel, matching the in-file rms_norm
    `gamma.load[...]` idiom.

    Parameters:
        in_dtype: Element type of both activation inputs (`bfloat16` or
            `float32`).
        out_dtype: Element type of the outputs (typically equal to `in_dtype`).
        target: `"cpu"` or a GPU target string.

    Args:
        q_out: Scaled Q output, shape `[M, Nq]`.
        k_out: Scaled K output, shape `[M, Nk]`.
        gamma_q: Per-column float32 Q scales, shape `[Nq]`.
        gamma_k: Per-column float32 K scales, shape `[Nk]`.
        qk_var: Per-row float32 statistics, shape `[M, 2]` (col 0 = mean(q^2),
            col 1 = mean(k^2)).
        q: Q activations, shape `[M, Nq]`.
        k: K activations, shape `[M, Nk]`.
        epsilon: RMSNorm epsilon, added to the variance before `rsqrt`.
        rows: Shared leading dimension of Q and K.
        q_cols: Number of columns of Q.
        k_cols: Number of columns of K.
        ctx: Device context (ignored on CPU).
    """

    @inline(.always)
    def description_fn() {imm} -> String:
        return trace_arg("qk", Coord(rows, q_cols + k_cols), in_dtype)

    with Trace[TraceLevel.OP, target=target](
        "apply_qk_rms_norm",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        if rows == 0:
            return

        comptime if is_cpu[target]():
            apply_qk_rms_norm_cpu(
                q_out,
                k_out,
                gamma_q,
                gamma_k,
                qk_var,
                q,
                k,
                epsilon,
                rows,
                q_cols,
                k_cols,
            )
        elif is_gpu[target]():
            apply_qk_rms_norm_gpu(
                q_out,
                k_out,
                gamma_q,
                gamma_k,
                qk_var,
                q,
                k,
                epsilon,
                rows,
                q_cols,
                k_cols,
                ctx,
            )
        else:
            comptime assert False, "unsupported target " + target


def group_norm_reshape[
    dtype: DType,
    rank: Int,
](
    shape: Coord,
    buf: TileTensor[dtype, ...],
    channels_per_group: Int,
    spatial: Int,
    out result: TileTensor[
        dtype,
        Layout[
            shape_types=DynamicCoord[.int64, 2].element_types,
            stride_types=DynamicCoord[.int64, 2].element_types,
        ],
        buf.origin,
        address_space=buf.address_space,
    ],
):
    """
    Reshapes an input buffer for group normalization by flattening all
    dimensions except the group dimension. Returns a 2D buffer of shape
    (num_groups * N, group_size), where group_size is the product of
    channels_per_group and spatial.
    """
    comptime assert buf.rank == rank, "buf.rank must equal rank"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    var group_size = channels_per_group * spatial
    var prod_all_but_group_dim = Int(shape.product()) // group_size

    # `buf` is contiguous, so the flattened view is row-major over the same
    # storage: the group dim strides by one, the row dim by a whole group.
    comptime Shape2D = DynamicCoord[.int64, 2]
    result = {
        buf.ptr,
        Layout(
            Shape2D(Int64(prod_all_but_group_dim), Int64(group_size)),
            Shape2D(Int64(group_size), Int64(1)),
        ),
    }


@__name(t"group_norm_gpu_warp_tiling_{dtype}")
def group_norm_gpu_warp_tiling[
    LayoutType: TensorLayout,
    origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](row: Int, col: Int) -> SIMD[dtype, width],
    GammaFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
    BetaFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
](
    output: TileTensor[dtype, LayoutType, origin],
    epsilon: Float32,
    num_groups: Int32,
    channels_per_group: Int32,
    spatial: Int32,
    input_fn: InputFnType,
    gamma_fn: GammaFnType,
    beta_fn: BetaFnType,
):
    """Warp-tiling group_norm kernel.

    `input_fn`, `gamma_fn`, and `beta_fn` are trailing host-layout arguments
    so enqueue does not DevicePassable-encode those capturing closures.
    """
    var _num_groups = Int(num_groups)
    var _channels_per_group = Int(channels_per_group)
    var _spatial = Int(spatial)
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var idx = thread_idx.x * simd_width

    var vec_data = SIMD[accum_type, simd_width]()
    var group_size = _channels_per_group * _spatial

    var row = block_idx.x
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    var thread_mean = Scalar[accum_type]()
    var thread_m2 = Scalar[accum_type]()
    var thread_count = Scalar[accum_type]()

    with PDL():
        if idx + simd_width <= group_size:
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

            comptime for i in range(simd_width):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        if idx + simd_width <= group_size:
            var g = umod(row, _num_groups)
            var c_base = g * _channels_per_group
            var gamma_vec = SIMD[accum_type, simd_width]()
            var beta_vec = SIMD[accum_type, simd_width]()
            comptime for i in range(simd_width):
                var c = c_base + (idx + i) // _spatial
                gamma_vec[i] = gamma_fn[1](Coord(c))[0].cast[accum_type]()
                beta_vec[i] = beta_fn[1](Coord(c))[0].cast[accum_type]()
            var norm_val = (
                vec_data - row_mean
            ) * norm_factor * gamma_vec + beta_vec

            output.store[alignment=align]((row, idx), norm_val.cast[dtype]())


@__name(t"group_norm_gpu_block_{dtype}")
def group_norm_gpu_block[
    LayoutType: TensorLayout,
    origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](row: Int, col: Int) -> SIMD[dtype, width],
    GammaFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
    BetaFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
](
    output: TileTensor[dtype, LayoutType, origin],
    epsilon: Float32,
    num_groups: Int32,
    channels_per_group: Int32,
    spatial: Int32,
    input_fn: InputFnType,
    gamma_fn: GammaFnType,
    beta_fn: BetaFnType,
):
    """Block-per-row group_norm kernel.

    `input_fn`, `gamma_fn`, and `beta_fn` are trailing host-layout arguments
    so enqueue does not DevicePassable-encode those capturing closures.
    """
    var _num_groups = Int(num_groups)
    var _channels_per_group = Int(channels_per_group)
    var _spatial = Int(spatial)
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var group_size = _channels_per_group * _spatial

    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    with PDL():
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        for x in range(ceildiv(group_size // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < group_size:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                comptime for i in range(simd_width):
                    welford_update(
                        vec_data[i], thread_mean, thread_m2, thread_count
                    )

        welford_block_all_reduce(
            thread_mean,
            thread_m2,
            thread_count,
            row_mean,
            row_m2,
            row_count,
        )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        for x in range(ceildiv(group_size // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < group_size:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                var g = umod(row, _num_groups)
                var c_base = g * _channels_per_group

                var gamma_vec = SIMD[accum_type, simd_width]()
                var beta_vec = SIMD[accum_type, simd_width]()
                comptime for i in range(simd_width):
                    var c = c_base + (offset + i) // _spatial
                    gamma_vec[i] = gamma_fn[1](Coord(c))[0].cast[accum_type]()
                    beta_vec[i] = beta_fn[1](Coord(c))[0].cast[accum_type]()
                var norm_val = (
                    vec_data - row_mean
                ) * norm_factor * gamma_vec + beta_vec

                output.store[alignment=align](
                    (row, offset), norm_val.cast[dtype]()
                )


@__name(t"group_norm_gpu_multi_block_stats_{dtype}")
def group_norm_gpu_multi_block_stats[
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](row: Int, col: Int) -> SIMD[dtype, width],
](
    stats: TileTensor[get_accum_type[dtype](), StatsLayoutType, stats_origin],
    num_splits: Int32,
    group_size: Int32,
    input_fn: InputFnType,
):
    """Multi-block stats kernel: computes partial Welford statistics per split.

    Grid: num_rows * _num_splits blocks. Each block handles one split of one
    group and writes partial (mean, m2, count) to the stats buffer.
    Stats layout: stats[block_idx * 3 + {0,1,2}] = {mean, m2, count}.

    `input_fn` is a trailing host-layout argument so enqueue does not
    DevicePassable-encode that capturing closure.
    """
    var _num_splits = Int(num_splits)
    var _group_size = Int(group_size)
    comptime accum_type = get_accum_type[dtype]()

    var block_id = block_idx.x
    var row, split_id = divmod(block_id, _num_splits)
    var tid = thread_idx.x

    # Compute chunk boundaries (each split handles a contiguous chunk,
    # aligned to simd_width).
    var total_simd_elems = _group_size // simd_width
    var chunk_simd_size = ceildiv(total_simd_elems, _num_splits)
    var chunk_start = split_id * chunk_simd_size * simd_width
    var chunk_end = min(chunk_start + chunk_simd_size * simd_width, _group_size)
    var chunk_iters = ceildiv(chunk_simd_size, block_dim.x)

    with PDL():
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        for x in range(chunk_iters):
            var offset = (
                chunk_start + x * block_dim.x * simd_width + tid * simd_width
            )
            if offset < chunk_end:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()
                comptime for i in range(simd_width):
                    welford_update(
                        vec_data[i],
                        thread_mean,
                        thread_m2,
                        thread_count,
                    )

        var row_mean = Scalar[accum_type]()
        var row_m2 = Scalar[accum_type]()
        var row_count = Scalar[accum_type]()
        welford_block_all_reduce(
            thread_mean,
            thread_m2,
            thread_count,
            row_mean,
            row_m2,
            row_count,
        )

        # Thread 0 writes partial stats to the global stats buffer.
        if tid == 0:
            var base_idx = block_id * 3
            stats.store((base_idx,), row_mean)
            stats.store((base_idx + 1,), row_m2)
            stats.store((base_idx + 2,), row_count)


@__name(t"group_norm_gpu_multi_block_norm_{dtype}")
def group_norm_gpu_multi_block_norm[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](row: Int, col: Int) -> SIMD[dtype, width],
    GammaFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
    BetaFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Coord) -> SIMD[dtype, width],
](
    output: TileTensor[dtype, OutputLayoutType, output_origin],
    stats: TileTensor[get_accum_type[dtype](), StatsLayoutType, stats_origin],
    epsilon: Float32,
    num_groups: Int32,
    channels_per_group: Int32,
    spatial: Int32,
    num_splits: Int32,
    group_size: Int32,
    input_fn: InputFnType,
    gamma_fn: GammaFnType,
    beta_fn: BetaFnType,
):
    """Multi-block normalize kernel: reduces partial stats and normalizes.

    Grid: num_rows * num_splits blocks. Each block reads all partial stats
    for its group, reduces to final mean/variance, then normalizes its
    chunk of elements.

    `input_fn`, `gamma_fn`, and `beta_fn` are trailing host-layout arguments
    so enqueue does not DevicePassable-encode those capturing closures.
    """
    var _num_groups = Int(num_groups)
    var _channels_per_group = Int(channels_per_group)
    var _spatial = Int(spatial)
    var _num_splits = Int(num_splits)
    var _group_size = Int(group_size)
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var block_id = block_idx.x
    var row, split_id = divmod(block_id, _num_splits)
    var tid = thread_idx.x

    # Same chunk boundaries as stats kernel.
    var total_simd_elems = _group_size // simd_width
    var chunk_simd_size = ceildiv(total_simd_elems, _num_splits)
    var chunk_start = split_id * chunk_simd_size * simd_width
    var chunk_end = min(chunk_start + chunk_simd_size * simd_width, _group_size)
    var chunk_iters = ceildiv(chunk_simd_size, block_dim.x)

    with PDL():
        # Reduce all partial stats for this group (_num_splits is small,
        # typically 4-16, so this loop is cheap).
        var row_mean = Scalar[accum_type]()
        var row_m2 = Scalar[accum_type]()
        var row_count = Scalar[accum_type]()
        var stats_row_base = row * _num_splits * 3
        for s in range(_num_splits):
            var base_idx = stats_row_base + s * 3
            welford_combine(
                stats[base_idx],
                stats[base_idx + 1],
                stats[base_idx + 2],
                row_mean,
                row_m2,
                row_count,
            )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        var g = row % _num_groups
        var c_base = g * _channels_per_group

        for x in range(chunk_iters):
            var offset = (
                chunk_start + x * block_dim.x * simd_width + tid * simd_width
            )
            if offset < chunk_end:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                var norm_val: SIMD[accum_type, simd_width]

                # Vectorized gamma/beta: when all SIMD elements share the
                # same channel (common case for large _spatial dims), load
                # gamma/beta once and broadcast.
                var c_rel, in_channel = divmod(offset, _spatial)
                var c_first = c_base + c_rel
                if in_channel + simd_width <= _spatial:
                    var gamma_val = gamma_fn[1](Coord(c_first)).cast[
                        accum_type
                    ]()
                    var beta_val = beta_fn[1](Coord(c_first)).cast[accum_type]()
                    norm_val = (
                        vec_data - row_mean
                    ) * norm_factor * gamma_val + beta_val
                else:
                    var gamma_vec = SIMD[accum_type, simd_width]()
                    var beta_vec = SIMD[accum_type, simd_width]()
                    comptime for i in range(simd_width):
                        var c = c_base + (offset + i) // _spatial
                        gamma_vec[i] = gamma_fn[1](Coord(c))[0].cast[
                            accum_type
                        ]()
                        beta_vec[i] = beta_fn[1](Coord(c))[0].cast[accum_type]()
                    norm_val = (
                        vec_data - row_mean
                    ) * norm_factor * gamma_vec + beta_vec

                output.store[alignment=align](
                    (row, offset), norm_val.cast[dtype]()
                )


def group_norm_gpu[
    dtype: DType,
    rank: Int,
    //,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    GammaFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    BetaFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
](
    input_fn: InputFn,
    gamma_fn: GammaFn,
    beta_fn: BetaFn,
    shape: Coord,
    epsilon: Float32,
    output: TileTensor[mut=True, dtype, ...],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime accum_type = get_accum_type[dtype]()
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"

    var N = Int(shape[0])
    var C = Int(shape[1])

    var spatial = Int(shape.product()) // (N * C)

    # The device closure reads only the innermost dim; a typed `Coord` capture
    # fails the launch (CUDA_ERROR_INVALID_VALUE), so unwrap it here and let the
    # closure capture plain `Int`s.
    var last_dim = Int(shape[rank - 1])
    var channels_per_group = C // num_groups

    var output_rs = group_norm_reshape[dtype, rank](
        shape, output, channels_per_group, spatial
    )

    comptime OutputLinearIdxType = Scalar[output_rs.linear_idx_type]

    var num_rows = output_rs.dim[0]()
    var num_cols = output_rs.dim[1]()

    # Zero-sized input (e.g. a ``(B, C, 0, 0)`` tensor flowing through a
    # diffusion VAE encoder for the text-to-image placeholder): nothing
    # to normalize.  The output buffer is pre-allocated zero-element by
    # the caller and the kernel's ``num_cols < simd_width`` misalignment
    # check below would otherwise abort.  Early-return is correct because
    # mean/var of an empty group has no defined value and the downstream
    # readers also have zero spatial dims.
    if num_rows == OutputLinearIdxType(0) or num_cols == OutputLinearIdxType(0):
        return

    @inline(.always)
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {
        var spatial,
        var last_dim,
        var num_groups,
        var channels_per_group,
        var input_fn,
    } -> SIMD[dtype, simd_width]:
        var n, g = divmod(row, num_groups)
        var c = g * channels_per_group

        comptime if rank == 4:
            var inner_volume = spatial
            var c_offset, hw = divmod(col, inner_volume)
            c += c_offset
            var h, w = divmod(hw, last_dim)

            # Guard against c_offset boundary straddling.  A view-fused
            # NHWC→NCHW transpose generates a strided_load(stride=C) for
            # the W-dimension.  That load is correct within a single c_offset
            # region (consecutive-w with stride C maps to adjacent NHWC
            # addresses), but it reads wrong elements when the simd_width
            # window crosses a c_offset boundary at a multiple of
            # inner_volume=H*W.  This happens when H*W % simd_width != 0 and
            # the thread's starting column lands near a boundary.  Fall back
            # to element-wise scalar loads in that case so each element's
            # full (n, c, h, w) index is recomputed independently.
            if hw + simd_width > inner_volume or w + simd_width > last_dim:
                var result = SIMD[dtype, simd_width]()
                for i in range(simd_width):
                    var cur_col = col + i
                    var c_off, hw_i = divmod(cur_col, inner_volume)
                    var h_i, w_i = divmod(hw_i, last_dim)
                    result[i] = input_fn[1](
                        Coord(n, g * channels_per_group + c_off, h_i, w_i)
                    )[0]
                return result

            return input_fn[simd_width](Coord(n, c, h, w))

        elif rank == 3:
            var inner_volume = spatial
            var c_offset, l = divmod(col, inner_volume)
            c += c_offset

            if l + simd_width > inner_volume:
                var result = SIMD[dtype, simd_width]()
                for i in range(simd_width):
                    var cur_col = col + i
                    var c_off, l_i = divmod(cur_col, inner_volume)
                    result[i] = input_fn[1](
                        Coord(n, g * channels_per_group + c_off, l_i)
                    )[0]
                return result

            return input_fn[simd_width](Coord(n, c, l))

        else:
            comptime assert False, "group_norm_gpu requires a rank of 3 or 4"

    comptime simd_width = simd_width_of[dtype, target=ctx.target]()
    if num_cols < OutputLinearIdxType(simd_width):
        raise Error(
            "group_norm_gpu requires num_cols >= simd_width; got num_cols="
            + String(num_cols)
            + " and simd_width="
            + String(simd_width)
        )

    comptime warp_size = ctx.default_device_info.warp_size
    comptime max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // warp_size
    )

    var grid_dim = num_rows
    var block_dim = min(
        ceildiv(
            ceildiv(num_cols, OutputLinearIdxType(simd_width)),
            OutputLinearIdxType(warp_size),
        )
        * OutputLinearIdxType(warp_size),
        OutputLinearIdxType(warp_size * max_warps_per_block),
    )

    if num_cols % OutputLinearIdxType(simd_width) == 0:
        # When the number of columns is small enough that they can be placed in
        # registers, we do warp tiling, which is a single pass to do mean/var
        # computation and normalization.
        if num_cols <= OutputLinearIdxType(
            warp_size * simd_width * max_warps_per_block
        ):
            comptime kernel = group_norm_gpu_warp_tiling[
                LayoutType=output_rs.LayoutType,
                origin=output_rs.origin,
                dtype=dtype,
                simd_width=simd_width,
                InputFnType=type_of(input_fn_2d),
                GammaFnType=GammaFn,
                BetaFnType=BetaFn,
            ]
            ctx.enqueue_function[kernel](
                output_rs,
                epsilon.cast[.float32](),
                Int32(num_groups),
                Int32(channels_per_group),
                Int32(spatial),
                host_args=(input_fn_2d, gamma_fn, beta_fn),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        else:
            # Use multi-block reduction when the grid is too small for
            # good GPU occupancy.  Each group is split across num_splits
            # blocks so that more SMs are active.
            comptime desired_min_grid = 256
            var num_splits = 1
            if Int(num_rows) < desired_min_grid:
                num_splits = min(ceildiv(desired_min_grid, Int(num_rows)), 32)
                # Ensure each split has enough work (≥ 1 SIMD iter per
                # thread at block_dim threads).
                var group_size = Int(num_cols)
                var max_useful_splits = max(
                    1,
                    group_size
                    // (Int(simd_width) * warp_size * max_warps_per_block),
                )
                num_splits = min(num_splits, max_useful_splits)

            if num_splits > 1:
                var group_size = Int(num_cols)

                # Allocate a small buffer for partial Welford statistics:
                # 3 values (mean, m2, count) per (row, split).
                var stats_size = Int(num_rows) * num_splits * 3
                var stats_buf = ctx.enqueue_create_buffer[accum_type](
                    stats_size
                )
                var stats = TileTensor(
                    stats_buf,
                    row_major(stats_size),
                )

                # Compute block_dim based on per-split chunk size.
                # Cap at 256 threads: both kernels capture closures
                # (input_fn_2d with its coordinate computation chain,
                # gamma_fn, beta_fn) that cause high register pressure,
                # especially for bfloat16 (simd_width=8).  256 threads
                # keeps total register usage within GPU limits while
                # each thread processes more elements per iteration.
                comptime mb_max_block_dim = min(
                    256, warp_size * max_warps_per_block
                )
                var total_simd_elems = group_size // simd_width
                var chunk_simd_size = ceildiv(total_simd_elems, num_splits)
                var mb_block_dim = min(
                    align_up(chunk_simd_size, warp_size),
                    mb_max_block_dim,
                )
                var mb_grid_dim = Int(num_rows) * num_splits

                # Kernel 1: compute partial Welford stats per split.
                comptime stats_kernel = group_norm_gpu_multi_block_stats[
                    StatsLayoutType=stats.LayoutType,
                    stats_origin=stats.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    InputFnType=type_of(input_fn_2d),
                ]
                ctx.enqueue_function[stats_kernel](
                    stats,
                    Int32(num_splits),
                    Int32(group_size),
                    host_args=(input_fn_2d,),
                    grid_dim=mb_grid_dim,
                    block_dim=mb_block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )

                # Kernel 2: reduce stats and normalize each chunk.
                comptime norm_kernel = group_norm_gpu_multi_block_norm[
                    OutputLayoutType=output_rs.LayoutType,
                    output_origin=output_rs.origin,
                    StatsLayoutType=stats.LayoutType,
                    stats_origin=stats.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    InputFnType=type_of(input_fn_2d),
                    GammaFnType=GammaFn,
                    BetaFnType=BetaFn,
                ]
                ctx.enqueue_function[norm_kernel](
                    output_rs,
                    stats,
                    epsilon.cast[.float32](),
                    Int32(num_groups),
                    Int32(channels_per_group),
                    Int32(spatial),
                    Int32(num_splits),
                    Int32(group_size),
                    host_args=(input_fn_2d, gamma_fn, beta_fn),
                    grid_dim=mb_grid_dim,
                    block_dim=mb_block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )

                _ = stats_buf^
            else:
                comptime kernel = group_norm_gpu_block[
                    LayoutType=output_rs.LayoutType,
                    origin=output_rs.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    InputFnType=type_of(input_fn_2d),
                    GammaFnType=GammaFn,
                    BetaFnType=BetaFn,
                ]
                ctx.enqueue_function[kernel](
                    output_rs,
                    epsilon.cast[.float32](),
                    Int32(num_groups),
                    Int32(channels_per_group),
                    Int32(spatial),
                    host_args=(input_fn_2d, gamma_fn, beta_fn),
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )
    else:
        comptime kernel = group_norm_gpu_block[
            LayoutType=output_rs.LayoutType,
            origin=output_rs.origin,
            dtype=dtype,
            simd_width=1,
            InputFnType=type_of(input_fn_2d),
            GammaFnType=GammaFn,
            BetaFnType=BetaFn,
        ]
        ctx.enqueue_function[kernel](
            output_rs,
            epsilon.cast[.float32](),
            Int32(num_groups),
            Int32(channels_per_group),
            Int32(spatial),
            host_args=(input_fn_2d, gamma_fn, beta_fn),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


def group_norm_cpu[
    dtype: DType,
    rank: Int,
    //,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    GammaFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    BetaFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
](
    input_fn: InputFn,
    gamma_fn: GammaFn,
    beta_fn: BetaFn,
    shape: Coord,
    epsilon: Float32,
    output: TileTensor[mut=True, dtype, ...],
    num_groups: Int,
    ctx: Optional[DeviceContext.CPU] = None,
) raises:
    """Computes group normalization on CPU.

    Reduces a single-pass Welford mean/variance over each `(batch, group)`
    block of `channels_per_group * spatial` elements, then applies the
    per-channel `gamma`/`beta` affine transform. Parallelizes across
    `N * num_groups` blocks.

    Parameters:
        dtype: Element type of the input and output tensors.
        rank: Tensor rank of the input and output tensors (3 or 4).
        InputFn: Type of the input load closure.
        GammaFn: Type of the gamma load closure.
        BetaFn: Type of the beta load closure.

    Args:
        input_fn: Function called to generate an input value.
        gamma_fn: Function called to generate a gamma value.
        beta_fn: Function called to generate a beta value.
        shape: The shape of the input/output tensor.
        epsilon: Small constant for numerical stability.
        output: Output tensor receiving the normalized result.
        num_groups: Number of groups the channel dimension is split into.
        ctx: Optional device context used to size the CPU thread pool.
    """
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime accum_type = get_accum_type[dtype]()
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"

    var N = Int(shape[0])
    var C = Int(shape[1])
    var spatial = Int(shape.product()) // (N * C)
    var channels_per_group = C // num_groups
    var group_size = channels_per_group * spatial
    var num_rows = N * num_groups

    if num_rows == 0 or group_size == 0:
        return

    var num_workers = min(parallelism_level(ctx), num_rows)
    var chunk_size = ceildiv(num_rows, num_workers)

    def task_func(
        thread_id: Int,
    ) raises {
        var shape,
        var num_groups,
        var channels_per_group,
        var spatial,
        var epsilon,
        imm,
    }:
        var row_start = thread_id * chunk_size
        var row_end = min(row_start + chunk_size, num_rows)

        for row in range(row_start, row_end):
            var n, g = divmod(row, num_groups)
            var c_base = g * channels_per_group

            @inline(.always)
            def indices_for(
                col: Int,
            ) {var shape, var n, var c_base, var spatial} -> DynamicCoord[
                .int64, rank
            ]:
                var c_offset, s = divmod(col, spatial)
                comptime if rank == 4:
                    var h, w = divmod(s, Int(shape[3]))
                    return rebind[DynamicCoord[.int64, rank]](
                        Coord(
                            Int64(n),
                            Int64(c_base + c_offset),
                            Int64(h),
                            Int64(w),
                        )
                    )
                else:
                    return rebind[DynamicCoord[.int64, rank]](
                        Coord(Int64(n), Int64(c_base + c_offset), Int64(s))
                    )

            # Single-pass Welford mean/variance over the group.
            var mean = Scalar[accum_type]()
            var m2 = Scalar[accum_type]()
            var count = Scalar[accum_type]()
            for col in range(group_size):
                var val = input_fn[1](indices_for(col))[0].cast[accum_type]()
                welford_update(val, mean, m2, count)

            var norm_factor = rsqrt(m2 / count + epsilon.cast[accum_type]())

            for col in range(group_size):
                var idx = indices_for(col)
                var val = input_fn[1](idx)[0].cast[accum_type]()
                var gamma_val = gamma_fn[1](Coord(idx[1]))[0].cast[accum_type]()
                var beta_val = beta_fn[1](Coord(idx[1]))[0].cast[accum_type]()
                var norm_val = (val - mean) * norm_factor * gamma_val + beta_val
                output.store(idx, norm_val.cast[dtype]())

    sync_parallelize(task_func, num_workers, ctx)


@inline(.always)
def group_norm[
    dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    GammaFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    BetaFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int](Coord) -> SIMD[dtype, width]),
    /,
    target: StaticString = "cpu",
](
    input_fn: InputFn,
    gamma_fn: GammaFn,
    beta_fn: BetaFn,
    shape: Coord,
    epsilon: Float32,
    groups: Int32,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime assert (
        rank > 2 and rank < 5
    ), "group_norm requires input rank of 3 or 4"

    if shape != output.layout.shape_coord():
        raise Error(
            "Input/output shape mismatch: input = {shape}, output ="
            " {output.dynamic_shape}"
        )

    var num_groups: Int = Int(groups[0])

    var C = Int(shape[1])
    if C % num_groups != 0:
        raise Error(
            "Invalid num_groups: channels (C = {C}) must be divisible by"
            " num_groups = {num_groups}"
        )

    @inline(.always)
    def description_fn() {imm} -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "group_norm",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        comptime if is_cpu[target]():
            group_norm_cpu[dtype=dtype, rank=rank](
                input_fn,
                gamma_fn,
                beta_fn,
                shape,
                epsilon,
                output,
                num_groups,
                Optional(ctx.assert_cpu_target()),
            )
        elif is_gpu[target]():
            group_norm_gpu[dtype=dtype, rank=rank](
                input_fn,
                gamma_fn,
                beta_fn,
                shape,
                epsilon,
                output,
                num_groups,
                ctx=ctx,
            )
        else:
            comptime assert False, "unsupported target " + target


# ===----------------------------------------------------------------------=== #
# RMSNorm + group-scaled dynamic FP8 quantize (blockwise-FP8 activations)
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _RMSNormQuantizeFp8GroupKernel[
    out_dtype: DType,
    scales_dtype: DType,
    output_layout: TensorLayout,
    output_origin: MutOrigin,
    output_engine: TensorEngine,
    output_idx_type: DType,
    scales_layout: TensorLayout,
    scales_origin: MutOrigin,
    scales_engine: TensorEngine,
    scales_idx_type: DType,
    gamma_layout: TensorLayout,
    gamma_origin: ImmOrigin,
    gamma_engine: TensorEngine,
    gamma_idx_type: DType,
    //,
    in_dtype: DType,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: Int, alignment: Int
        ](row: Int, col: Int) -> SIMD[in_dtype, width]
    ),
    num_cols: Int,
    group_size: Int,
    simd_width: Int,
    multiply_before_cast: Bool,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """RMS-normalizes one row per block and quantizes it to FP8 with one
    dynamic scale per `group_size` columns (the blockwise-FP8 activation
    layout; scales stored `[num_cols // group_size, rows]`).

    Every thread owns one `simd_width`-wide vector of the row, so the row
    stays register-resident across the sum-of-squares block reduction and
    the group max, which is a `lane_group_max` over the
    `group_size // simd_width` lanes tiling the group (an aligned lane
    segment of one warp). The normalized value is formed and rounded to
    `in_dtype` exactly as `rms_norm` writes it for `multiply_before_cast`, so
    the quantized bytes match the unfused `rms_norm` +
    `quantize_dynamic_scaled_fp8` pair up to the sum-of-squares reduction
    order.
    """

    var input_fn: Self.InputFnType
    var output: TileTensor[
        mut=True,
        Self.out_dtype,
        Self.output_layout,
        Self.output_origin,
        Engine=Self.output_engine,
        linear_idx_type=Self.output_idx_type,
    ]
    var scales: TileTensor[
        mut=True,
        Self.scales_dtype,
        Self.scales_layout,
        Self.scales_origin,
        Engine=Self.scales_engine,
        linear_idx_type=Self.scales_idx_type,
    ]
    var gamma: TileTensor[
        mut=False,
        Self.in_dtype,
        Self.gamma_layout,
        Self.gamma_origin,
        Engine=Self.gamma_engine,
        linear_idx_type=Self.gamma_idx_type,
    ]
    var epsilon: Scalar[Self.in_dtype]
    var weight_offset: Scalar[Self.in_dtype]
    var scale_ub: Scalar[Self.scales_dtype]

    comptime num_threads = Self.num_cols // Self.simd_width
    comptime lanes_per_group = Self.group_size // Self.simd_width

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    def __call__(self):
        comptime accum = get_accum_type[Self.in_dtype]()
        comptime assert (
            Self.num_cols % Self.group_size == 0
        ), "group_size must tile the row"
        comptime assert (
            Self.group_size % Self.simd_width == 0
        ), "simd_width must tile group_size"
        comptime assert (
            Self.lanes_per_group.is_power_of_two()
            and Self.lanes_per_group <= WARP_SIZE
        ), "a scale group must be a power-of-two lane segment of one warp"
        comptime assert (
            Self.num_threads % WARP_SIZE == 0
        ), "the row must tile whole warps"
        comptime align = align_of[SIMD[Self.in_dtype, Self.simd_width]]()

        var input_fn = self.input_fn
        var output = self.output
        var scales = self.scales
        var gamma = self.gamma

        var tid = thread_idx.x
        var row = block_idx.x
        var col = tid * Self.simd_width
        var group_idx = tid // Self.lanes_per_group
        var lane_in_group = tid % Self.lanes_per_group

        with PDL():
            var x = input_fn.__call__[
                width=Self.simd_width, alignment=Self.simd_width
            ](row, col).cast[accum]()
            var ssq = block.sum[block_size=Self.num_threads, broadcast=True](
                (x * x).reduce_add()
            )
            var inv_rms = rsqrt(
                ssq / Scalar[accum](Self.num_cols) + self.epsilon.cast[accum]()
            )
            var gamma_val = gamma.load[width=Self.simd_width, alignment=align](
                (col,)
            )
            # The pair quantizes the bf16 row `rms_norm` wrote, so round the
            # normalized value to `in_dtype` exactly where `rms_norm` does.
            var normed_in: SIMD[Self.in_dtype, Self.simd_width]
            comptime if Self.multiply_before_cast:
                normed_in = (
                    (x * inv_rms)
                    * (
                        gamma_val.cast[accum]()
                        + self.weight_offset.cast[accum]()
                    )
                ).cast[Self.in_dtype]()
            else:
                normed_in = (x * inv_rms).cast[Self.in_dtype]() * (
                    gamma_val + self.weight_offset
                )
            var normed = normed_in.cast[accum]()

            var group_max: Scalar[accum]
            comptime if Self.lanes_per_group >= 2:
                group_max = warp.lane_group_max[num_lanes=Self.lanes_per_group](
                    abs(normed).reduce_max()
                )
            else:
                group_max = abs(normed).reduce_max()

            var scale_factor: Scalar[Self.scales_dtype]
            var scale_recip: Scalar[accum]
            scale_factor, scale_recip = compute_dynamic_fp8_scale[
                Self.out_dtype
            ](group_max, self.scale_ub)

            if lane_in_group == 0:
                scales.store(Coord(group_idx, row), scale_factor)
            output.store(
                Coord(row, col),
                fp8_quantize[Self.out_dtype](normed, scale_recip),
            )


def _rms_norm_quantize_lane_width[group_size: Int, num_cols: Int]() -> Int:
    """Returns the widest lane (elements per thread) the fused kernel can use,
    or 0 when none fits.

    The row must tile whole warps, and a scale group must span a power-of-two
    lane segment that fits in one warp. Trying widths from widest to narrowest
    keeps every shape that already compiles on its current width; the narrower
    candidates only matter where a 32-lane warp fits and a 64-lane wavefront
    does not (256 columns at group 128 needs width 4 on CDNA).
    """
    comptime for width in [16, 8, 4, 2, 1]:
        comptime lanes_per_group = group_size // width
        comptime if (
            group_size % width == 0
            and (num_cols // width) % WARP_SIZE == 0
            and lanes_per_group.is_power_of_two()
            and lanes_per_group <= WARP_SIZE
        ):
            return width
    return 0


def rms_norm_quantize_dynamic_scaled_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: Int, alignment: Int
        ](row: Int, col: Int) -> SIMD[in_dtype, width]
    ),
    //,
    group_size: Int,
    num_cols: Int,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    input_fn: InputFnType,
    output: TileTensor[mut=True, out_dtype, ...],
    scales: TileTensor[mut=True, scales_dtype, ...],
    gamma: TileTensor[mut=False, in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    ctx: DeviceContext,
    num_rows: Int,
) raises:
    """Fused `rms_norm` + `quantize_dynamic_scaled_fp8` for group (blockwise)
    activation scales: one launch, no normalized round trip through HBM.

    Produces the outputs of `rms_norm[multiply_before_cast=...]` followed by
    `quantize_dynamic_scaled_fp8[group_size_or_per_token=group_size]`:
    `output` is `[rows, num_cols]` FP8 and `scales` is
    `[num_cols // group_size, rows]` (row `r`'s scale for column group `g`
    at `scales[g, r]`; a padded trailing scale column is left untouched).
    Per-token scales (`group_size == num_cols`) stay on
    `rms_norm_fused_quantize_dynamic_scaled_fp8`.
    """
    comptime assert output.rank == 2, "expected rank-2 output"
    comptime assert scales.rank == 2, "expected rank-2 scales"
    comptime assert gamma.rank == 1, "gamma must have rank 1"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"
    comptime assert scales_dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "scales dtype should be bfloat16, float16 or float32"
    comptime assert (
        group_size > 0 and group_size < num_cols and num_cols % group_size == 0
    ), "group_size must be a proper divisor of the row width"

    # Widest lane that still tiles the row into whole warps. A 64-lane CDNA
    # wavefront can need a narrower lane than a 32-lane warp at the same shape.
    comptime simd_width = _rms_norm_quantize_lane_width[group_size, num_cols]()
    comptime assert simd_width > 0, (
        "no lane width tiles the row into whole warps with a scale group per"
        " warp segment"
    )
    comptime num_threads = num_cols // simd_width
    comptime assert (
        group_size % simd_width == 0
    ), "group_size must be a multiple of the lane width"
    comptime assert (
        num_threads % WARP_SIZE == 0
    ), "the row must tile whole warps at the lane width"
    comptime assert (
        num_threads <= ctx.default_device_info.max_thread_block_size
    ), "row too wide for a single block"
    comptime assert (
        group_size // simd_width <= WARP_SIZE
    ), "a scale group must fit in one warp"

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "rms_norm_quantize_dynamic_scaled_fp8",
        task_id=Int(ctx.id()),
    ):
        if num_rows == 0:
            return

        var kernel = _RMSNormQuantizeFp8GroupKernel[
            num_cols=num_cols,
            group_size=group_size,
            simd_width=simd_width,
            multiply_before_cast=multiply_before_cast,
        ](
            input_fn,
            output.address_space_cast[.GENERIC](),
            scales.address_space_cast[.GENERIC](),
            gamma.address_space_cast[.GENERIC](),
            epsilon,
            weight_offset,
            scale_ub.cast[scales_dtype](),
        )
        ctx.enqueue_function(
            kernel,
            grid_dim=(num_rows, 1, 1),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(pdl_level),
        )
