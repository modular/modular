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

from std.math import sqrt
from std.sys import align_of
from std.random import rand

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import *
from builtin_kernels.rowwise_normalization import rms_norm
from std.testing import assert_almost_equal, assert_equal
from std.utils.coord import ComptimeInt

from std.utils.index import Index, IndexList


def compute_rms[
    dtype: DType
](data: TileTensor[dtype, ...], size: Int, eps: Float32) -> Scalar[dtype]:
    comptime assert data.flat_rank == 1, "data.rank must be 1"
    comptime assert data.element_size == 1

    comptime accum_type = get_accum_type[dtype]()
    var sum_of_squares = Scalar[accum_type]()
    for i in range(size):
        var val = data.load[width=1]((i,)).cast[accum_type]()
        sum_of_squares += val * val
    var result = sqrt(
        (sum_of_squares / Scalar[accum_type](data.num_elements()))
        + eps.cast[accum_type]()
    )
    return result.cast[dtype]()


def run_rms_norm_gpu[
    rank: Int,
    //,
    dtype: DType,
    *,
    multiply_before_cast: Bool = True,
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_rms_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)

    rand[dtype](data_h.as_span())

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    comptime epsilon = Float32(0.001)
    comptime weight_offset = Float32(0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @inline(.always)
    def input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        return data_buf.load[width=width](coords)

    @inline(.always)
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var data_buf} -> None:
        data_buf.store[alignment=alignment](coords, val)

    # `rms_norm_gpu` migrated to a `Coord` shape boundary (softmax PR #88203);
    # `rank` is now an explicit parameter (no longer inferred from `shape`).
    rms_norm_gpu[
        rank,
        epsilon=epsilon,
        weight_offset=weight_offset,
        multiply_before_cast=multiply_before_cast,
    ](input_fn, identity_output_fn, Coord(shape), gamma, ctx)
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr().unsafe_offset(r * cols),
            row_major(cols),
        )
        var rms_ref = compute_rms(vec, cols, epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = (data_h[idx] / rms_ref) * (
                gamma_h[c] + weight_offset.cast[dtype]()
            )
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d


def run_rms_norm_gpu_static_dims[
    dtype: DType, *, multiply_before_cast: Bool = True
](ctx: DeviceContext, var shape: Coord, rtol: Float64 = 0.01) raises:
    """Drives `rms_norm_gpu` with a shape whose inner dims are `ComptimeInt`.

    The dynamic-only shapes above (`Coord(IndexList)`) leave every leaf
    runtime, but production reaches this kernel through
    `TileTensor.shape_coord()`, which preserves static dims. Static inner dims
    make `row_major(shape)`'s stride list all-`ComptimeInt` and therefore
    zero-sized, which collapses the `Layout` capture carrier and the
    `_RowSpec` kernel argument back to a bare `Coord` -- the MOCO-4307 arity
    mismatch both types exist to avoid. Only a static-dim shape exercises
    that collapse.
    """
    print("== run_rms_norm_gpu_static_dims")

    comptime rank = type_of(shape).rank
    var cols = Int(shape[rank - 1])
    var rows = Int(shape.product()) // cols

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)

    rand[dtype](data_h.as_span())

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var data_buf = TileTensor(data_d, row_major(shape))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))
    comptime epsilon = Float32(0.001)
    comptime weight_offset = Float32(0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @inline(.always)
    def input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        return data_buf.load[width=width](coords)

    @inline(.always)
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var data_buf} -> None:
        data_buf.store[alignment=alignment](coords, val)

    rms_norm_gpu[
        rank,
        epsilon=epsilon,
        weight_offset=weight_offset,
        multiply_before_cast=multiply_before_cast,
    ](input_fn, identity_output_fn, shape, gamma, ctx)
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr().unsafe_offset(r * cols),
            row_major(cols),
        )
        var rms_ref = compute_rms(vec, cols, epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = (data_h[idx] / rms_ref) * (
                gamma_h[c] + weight_offset.cast[dtype]()
            )
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d


def run_rms_norm_gpu_zero_rows[dtype: DType](ctx: DeviceContext) raises:
    """A zero-row shape must be a no-op, not a rejected zero-sized launch.

    A rank owns an empty shard when rows < TP group size (bs=1 decode at TP4).
    """
    print("== run_rms_norm_gpu_zero_rows")

    comptime cols = 128
    comptime sentinel = Scalar[dtype](-7.0)

    var data_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    for i in range(cols):
        data_h[i] = sentinel
        gamma_h[i] = Scalar[dtype](1.0)

    var data_d = ctx.enqueue_create_buffer[dtype](cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    var data_buf = TileTensor(data_d, row_major(Coord(Index(0, cols))))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))

    @inline(.always)
    def input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        return data_buf.load[width=width](coords)

    @inline(.always)
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var data_buf} -> None:
        data_buf.store[alignment=alignment](coords, val)

    rms_norm_gpu[
        2,
        epsilon=Float32(0.001),
        weight_offset=Float32(0.0),
        multiply_before_cast=True,
    ](input_fn, identity_output_fn, Coord(Index(0, cols)), gamma, ctx)
    var res = ctx.enqueue_create_host_buffer[dtype](cols)
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for i in range(cols):
        assert_almost_equal(res[i], sentinel)

    _ = data_d
    _ = gamma_d


def run_rms_norm_row_based[
    dtype: DType,
    rank: Int,
    *,
    multiply_before_cast: Bool = True,
    static_cols: Int = 0,
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    """Drives the row-based `rms_norm` entry point, not `rms_norm_gpu`.

    `mo.reduce.rms_norm` lowers through this overload, so its epilogue-closure
    contract needs coverage of its own; the `run_rms_norm_gpu` cases above
    exercise a different function. A positive `static_cols` passes the row
    width as a `ComptimeInt`, as the graph compiler does for a static hidden
    size; that is the register-cached path.
    """
    print("== run_rms_norm_row_based")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    comptime eps_f32 = Float32(0.001)

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)

    rand[dtype](data_h.as_span())

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    # Distinct output buffer: input_fn (reads) and output_fn (writes) are
    # separate value-closure args, so they must reference distinct buffer
    # origins (writing in place into `data_d` would alias the read).
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    comptime weight_offset = Float32(0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        # The closure's `alignment` counts elements, as the graph compiler's
        # closures read it; `load` / `store` take bytes.
        return data_buf.load[
            width=width, alignment=alignment * align_of[dtype]()
        ](coords)

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_buf}:
        out_buf.store[alignment=alignment * align_of[dtype]()](coords, val)

    comptime if static_cols > 0:
        assert_equal(cols, static_cols)
        rms_norm[
            dtype,
            rank,
            target="gpu",
            epsilon=eps_f32,
            weight_offset=weight_offset,
            multiply_before_cast=multiply_before_cast,
        ](
            input_fn,
            output_fn,
            Coord(shape),
            ComptimeInt[static_cols](),
            gamma,
            ctx,
        )
    else:
        rms_norm[
            dtype,
            rank,
            target="gpu",
            epsilon=eps_f32,
            weight_offset=weight_offset,
            multiply_before_cast=multiply_before_cast,
        ](
            input_fn,
            output_fn,
            Coord(shape),
            Scalar[DType.int](cols),
            gamma,
            ctx,
        )
    ctx.enqueue_copy(res, out_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr().unsafe_offset(r * cols),
            row_major(cols),
        )
        var rms_ref = compute_rms(vec, cols, eps_f32)
        for c in range(cols):
            var idx = r * cols + c
            var val = (data_h[idx] / rms_ref) * (
                gamma_h[c] + weight_offset.cast[dtype]()
            )
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = out_d
    _ = gamma_d


def main() raises:
    with DeviceContext() as ctx:
        run_rms_norm_gpu_zero_rows[.bfloat16](ctx)
        run_rms_norm_gpu_zero_rows[.float32](ctx)
        run_rms_norm_gpu[.float32](ctx, Index(5))
        run_rms_norm_gpu[.float32](ctx, Index(3, 4, 10, 20, 8))
        run_rms_norm_gpu[.float32](ctx, Index(1, 5, 6, 10, 128))
        run_rms_norm_gpu[.float32](ctx, Index(2, 5))
        run_rms_norm_gpu[.float32](ctx, Index(2, 55))
        run_rms_norm_gpu[.float32](ctx, Index(7, 557))
        run_rms_norm_gpu[.float32](ctx, Index(2, 8191))
        run_rms_norm_gpu[.float32](ctx, Index(2, 8192))
        run_rms_norm_gpu[.float32](ctx, Index(2, 16384))
        run_rms_norm_gpu[.float32](ctx, Index(2, 16385))
        run_rms_norm_gpu[.bfloat16](ctx, Index(3000, 32, 128), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(2999, 31, 128), rtol=2e-2)

        # Rank-3 `[B, S, H]` cases (added with the `IndexList`->`Coord` boundary
        # migration). These drive a non-trivial multi-axis
        # `_get_start_indices_of_nth_subvolume` per-row translation over the
        # outer `[B, S]` dims, exercising the divmod path the migration targets.
        # f32 cols=256 enters the single-pass warp-per-row regime; the wider
        # f32/bf16 cols cover the warp-tiling path.
        run_rms_norm_gpu[.float32](ctx, Index(8, 3072, 256))
        run_rms_norm_gpu[.float32](ctx, Index(4, 1024, 2048))
        run_rms_norm_gpu[.bfloat16](ctx, Index(4, 1024, 4096), rtol=2e-2)
        # Rank 4 through the warp-tiling kernel: the widest coord that reached
        # it was rank 3, leaving the `_RowSpec` argument untested past three
        # dynamic leaves. cols=2048 is an exact fit at bf16.
        run_rms_norm_gpu[.bfloat16](ctx, Index(2, 3, 4, 2048), rtol=2e-2)

        run_rms_norm_gpu[.float32](ctx, Index(32768, 1536))
        run_rms_norm_gpu[.bfloat16](ctx, Index(32768, 1536), rtol=2e-2)
        run_rms_norm_gpu[.float32](ctx, Index(4095, 1536))
        run_rms_norm_gpu[.float32](ctx, Index(64, 256))
        run_rms_norm_gpu[.bfloat16](ctx, Index(64, 256), rtol=2e-2)
        run_rms_norm_gpu[.float32](ctx, Index(32768, 2048))
        run_rms_norm_gpu[.bfloat16](ctx, Index(32768, 4096), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(8, 2048), rtol=2e-2)
        run_rms_norm_gpu[.float32](ctx, Index(8, 8193))

        # Static-dim shapes: `row_major` derives an all-`ComptimeInt` stride
        # list from these, so they are the only cases that exercise the
        # zero-sized-stride collapse of the `Layout` capture carrier and the
        # `_RowSpec` kernel argument (MOCO-4307). Rank 2 and rank 3 keep dim 0
        # dynamic (the batch-dynamic production shape); the fully static rank-2
        # case carries no dynamic leaf at all.
        run_rms_norm_gpu_static_dims[.bfloat16](
            ctx, Coord(Int64(2), Idx[4096]), rtol=2e-2
        )
        run_rms_norm_gpu_static_dims[.bfloat16](
            ctx, Coord(Int64(2), Idx[16384]), rtol=2e-2
        )
        run_rms_norm_gpu_static_dims[.float32](
            ctx, Coord(Int64(4), Idx[1024], Idx[2048])
        )
        run_rms_norm_gpu_static_dims[.float32](ctx, Coord(Idx[2], Idx[8192]))

        # High-row-count, register-resident widths: exercises the CDNA4
        # wide-SIMD warp-tiling path (gated on row count). cols are multiples
        # of 16 in the (128, 8192] warp-tiling range.
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 4096, 4096), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 8192, 2880), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 8192, 5120), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 8192, 8192), rtol=2e-2)

        # Single-pass warp-per-row path: narrow exact-fit rows of 1..4 SIMD
        # vectors per lane (cols == chunks * WARP_SIZE * simd_width) with enough
        # rows to enter the warp-per-row region. f32 sp_stride=128 so cols
        # 128/256/384/512 => chunks 1..4; bf16 sp_stride=256 so cols
        # 256/512/768/1024 => chunks 1..4. The rank-4 case is a
        # production-representative normalized shape [4096,6,1,256] (rows=24576).
        # Cover both `multiply_before_cast` values.
        run_rms_norm_gpu[.float32](ctx, Index(24576, 128))
        run_rms_norm_gpu[.float32](ctx, Index(24576, 256))
        run_rms_norm_gpu[.float32](ctx, Index(24576, 384))
        run_rms_norm_gpu[.float32](ctx, Index(24576, 512))
        run_rms_norm_gpu[.float32](ctx, Index(4096, 6, 1, 256))
        run_rms_norm_gpu[.float32, multiply_before_cast=False](
            ctx, Index(24576, 512)
        )
        # bf16 narrow exact-fit rows now also use single-pass (previously
        # f32-only); chunks 1..4 across both `multiply_before_cast` values.
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 256), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 512), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 768), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 1024), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(24576, 768), rtol=2e-2
        )

        # Cover the `multiply_before_cast=False` path (used by e.g. Llama)
        # across the multi-chunk warp-tiling dispatch: exact-fit and ragged,
        # narrow (simd) and wide (simd*2) branches.
        run_rms_norm_gpu[.float32, multiply_before_cast=False](
            ctx, Index(2, 8192)
        )
        run_rms_norm_gpu[.float32, multiply_before_cast=False](
            ctx, Index(7, 557)
        )
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 4096), rtol=2e-2
        )
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 8192), rtol=2e-2
        )
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 5120), rtol=2e-2
        )

        # Row-based `rms_norm` (the `mo.reduce.rms_norm` lowering path), across
        # both `multiply_before_cast` branches and the narrow/wide dispatch.
        run_rms_norm_row_based[DType.float32](ctx, Index(2, 5))
        run_rms_norm_row_based[DType.float32](ctx, Index(7, 557))
        run_rms_norm_row_based[DType.float32](ctx, Index(24576, 256))
        run_rms_norm_row_based[DType.float32](ctx, Index(2, 8192))
        run_rms_norm_row_based[DType.float32](ctx, Index(3, 4, 10, 20, 8))
        run_rms_norm_row_based[DType.bfloat16](ctx, Index(64, 256), rtol=2e-2)
        run_rms_norm_row_based[DType.bfloat16](
            ctx, Index(4, 1024, 4096), rtol=2e-2
        )
        run_rms_norm_row_based[DType.float32, multiply_before_cast=False](
            ctx, Index(7, 557)
        )
        run_rms_norm_row_based[DType.float32, multiply_before_cast=False](
            ctx, Index(24576, 512)
        )
        run_rms_norm_row_based[DType.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 4096), rtol=2e-2
        )

        # Static row width (the register-cached `_fuse` path) under a rank-3
        # `[B, S, H]` shape with dynamic outer dims: exercises the row -> n-d
        # decode over two outer dims, a chunk count that does not divide the
        # block, and the warp, block and ragged-tail tiers.
        run_rms_norm_row_based[DType.bfloat16, static_cols=4096](
            ctx, Index(3, 41, 4096), rtol=2e-2
        )
        run_rms_norm_row_based[DType.bfloat16, static_cols=7168](
            ctx, Index(2, 7, 7168), rtol=2e-2
        )
        run_rms_norm_row_based[DType.bfloat16, static_cols=2880](
            ctx, Index(1, 513, 2880), rtol=2e-2
        )
        run_rms_norm_row_based[DType.bfloat16, static_cols=1024](
            ctx, Index(5, 1000, 1024), rtol=2e-2
        )
        run_rms_norm_row_based[DType.bfloat16, static_cols=2050](
            ctx, Index(2, 33, 2050), rtol=2e-2
        )
        run_rms_norm_row_based[DType.float32, static_cols=557](
            ctx, Index(3, 5, 557)
        )
        run_rms_norm_row_based[DType.float32, static_cols=8192](
            ctx, Index(2, 3, 5, 8192)
        )
        run_rms_norm_row_based[
            DType.bfloat16, multiply_before_cast=False, static_cols=4096
        ](ctx, Index(4, 4096), rtol=2e-2)

        # Few rows of at least `_SPLITK_MIN_ROW_BYTES` take the phase-aware
        # split-K tier: one row, the 16-row cap, an odd width, and both
        # `multiply_before_cast` branches.
        run_rms_norm_row_based[DType.float32](ctx, Index(1, 65536))
        run_rms_norm_row_based[DType.float32](ctx, Index(3, 100003))
        run_rms_norm_row_based[DType.bfloat16](
            ctx, Index(16, 262144), rtol=2e-2
        )
        run_rms_norm_row_based[DType.bfloat16, multiply_before_cast=False](
            ctx, Index(2, 1048576), rtol=2e-2
        )
