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

from max.gpu.host import DeviceContext
from layout import Coord, CoordLike, Idx, MixedLayout, TileTensor, row_major
from layout._fillers import random
from linalg.fp8_quantization import (
    batched_quantize_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
    quantize_tensor_dynamic_scaled_fp8,
)
from std.testing import assert_equal, assert_true, assert_almost_equal

from std.utils.numerics import (
    get_accum_type,
    isinf,
    isnan,
    max_finite,
    min_finite,
)


def test_static_scaled_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
](ctx: DeviceContext, scale: Float32, m: Int, n: Int) raises:
    var shape = row_major(Coord(Int64(m), Int64(n)))
    var total_size = m * n

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)

    random(in_host)
    _ = out_host.fill(0)

    ctx.enqueue_copy(in_device, in_host_ptr)
    ctx.enqueue_copy(out_device, out_host_ptr)

    var in_tt = TileTensor(in_device, shape)
    var out_tt = TileTensor(out_device, shape)

    quantize_static_scaled_fp8[out_dtype, in_dtype](out_tt, in_tt, scale, ctx)

    ctx.enqueue_copy(out_host_ptr, out_device)

    ctx.synchronize()

    for i in range(m):
        for j in range(n):
            var in_val_scaled_f32: Float32

            in_val_scaled_f32 = in_host[i, j][0].cast[.float32]() * (
                1.0 / scale
            )

            in_val_scaled_f32 = max(
                Float32(min_finite[out_dtype]()),
                min(Float32(max_finite[out_dtype]()), in_val_scaled_f32),
            )

            assert_equal(
                in_val_scaled_f32.cast[.float8_e4m3fn]().cast[DType.float64](),
                out_host[i, j][0].cast[.float64](),
            )

    in_host_ptr.free()
    out_host_ptr.free()


def test_dynamic_scaled_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    comptime group_size: Int = NType.static_value
    comptime accum_dtype = get_accum_type[in_dtype]()

    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)
    var scales_host = TileTensor(scales_host_ptr, scales_shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_tensor_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=-1,
        num_cols=in_tensor.static_shape[1],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        Int(in_tensor.dim[0]()),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    comptime assert in_host.flat_rank == 2
    comptime assert scales_host.flat_rank == 2

    comptime assert in_host.flat_rank == 2
    comptime assert scales_host.flat_rank == 2

    var max_scale = Scalar[scales_dtype](0)
    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            max_scale = max(
                max_scale,
                abs(in_host[i, j].cast[scales_dtype]()),
            )

    var scale_factor: Scalar[scales_dtype]

    scale_factor = (
        min(max_scale, 1200.0)
        / Scalar[out_dtype].MAX_FINITE.cast[scales_dtype]()
    )
    var scale_factor_recip = 1.0 / scale_factor.cast[accum_dtype]()

    assert_equal(
        scales_host[0, 0].cast[.float32](),
        scale_factor.cast[.float32](),
    )

    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            var in_val = in_host[i, j]
            var out_val = out_host[i, j]
            assert_equal(
                out_val.cast[.float32](),
                (in_val.cast[accum_dtype]() * scale_factor_recip)
                .cast[out_dtype]()
                .cast[.float32](),
                msg="At [" + String(i) + ", " + String(j) + "]",
            )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_dynamic_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    comptime group_size: Int = NType.static_value if group_size_or_per_token == -1 else group_size_or_per_token
    comptime accum_dtype = get_accum_type[in_dtype]()

    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)
    var scales_host = TileTensor(scales_host_ptr, scales_shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=in_tensor.static_shape[1],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        Int(in_tensor.dim[0]()),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    comptime assert in_host.flat_rank == 2
    comptime assert scales_host.flat_rank == 2
    for i in range(Int(m.value())):
        for group_idx in range(Int(n.value()) // group_size):
            var group_max = Scalar[in_dtype](0)
            for j in range(group_size):
                group_max = max(
                    group_max,
                    abs(in_host[i, j + group_idx * group_size]),
                )

            var scale_factor: Scalar[scales_dtype]

            comptime if scales_dtype == .float8_e8m0fnu:
                scale_factor = max(
                    group_max.cast[accum_dtype]()
                    / Scalar[out_dtype].MAX_FINITE.cast[accum_dtype](),
                    Scalar[accum_dtype](1e-10),
                ).cast[scales_dtype]()
            else:
                scale_factor = (
                    min(group_max.cast[scales_dtype](), 1200.0)
                    / Scalar[out_dtype].MAX_FINITE.cast[scales_dtype]()
                )
            var scale_factor_recip = 1.0 / scale_factor.cast[accum_dtype]()

            assert_equal(
                scales_host[group_idx, i].cast[.float32](),
                scale_factor.cast[.float32](),
            )

            for j in range(group_size):
                var in_val = in_host[i, j + group_idx * group_size]
                var out_val = out_host[i, j + group_idx * group_size]
                assert_equal(
                    out_val.cast[.float32](),
                    (in_val.cast[accum_dtype]() * scale_factor_recip)
                    .cast[out_dtype]()
                    .cast[.float32](),
                    msg="At ["
                    + String(i)
                    + ", "
                    + String(j + group_idx * group_size)
                    + "]",
                )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_dynamic_fp8_quant_amax_floor[
    out_dtype: DType,
    in_dtype: DType,
    group_size: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType, amax_floor: Float32) raises:
    """Checks the `amax_floor` lower bound on the `float8_e8m0fnu` scale path.

    QAT recipes that floor the activation max-abs before deriving the scale
    (DeepSeek-V4's `act_quant` floors at 1e-4) only match MAX bit for bit when
    the kernel applies the same floor. Even-numbered groups here are filled
    below the floor and odd-numbered ones above it, so the same run covers both
    "floor applies" and "floor is inert", and each is also compared against an
    unfloored run of the same input.
    """
    comptime scales_dtype = DType.float8_e8m0fnu
    comptime accum_dtype = get_accum_type[in_dtype]()
    comptime below_floor = Scalar[in_dtype](1e-6)
    comptime above_floor = Scalar[in_dtype](0.5)

    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())
    var num_groups = Int(n.value()) // group_size

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var base_out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)
    var base_scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    for i in range(Int(m.value())):
        for g in range(num_groups):
            var magnitude = below_floor if g % 2 == 0 else above_floor
            for j in range(group_size):
                var col = g * group_size + j
                in_host_ptr[i * Int(n.value()) + col] = (
                    -magnitude if j % 3 == 0 else magnitude
                )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var base_out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    var base_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_size
    )
    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var base_out_tensor = TileTensor(base_out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)
    var base_scales_tensor = TileTensor(base_scales_device, scales_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size,
        num_cols=in_tensor.static_shape[1],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        Int(in_tensor.dim[0]()),
        amax_floor,
    )

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size,
        num_cols=in_tensor.static_shape[1],
    ](
        input_fn,
        base_out_tensor,
        base_scales_tensor,
        1200.0,
        ctx,
        Int(in_tensor.dim[0]()),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(base_out_host_ptr, base_out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.enqueue_copy(base_scales_host_ptr, base_scales_device)
    ctx.synchronize()

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)
    var base_out_host = TileTensor(base_out_host_ptr, shape)
    var scales_host = TileTensor(scales_host_ptr, scales_shape)
    var base_scales_host = TileTensor(base_scales_host_ptr, scales_shape)

    for i in range(Int(m.value())):
        for group_idx in range(num_groups):
            var group_max = Scalar[accum_dtype](0)
            for j in range(group_size):
                group_max = max(
                    group_max,
                    abs(in_host[i, j + group_idx * group_size]).cast[
                        accum_dtype
                    ](),
                )

            var floored_max = max(group_max, amax_floor.cast[accum_dtype]())
            var scale_factor = max(
                floored_max / Scalar[out_dtype].MAX_FINITE.cast[accum_dtype](),
                Scalar[accum_dtype](1e-10),
            ).cast[scales_dtype]()
            var scale_factor_recip = 1.0 / scale_factor.cast[accum_dtype]()

            assert_equal(
                scales_host[group_idx, i].cast[.float32](),
                scale_factor.cast[.float32](),
            )

            var floor_applies = group_max < amax_floor.cast[accum_dtype]()
            if floor_applies:
                assert_true(
                    scales_host[group_idx, i].cast[.float32]()
                    != base_scales_host[group_idx, i].cast[.float32](),
                    msg="floor left the scale unchanged on a near-zero group",
                )
            else:
                assert_equal(
                    scales_host[group_idx, i].cast[.float32](),
                    base_scales_host[group_idx, i].cast[.float32](),
                )

            for j in range(group_size):
                var col = j + group_idx * group_size
                var expected = (
                    in_host[i, col].cast[accum_dtype]() * scale_factor_recip
                ).cast[out_dtype]()
                assert_equal(
                    out_host[i, col].cast[.float32](),
                    expected.cast[.float32](),
                    msg="At [" + String(i) + ", " + String(col) + "]",
                )
                if not floor_applies:
                    assert_equal(
                        out_host[i, col].cast[.float32](),
                        base_out_host[i, col].cast[.float32](),
                    )

    in_host_ptr.free()
    out_host_ptr.free()
    base_out_host_ptr.free()
    scales_host_ptr.free()
    base_scales_host_ptr.free()


def test_batched_dynamic_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
    BSType: CoordLike,
    MType: CoordLike,
    KType: CoordLike,
](ctx: DeviceContext, bs: BSType, m: MType, k: KType) raises:
    comptime group_size: Int = KType.static_value if group_size_or_per_token == -1 else group_size_or_per_token
    comptime accum_dtype = get_accum_type[in_dtype]()

    var shape = row_major(Coord(bs, m, k))
    var scales_shape = row_major(
        Coord(bs, Idx[KType.static_value // group_size], m)
    )
    var total_size = Int(bs.value()) * Int(m.value()) * Int(k.value())
    var scales_size = (
        Int(bs.value()) * (Int(k.value()) // group_size) * Int(m.value())
    )

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)
    var scales_host = TileTensor(scales_host_ptr, scales_shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment](
            (batch, row, col)
        )

    batched_quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=in_tensor.static_shape[2],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        num_rows=Int(m.value()),
        batch_size=Int(bs.value()),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    comptime assert in_host.flat_rank == 3
    comptime assert scales_host.flat_rank == 3
    for batch_idx in range(Int(bs.value())):
        for i in range(Int(m.value())):
            for group_idx in range(Int(k.value()) // group_size):
                var group_max = Scalar[in_dtype](0)
                for j in range(group_size):
                    group_max = max(
                        group_max,
                        abs(in_host[batch_idx, i, j + group_idx * group_size]),
                    )

                var scale_factor = (
                    min(group_max, 1200.0)
                    / Scalar[out_dtype].MAX_FINITE.cast[in_dtype]()
                )
                var scale_factor_recip = 1.0 / scale_factor.cast[accum_dtype]()

                assert_equal(
                    scales_host[batch_idx, group_idx, i].cast[.float64](),
                    scale_factor.cast[.float64](),
                )

                for j in range(group_size):
                    var in_val = in_host[
                        batch_idx, i, j + group_idx * group_size
                    ]
                    var out_val = out_host[
                        batch_idx, i, j + group_idx * group_size
                    ]

                    assert_equal(
                        out_val.cast[.float32](),
                        (in_val.cast[accum_dtype]() * scale_factor_recip)
                        .cast[out_dtype]()
                        .cast[.float32](),
                        msg="At ["
                        + String(i)
                        + ", "
                        + String(j + group_idx * group_size)
                        + "]",
                    )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_dynamic_fp8_quant_row_bounded[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType, live_rows: Int) raises:
    """Checks a row-bounded quantize against the unbounded one.

    The bounded launch must reproduce the unbounded result bit-for-bit on every
    row below `live_rows`, for both the quantized values and the transposed
    scales, and must leave every row at or past it exactly as it found it. That
    second half is what lets the EP down projection skip its padding: the rows
    it stops writing are rows the grouped matmul never reads.
    """
    comptime group_size: Int = NType.static_value if group_size_or_per_token == -1 else group_size_or_per_token

    var rows = Int(m.value())
    var cols = Int(n.value())
    var num_groups = cols // group_size
    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = rows * cols
    var scales_size = num_groups * rows

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var ref_out_ptr = alloc[Scalar[out_dtype]](total_size)
    var ref_scales_ptr = alloc[Scalar[scales_dtype]](scales_size)
    var got_out_ptr = alloc[Scalar[out_dtype]](total_size)
    var got_scales_ptr = alloc[Scalar[scales_dtype]](scales_size)
    var zero_out_ptr = alloc[Scalar[out_dtype]](total_size)
    var zero_scales_ptr = alloc[Scalar[scales_dtype]](scales_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var ref_out = TileTensor(ref_out_ptr, shape)
    var ref_scales = TileTensor(ref_scales_ptr, scales_shape)
    var got_out = TileTensor(got_out_ptr, shape)
    var got_scales = TileTensor(got_scales_ptr, scales_shape)

    # `random` never returns exactly zero across a whole group, so a zeroed
    # buffer is a poison the unbounded launch would visibly overwrite. The
    # negative control below asserts that before trusting the skip.
    _ = TileTensor(zero_out_ptr, shape).fill(0)
    _ = TileTensor(zero_scales_ptr, scales_shape).fill(0)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    var offsets_device = ctx.enqueue_create_buffer[.uint32](4)

    random(in_host, -1.0, 1.0)
    ctx.enqueue_copy(in_device, in_host_ptr)

    var offsets_host_ptr = alloc[UInt32](4)
    offsets_host_ptr.unsafe_store(UInt32(0))
    offsets_host_ptr.unsafe_offset(1).unsafe_store(UInt32(live_rows // 3))
    offsets_host_ptr.unsafe_offset(2).unsafe_store(UInt32(live_rows // 2))
    offsets_host_ptr.unsafe_offset(3).unsafe_store(UInt32(live_rows))
    ctx.enqueue_copy(offsets_device, offsets_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=in_tensor.static_shape[1],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        rows,
    )
    ctx.enqueue_copy(ref_out_ptr, out_device)
    ctx.enqueue_copy(ref_scales_ptr, scales_device)

    ctx.enqueue_copy(out_device, zero_out_ptr)
    ctx.enqueue_copy(scales_device, zero_scales_ptr)

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=in_tensor.static_shape[1],
        row_bounded=True,
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        rows,
        row_limit=OptionalPointer[UInt32, ImmUntrackedOrigin](
            offsets_device.unsafe_ptr()
            .unsafe_offset(3)
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_mut_cast[False]()
        ),
    )
    ctx.enqueue_copy(got_out_ptr, out_device)
    ctx.enqueue_copy(got_scales_ptr, scales_device)
    ctx.synchronize()

    var skipped_ref_is_nonzero = False
    for i in range(rows):
        for group_idx in range(num_groups):
            var reference = ref_scales[group_idx, i].cast[.float64]()
            var bounded = got_scales[group_idx, i].cast[.float64]()
            if i < live_rows:
                assert_equal(
                    bounded,
                    reference,
                    msg=String("scale at [", group_idx, ", ", i, "]"),
                )
            else:
                assert_equal(
                    bounded,
                    Float64(0),
                    msg=String("scale written past live rows at ", i),
                )
                if reference != 0:
                    skipped_ref_is_nonzero = True

        for j in range(cols):
            var reference = ref_out[i, j].cast[.float64]()
            var bounded = got_out[i, j].cast[.float64]()
            if i < live_rows:
                assert_equal(
                    bounded,
                    reference,
                    msg=String("value at [", i, ", ", j, "]"),
                )
            else:
                assert_equal(
                    bounded,
                    Float64(0),
                    msg=String("value written past live rows at ", i),
                )

    # Without this the "untouched" assertions above pass vacuously whenever the
    # unbounded launch would also have left zeros there.
    assert_true(
        live_rows >= rows or skipped_ref_is_nonzero,
        msg="unbounded run left the skipped rows zero, so the skip is untested",
    )

    in_host_ptr.free()
    ref_out_ptr.free()
    ref_scales_ptr.free()
    got_out_ptr.free()
    got_scales_ptr.free()
    zero_out_ptr.free()
    zero_scales_ptr.free()
    offsets_host_ptr.free()


def test_dynamic_fp8_quant_near_zero[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    """Regression for the FP8 dynamic-quant `0*Inf` NaN on a near-zero group.

    When a group's max-abs is a tiny f32 denormal (~1e-38), the dynamic scale
    `scale_factor = group_max / fp8_max` underflows to a NONZERO denormal, so
    `scale_factor_recip = 1/scale_factor` OVERFLOWS to +Inf (the `==0` guard
    misses it). `fp8_quantize` then computes `value * Inf` (NaN on a zero lane,
    Inf otherwise) and casts to fp8 — and on NVIDIA `use_clamp` is False, so the
    NaN/Inf flows into the output. A correct quant emits finite fp8 (the group
    is effectively zero → ~0). Asserts FINITENESS of every output element.
    """
    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    # Fill the whole input with a near-zero magnitude (~1e-38) and a zero lane
    # at the start of each group — so every group's max-abs is a tiny denormal
    # and each group has an exactly-zero element (the 0*Inf lane).
    for i in range(total_size):
        in_host_ptr[i] = Scalar[in_dtype](1e-38)
    for i in range(Int(m.value())):
        for g in range(Int(n.value()) // group_size):
            in_host_ptr[i * Int(n.value()) + g * group_size] = Scalar[in_dtype](
                0
            )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    out_device.enqueue_fill(
        Scalar[out_dtype](0)
    )  # so unwritten cells are finite
    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size,
        num_cols=in_tensor.static_shape[1],
    ](input_fn, out_tensor, scales_tensor, 1200.0, ctx, Int(in_tensor.dim[0]()))

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.synchronize()

    # Post-#87813 the cast clamp saturates the 0*Inf/+Inf, so a finiteness-only
    # check passes even WITHOUT the guard (vacuous — see
    # test_dynamic_tensor_fp8_quant_near_zero). Assert the VALUES are a clean fp8
    # zero, which the guard produces (scale_recip = 0) and the unguarded +Inf
    # path does NOT (it saturates to +-max_finite).
    var n_nonfinite = 0
    var n_nonzero = 0
    for i in range(total_size):
        var v = out_host_ptr[i].cast[.float32]()
        if isnan(v) or isinf(v):
            n_nonfinite += 1
        if v != 0.0:
            n_nonzero += 1
    print(
        "  near-zero group: fp8 out num_nonfinite =",
        n_nonfinite,
        " num_nonzero =",
        n_nonzero,
    )
    assert_equal(n_nonfinite, 0)
    assert_equal(n_nonzero, 0)

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_dynamic_tensor_fp8_quant_near_zero[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    """Regression for the FP8 dynamic-quant `0*Inf` NaN on the PER-TENSOR path.

    Same denormal-scale reciprocal overflow as test_dynamic_fp8_quant_near_zero,
    but routed through `quantize_tensor_dynamic_scaled_fp8` with num_rows > 1, so
    it exercises the two-launch per-tensor reduction (compute_scales_fp8_kernel +
    quantize_fp8_kernel_per_tensor). That second kernel re-derives the scale from
    the tensor-wide max and previously open-coded a `== 0`-only reciprocal guard,
    so the denormal-max overflow could still NaN here (the per-group near-zero
    test above does not reach it). Asserts FINITENESS of every output element.
    """
    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    # Every group's max-abs is a tiny f32 denormal (~1e-38) with an exactly-zero
    # lane at each group start (the 0*Inf lane). With num_rows > 1 the per-tensor
    # path reduces all group maxes to one tensor-wide denormal scale.
    for i in range(total_size):
        in_host_ptr[i] = Scalar[in_dtype](1e-38)
    for i in range(Int(m.value())):
        for g in range(Int(n.value()) // group_size):
            in_host_ptr[i * Int(n.value()) + g * group_size] = Scalar[in_dtype](
                0
            )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    out_device.enqueue_fill(
        Scalar[out_dtype](0)
    )  # so unwritten cells are finite
    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_tensor_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size,
        num_cols=in_tensor.static_shape[1],
    ](input_fn, out_tensor, scales_tensor, 1200.0, ctx, Int(in_tensor.dim[0]()))

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.synchronize()

    # WITH the reciprocal guard a near-zero group quantizes to a CLEAN fp8 zero
    # (scale_recip = 0 -> value*0 = 0). WITHOUT it, scale_recip = +Inf and every
    # lane saturates to +-max_finite at the cast clamp (use_clamp on since
    # #87813): finite, but WRONG. A finiteness-only check is therefore MASKED by
    # the clamp and would not catch a missing guard, so assert the VALUES are 0.
    var n_nonfinite = 0
    var n_nonzero = 0
    for i in range(total_size):
        var v = out_host_ptr[i].cast[.float32]()
        if isnan(v) or isinf(v):
            n_nonfinite += 1
        if v != 0.0:
            n_nonzero += 1
    print(
        "  near-zero per-tensor: fp8 out num_nonfinite =",
        n_nonfinite,
        " num_nonzero =",
        n_nonzero,
    )
    assert_equal(n_nonfinite, 0)
    assert_equal(n_nonzero, 0)

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        # Row-bounded quantize: same bits below the live count, nothing
        # written above it. `live_rows` deliberately straddles the grid-row
        # cap so the second case makes each block grid-stride more than once.
        test_dynamic_fp8_quant_row_bounded[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Int(64), Idx[1024], live_rows=7)
        test_dynamic_fp8_quant_row_bounded[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Int(4096), Idx[1024], live_rows=2000)
        test_dynamic_fp8_quant_row_bounded[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(129), Idx[512], live_rows=33)

        test_dynamic_fp8_quant_row_bounded[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            128,
        ](ctx, Int(1000), Idx[2048], live_rows=333)

        # A group of 3 lanes does not divide the warp, so this stays on the
        # block-per-group fallback (and its larger bounded-grid cap).
        test_dynamic_fp8_quant_row_bounded[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            48,
        ](ctx, Int(300), Idx[192], live_rows=111)
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            48,
        ](ctx, Int(37), Idx[192])

        # Warp-per-several-groups path: row counts that leave a partial tile,
        # several group sizes, 16-bit inputs, and e8m0 scales.
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Int(37), Idx[2048])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(129), Idx[7168])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float16,
            DType.float16,
            128,
        ](ctx, Int(21), Idx[512])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float8_e8m0fnu,
            128,
        ](ctx, Int(33), Idx[1024])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            32,
        ](ctx, Int(35), Idx[512])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            64,
        ](ctx, Int(35), Idx[512])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            256,
        ](ctx, Int(35), Idx[1024])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            512,
        ](ctx, Int(5), Idx[1024])

        # Regression: FP8 dynamic-quant must not emit NaN on a near-zero group
        # (the 0*Inf / denormal-scale-reciprocal bug). Run first.
        #
        # The scales dtype is load-bearing for whether the bug even manifests:
        # for a near-zero group `scale_factor = group_max / fp8_max` is ~2e-41.
        # With FLOAT32 scales that value is preserved (a nonzero f32 denormal),
        # so `1/scale_factor` overflows to +Inf and the unguarded `==0`-only
        # check misses it — this is the case the fix exists for. With BFLOAT16
        # scales the same ~2e-41 underflows below bf16's min denormal (~9.2e-41)
        # and FLUSHES to exactly 0, so the `==0` guard already catches it and no
        # overflow occurs. Cover BOTH so the regression actually exercises the
        # overflow path (float32) and not just the benign bf16-flush path.
        test_dynamic_fp8_quant_near_zero[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            group_size=128,
        ](ctx, Int(4), Idx[512])
        test_dynamic_fp8_quant_near_zero[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            group_size=128,
        ](ctx, Int(4), Idx[512])
        # Same near-zero denormal-scale regression on the PER-TENSOR path
        # (quantize_tensor_dynamic_scaled_fp8 -> quantize_fp8_kernel_per_tensor).
        # num_rows > 1 (m=4) forces the two-launch reduce-then-requantize path,
        # whose reciprocal was previously only `== 0`-guarded. Float32 scales so
        # the denormal survives to overflow `1/scale` (see the dtype note above).
        test_dynamic_tensor_fp8_quant_near_zero[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            group_size=128,
        ](ctx, Int(4), Idx[512])
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
        ](ctx, 0.5, 32, 16)
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.float16,
        ](ctx, 0.33, 31, 15)
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
        ](ctx, 0.3323, 31, 15)

        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Idx[800], Idx[8192])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Idx[1000], Idx[128])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(1), Idx[256])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(1), Idx[1024])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(1), Idx[16384])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(4), Idx[16384])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
        ](ctx, Int(4), Idx[576])

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(2), Idx[260])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(2), Idx[264])

        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(1), Idx[256])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(1), Idx[1024])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(1), Idx[16384])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Int(4), Idx[16384])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(4), Idx[576])

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(2), Idx[260])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(2), Idx[264])

        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(2), Int(1), Idx[256])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(3), Int(1), Idx[1024])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(4), Int(1), Idx[16384])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Int(128), Int(400), Idx[512])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(128), Int(1024), Idx[128])

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            132,
        ](ctx, Int(128), Int(400), Idx[528])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            136,
        ](ctx, Int(128), Int(1024), Idx[544])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(128), Int(1024), Idx[192])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(7), Int(1000), Idx[576])

        # AMD serves FP8 weights as float8_e4m3fnuz (max finite 240, no -0), so
        # the activation quantize must produce that encoding too.
        comptime if ctx.target.is_amd_gpu():
            test_dynamic_fp8_quant[
                DType.float8_e4m3fnuz,
                DType.bfloat16,
                DType.float32,
                128,
            ](ctx, Int(37), Idx[2048])
            test_dynamic_fp8_quant[
                DType.float8_e4m3fnuz,
                DType.float32,
                DType.float32,
                64,
            ](ctx, Int(35), Idx[512])
            test_dynamic_fp8_quant_row_bounded[
                DType.float8_e4m3fnuz,
                DType.bfloat16,
                DType.bfloat16,
                128,
            ](ctx, Int(1000), Idx[1024], live_rows=333)

        # DType.float8_e8m0fnu is only supported on NVIDIA GPUs
        comptime if ctx.target.is_nvidia_gpu():
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Int(43), Idx[1024])
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Int(3), Idx[16384])
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.float32,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Int(1), Idx[576])

            # 1e-4 is the activation amax floor of DeepSeek-V4's `act_quant`.
            test_dynamic_fp8_quant_amax_floor[
                DType.float8_e4m3fn,
                DType.bfloat16,
                128,
            ](ctx, Int(3), Idx[512], 1e-4)
            test_dynamic_fp8_quant_amax_floor[
                DType.float8_e4m3fn,
                DType.float32,
                128,
            ](ctx, Int(1), Idx[256], 1e-4)
