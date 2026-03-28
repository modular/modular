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

from std.gpu.host import DeviceContext
from layout import Coord, CoordLike, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.fp8_quantization import (
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
    batched_quantize_dynamic_scaled_fp8,
)
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_equal

from std.utils.numerics import get_accum_type, max_finite, min_finite


def test_static_scaled_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
](ctx: DeviceContext, scale: Float32, m: Int, n: Int) raises:
    var shape = row_major(Coord(Idx(Int64(m)), Idx(Int64(n))))
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

    var in_tt = TileTensor(in_device.unsafe_ptr(), shape)
    var out_tt = TileTensor(out_device.unsafe_ptr(), shape)

    quantize_static_scaled_fp8[out_dtype, in_dtype](out_tt, in_tt, scale, ctx)

    ctx.enqueue_copy(out_host_ptr, out_device)

    ctx.synchronize()

    for i in range(m):
        for j in range(n):
            var in_val_scaled_f32: Float32

            in_val_scaled_f32 = in_host[i, j][0].cast[DType.float32]() * (
                1.0 / scale
            )

            in_val_scaled_f32 = max(
                Float32(min_finite[out_dtype]()),
                min(Float32(max_finite[out_dtype]()), in_val_scaled_f32),
            )

            assert_equal(
                in_val_scaled_f32.cast[DType.float8_e4m3fn]().cast[
                    DType.float64
                ](),
                out_host[i, j][0].cast[DType.float64](),
            )

    in_host_ptr.free()
    out_host_ptr.free()


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
        Coord(Idx[NType.static_value // group_size](), m)
    )
    var total_size = m.value() * n.value()
    var scales_size = (n.value() // group_size) * m.value()

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

    @__copy_capture(in_tensor)
    @always_inline
    @parameter
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment](
            (Idx(row), Idx(col))
        )

    quantize_dynamic_scaled_fp8[
        input_fn, group_size_or_per_token, in_tensor.static_shape[1]
    ](
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
    for i in range(m.value()):
        for group_idx in range(n.value() // group_size):
            var group_max = Scalar[in_dtype](0)
            for j in range(group_size):
                group_max = max(
                    group_max,
                    abs(in_host[i, j + group_idx * group_size]),
                )

            var scale_factor: Scalar[scales_dtype]

            comptime if scales_dtype == DType.float8_e8m0fnu:
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
                scales_host[group_idx, i].cast[DType.float32](),
                scale_factor.cast[DType.float32](),
            )

            for j in range(group_size):
                var in_val = in_host[i, j + group_idx * group_size]
                var out_val = out_host[i, j + group_idx * group_size]
                assert_equal(
                    out_val.cast[DType.float32](),
                    (in_val.cast[accum_dtype]() * scale_factor_recip)
                    .cast[out_dtype]()
                    .cast[DType.float32](),
                    msg="At ["
                    + String(i)
                    + ", "
                    + String(j + group_idx * group_size)
                    + "]",
                )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


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
        Coord(bs, Idx[KType.static_value // group_size](), m)
    )
    var total_size = bs.value() * m.value() * k.value()
    var scales_size = bs.value() * (k.value() // group_size) * m.value()

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

    @parameter
    @__copy_capture(in_tensor)
    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) capturing -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment](
            (Idx(batch), Idx(row), Idx(col))
        )

    batched_quantize_dynamic_scaled_fp8[
        input_fn=input_fn,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=in_tensor.static_shape[2],
    ](
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        num_rows=m.value(),
        batch_size=bs.value(),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    comptime assert in_host.flat_rank == 3
    comptime assert scales_host.flat_rank == 3
    for batch_idx in range(bs.value()):
        for i in range(m.value()):
            for group_idx in range(k.value() // group_size):
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
                    scales_host[batch_idx, group_idx, i].cast[DType.float64](),
                    scale_factor.cast[DType.float64](),
                )

                for j in range(group_size):
                    var in_val = in_host[
                        batch_idx, i, j + group_idx * group_size
                    ]
                    var out_val = out_host[
                        batch_idx, i, j + group_idx * group_size
                    ]

                    assert_equal(
                        out_val.cast[DType.float32](),
                        (in_val.cast[accum_dtype]() * scale_factor_recip)
                        .cast[out_dtype]()
                        .cast[DType.float32](),
                        msg="At ["
                        + String(i)
                        + ", "
                        + String(j + group_idx * group_size)
                        + "]",
                    )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
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

        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(1)), Idx[256]())
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(1)), Idx[1024]())
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(1)), Idx[16384]())
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Idx(Int(4)), Idx[16384]())
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Idx(Int(4)), Idx[576]())

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(2)), Idx[260]())
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(2)), Idx[264]())

        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(2)), Idx(Int(1)), Idx[256]())
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(3)), Idx(Int(1)), Idx[1024]())
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Idx(Int(4)), Idx(Int(1)), Idx[16384]())
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Idx(Int(128)), Idx(Int(400)), Idx[512]())
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Idx(Int(128)), Idx(Int(1024)), Idx[128]())

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            132,
        ](ctx, Idx(Int(128)), Idx(Int(400)), Idx[528]())
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            136,
        ](ctx, Idx(Int(128)), Idx(Int(1024)), Idx[544]())

        # DType.float8_e8m0fnu is only supported on NVIDIA GPUs
        comptime if has_nvidia_gpu_accelerator():
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Idx(Int(43)), Idx[1024]())
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Idx(Int(3)), Idx[16384]())
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.float32,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Idx(Int(1)), Idx[576]())
