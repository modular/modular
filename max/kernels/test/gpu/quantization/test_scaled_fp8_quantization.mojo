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

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from linalg.fp8_quantization import (
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
    batched_quantize_dynamic_scaled_fp8,
)
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import has_nvidia_gpu_accelerator
from testing import assert_equal

from utils import Index, IndexList
from utils.numerics import get_accum_type, max_finite, min_finite


comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


fn test_static_scaled_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    M: Optional[Int],
    N: Optional[Int],
](ctx: DeviceContext, scale: Float32, m: Int, n: Int) raises:
    comptime static_shape = DimList(to_dim[M], to_dim[N])
    var dynamic_shape = Index(M.or_else(m), N.or_else(n))
    var total_size = m * n

    comptime layout_2d = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )

    var in_host_ptr = UnsafePointer[Scalar[in_dtype]].alloc(total_size)
    var out_host_ptr = UnsafePointer[Scalar[out_dtype]].alloc(total_size)

    var in_host = LayoutTensor[in_dtype, layout_2d](
        in_host_ptr,
        RuntimeLayout[layout_2d].row_major(dynamic_shape),
    )
    var out_host = LayoutTensor[out_dtype, layout_2d](
        out_host_ptr,
        RuntimeLayout[layout_2d].row_major(dynamic_shape),
    )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)

    random(in_host)
    _ = out_host.fill(0)

    ctx.enqueue_copy(in_device, in_host_ptr)
    ctx.enqueue_copy(out_device, out_host_ptr)

    var in_ndbuffer = NDBuffer[in_dtype, 2, _, static_shape](
        in_device.unsafe_ptr(),
        DimList(m, n),
    )
    var out_ndbuffer = NDBuffer[out_dtype, 2, _, static_shape](
        out_device.unsafe_ptr(),
        DimList(m, n),
    )

    quantize_static_scaled_fp8[out_dtype, in_dtype](
        out_ndbuffer, in_ndbuffer, scale, ctx
    )

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
    _ = in_device^
    _ = out_device^


fn test_dynamic_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
    M: Optional[Int],
    N: Optional[Int],
](ctx: DeviceContext, m: Int, n: Int) raises:
    comptime group_size = N.or_else(
        UNKNOWN_VALUE
    ) if group_size_or_per_token == -1 else group_size_or_per_token
    comptime accum_dtype = get_accum_type[in_dtype]()

    comptime static_shape = DimList(to_dim[M], to_dim[N])
    comptime static_scales_shape = DimList(to_dim[N] // group_size, to_dim[M])
    var dynamic_shape = Index(M.or_else(m), N.or_else(n))
    var dynamic_scales_shape = Index(n // group_size, m)
    var total_size = m * n
    var scales_size = (n // group_size) * m

    comptime layout_2d = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )
    comptime scales_layout = Layout.row_major(
        N.or_else(UNKNOWN_VALUE) // group_size, M.or_else(UNKNOWN_VALUE)
    )

    var in_host_ptr = UnsafePointer[Scalar[in_dtype]].alloc(total_size)
    var out_host_ptr = UnsafePointer[Scalar[out_dtype]].alloc(total_size)
    var scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(scales_size)

    var in_host = LayoutTensor[in_dtype, layout_2d](
        in_host_ptr,
        RuntimeLayout[layout_2d].row_major(dynamic_shape),
    )
    var out_host = LayoutTensor[out_dtype, layout_2d](
        out_host_ptr,
        RuntimeLayout[layout_2d].row_major(dynamic_shape),
    )
    var scales_host = LayoutTensor[scales_dtype, scales_layout](
        scales_host_ptr,
        RuntimeLayout[scales_layout].row_major(dynamic_scales_shape),
    )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_ndbuffer = NDBuffer[in_dtype, 2, _, static_shape](
        in_device.unsafe_ptr(),
        DimList(m, n),
    )
    var out_ndbuffer = NDBuffer[out_dtype, 2, _, static_shape](
        out_device.unsafe_ptr(),
        DimList(m, n),
    )
    var scales_ndbuffer = NDBuffer[scales_dtype, 2, _, static_scales_shape](
        scales_device.unsafe_ptr(),
        DimList(n // group_size, m),
    )

    @__copy_capture(in_ndbuffer)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[in_dtype, width]:
        return in_ndbuffer.load[width=width, alignment=alignment](row, col)

    quantize_dynamic_scaled_fp8[
        input_fn, group_size_or_per_token, in_ndbuffer.shape.get[1]()
    ](
        out_ndbuffer.make_dims_unknown(),
        scales_ndbuffer.make_dims_unknown(),
        1200.0,
        ctx,
        in_ndbuffer.dim[0](),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    for i in range(m):
        for group_idx in range(n // group_size):
            var group_max = Scalar[in_dtype](0)
            for j in range(group_size):
                group_max = max(
                    group_max,
                    abs(in_host[i, j + group_idx * group_size][0]),
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
    _ = in_device^
    _ = out_device^
    _ = scales_device^


fn test_batched_dynamic_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
    BS: Optional[Int],
    M: Optional[Int],
    K: Optional[Int],
](ctx: DeviceContext, bs: Int, m: Int, k: Int) raises:
    comptime group_size = K.or_else(
        UNKNOWN_VALUE
    ) if group_size_or_per_token == -1 else group_size_or_per_token
    comptime accum_dtype = get_accum_type[in_dtype]()

    comptime static_shape = DimList(to_dim[BS], to_dim[M], to_dim[K])
    comptime static_scales_shape = DimList(
        to_dim[BS],
        to_dim[K] // group_size,
        to_dim[M],
    )
    var dynamic_shape = Index(BS.or_else(bs), M.or_else(m), K.or_else(k))
    var dynamic_scales_shape = Index(bs, k // group_size, m)
    var total_size = bs * m * k
    var scales_size = bs * (k // group_size) * m

    comptime layout_3d = Layout.row_major(
        BS.or_else(UNKNOWN_VALUE),
        M.or_else(UNKNOWN_VALUE),
        K.or_else(UNKNOWN_VALUE),
    )
    comptime scales_layout = Layout.row_major(
        BS.or_else(UNKNOWN_VALUE),
        K.or_else(UNKNOWN_VALUE) // group_size,
        M.or_else(UNKNOWN_VALUE),
    )

    var in_host_ptr = UnsafePointer[Scalar[in_dtype]].alloc(total_size)
    var out_host_ptr = UnsafePointer[Scalar[out_dtype]].alloc(total_size)
    var scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(scales_size)

    var in_host = LayoutTensor[in_dtype, layout_3d](
        in_host_ptr,
        RuntimeLayout[layout_3d].row_major(dynamic_shape),
    )
    var out_host = LayoutTensor[out_dtype, layout_3d](
        out_host_ptr,
        RuntimeLayout[layout_3d].row_major(dynamic_shape),
    )
    var scales_host = LayoutTensor[scales_dtype, scales_layout](
        scales_host_ptr,
        RuntimeLayout[scales_layout].row_major(dynamic_scales_shape),
    )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_ndbuffer = NDBuffer[in_dtype, 3, _, static_shape](
        in_device.unsafe_ptr(),
        DimList(bs, m, k),
    )
    var out_ndbuffer = NDBuffer[out_dtype, 3, _, static_shape](
        out_device.unsafe_ptr(),
        DimList(bs, m, k),
    )
    var scales_ndbuffer = NDBuffer[scales_dtype, 3, _, static_scales_shape](
        scales_device.unsafe_ptr(),
        DimList(bs, k // group_size, m),
    )

    @parameter
    @__copy_capture(in_ndbuffer)
    @always_inline
    fn input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) capturing -> SIMD[in_dtype, width]:
        return in_ndbuffer.load[width=width, alignment=alignment](
            Index(batch, row, col)
        )

    batched_quantize_dynamic_scaled_fp8[
        input_fn=input_fn,
        group_size_or_per_token=group_size_or_per_token,
        num_cols = in_ndbuffer.shape.get[2](),
    ](
        out_ndbuffer.make_dims_unknown(),
        scales_ndbuffer.make_dims_unknown(),
        1200.0,
        ctx,
        num_rows=m,
        batch_size=bs,
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    for batch_idx in range(bs):
        for i in range(m):
            for group_idx in range(k // group_size):
                var group_max = Scalar[in_dtype](0)
                for j in range(group_size):
                    group_max = max(
                        group_max,
                        abs(
                            in_host[batch_idx, i, j + group_idx * group_size][0]
                        ),
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
    _ = in_device^
    _ = out_device^
    _ = scales_device^


def main():
    with DeviceContext() as ctx:
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn, DType.bfloat16, M=None, N = Int(16)
        ](ctx, 0.5, 32, 16)
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn, DType.float16, M=None, N = Int(15)
        ](ctx, 0.33, 31, 15)
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn, DType.bfloat16, M=None, N = Int(15)
        ](ctx, 0.3323, 31, 15)

        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            M=None,
            N = Int(256),
        ](ctx, 1, 256)
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            M=None,
            N = Int(1024),
        ](ctx, 1, 1024)
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            M=None,
            N = Int(16384),
        ](ctx, 1, 16384)
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
            M=None,
            N = Int(16384),
        ](ctx, 4, 16384)
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
            M=None,
            N = Int(576),
        ](ctx, 4, 576)

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            M=None,
            N = Int(260),
        ](ctx, 2, 260)
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            M=None,
            N = Int(264),
        ](ctx, 2, 264)

        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            BS=None,
            M=None,
            K = Int(256),
        ](ctx, 2, 1, 256)
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            BS=None,
            M=None,
            K = Int(1024),
        ](ctx, 3, 1, 1024)
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
            BS=None,
            M=None,
            K = Int(16384),
        ](ctx, 4, 1, 16384)
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
            BS=None,
            M=None,
            K = Int(512),
        ](ctx, 128, 400, 512)
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
            BS=None,
            M=None,
            K = Int(128),
        ](ctx, 128, 1024, 128)

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            132,
            BS=None,
            M=None,
            K = Int(528),
        ](ctx, 128, 400, 528)
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            136,
            BS=None,
            M=None,
            K = Int(544),
        ](ctx, 128, 1024, 544)

        # DType.float8_e8m0fnu is only supported on NVIDIA GPUs
        comptime if has_nvidia_gpu_accelerator():
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
                M=None,
                N = Int(1024),
            ](ctx, 43, 1024)
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
                M=None,
                N = Int(16384),
            ](ctx, 3, 16384)
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.float32,
                DType.float8_e8m0fnu,
                128,
                M=None,
                N = Int(576),
            ](ctx, 1, 576)
