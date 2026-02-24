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
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from testing import assert_almost_equal
from utils.index import Index, IndexList


comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


fn test_matmul_dynamic_scaled_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    transpose_b: Bool,
    M: Optional[Int],
    N: Optional[Int],
    K: Optional[Int],
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    comptime static_a_shape = DimList(to_dim[M], to_dim[K])
    comptime static_b_shape = DimList(
        to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[K], to_dim[N])
    comptime static_c_shape = DimList(to_dim[M], to_dim[N])
    comptime static_a_scales_shape = DimList(Dim(1), to_dim[M])
    comptime static_b_scales_shape = DimList(
        to_dim[N], Dim(1)
    ) if transpose_b else DimList(Dim(1), to_dim[N])

    var dynamic_a_shape = IndexList[2](M.or_else(m), K.or_else(k))
    var dynamic_b_shape = IndexList[2](
        N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[2](K.or_else(k), N.or_else(n))
    var dynamic_c_shape = IndexList[2](M.or_else(m), N.or_else(n))
    var dynamic_a_scales_shape = IndexList[2](1, M.or_else(m))
    var dynamic_b_scales_shape = IndexList[2](
        N.or_else(n), 1
    ) if transpose_b else IndexList[2](1, N.or_else(n))

    var a_size = m * k
    var b_size = n * k if transpose_b else k * n
    var c_size = m * n
    var a_scales_size = 1 * m
    var b_scales_size = n * 1 if transpose_b else 1 * n

    comptime a_layout = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), K.or_else(UNKNOWN_VALUE)
    )
    comptime b_layout = Layout.row_major(
        N.or_else(UNKNOWN_VALUE), K.or_else(UNKNOWN_VALUE)
    ) if transpose_b else Layout.row_major(
        K.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )
    comptime c_layout = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )
    comptime a_scales_layout = Layout.row_major(1, M.or_else(UNKNOWN_VALUE))
    comptime b_scales_layout = Layout.row_major(
        N.or_else(UNKNOWN_VALUE), 1
    ) if transpose_b else Layout.row_major(1, N.or_else(UNKNOWN_VALUE))

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[in_dtype]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[in_dtype]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[out_dtype]].alloc(c_size)
    var a_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        a_scales_size
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        b_scales_size
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[DType.float32]].alloc(c_size)

    var a_host = LayoutTensor[in_dtype, a_layout](
        a_host_ptr,
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
    )
    var b_host = LayoutTensor[in_dtype, b_layout](
        b_host_ptr,
        RuntimeLayout[b_layout].row_major(dynamic_b_shape),
    )
    var c_host = LayoutTensor[out_dtype, c_layout](
        c_host_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )
    var a_scales_host = LayoutTensor[scales_dtype, a_scales_layout](
        a_scales_host_ptr,
        RuntimeLayout[a_scales_layout].row_major(dynamic_a_scales_shape),
    )
    var b_scales_host = LayoutTensor[scales_dtype, b_scales_layout](
        b_scales_host_ptr,
        RuntimeLayout[b_scales_layout].row_major(dynamic_b_scales_shape),
    )
    var c_host_ref = LayoutTensor[DType.float32, c_layout](
        c_host_ref_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[in_dtype](a_size)
    var b_device = ctx.enqueue_create_buffer[in_dtype](b_size)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](c_size)

    comptime k_dim = K.value()

    random(a_host)
    random(b_host)
    random(a_scales_host)
    random(b_scales_host)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    @parameter
    fn stride_from_shape[shape: DimList]() -> DimList:
        comptime assert len(shape) == 2, "rank must be 2"
        return DimList(shape.at[1](), 1)

    var a_ndbuffer = NDBuffer[
        in_dtype,
        2,
        ImmutAnyOrigin,
        static_a_shape,
        stride_from_shape[static_a_shape](),
    ](
        a_device.unsafe_ptr(),
        DimList(m, k),
    )
    var b_ndbuffer = NDBuffer[
        in_dtype,
        2,
        ImmutAnyOrigin,
        static_b_shape,
        stride_from_shape[static_b_shape](),
    ](
        b_device.unsafe_ptr(),
        DimList(n, k) if transpose_b else DimList(k, n),
    )
    var c_ndbuffer = NDBuffer[
        out_dtype, 2, _, static_c_shape, stride_from_shape[static_c_shape]()
    ](
        c_device.unsafe_ptr(),
        DimList(m, n),
    )
    var a_scales_ndbuffer = NDBuffer[
        scales_dtype,
        2,
        _,
        static_a_scales_shape,
        stride_from_shape[static_a_scales_shape](),
    ](
        a_scales_device.unsafe_ptr(),
        DimList(1, m),
    )
    var b_scales_ndbuffer = NDBuffer[
        scales_dtype,
        2,
        _,
        static_b_scales_shape,
        stride_from_shape[static_b_scales_shape](),
    ](
        b_scales_device.unsafe_ptr(),
        DimList(n, 1) if transpose_b else DimList(1, n),
    )
    var c_ref_ndbuffer = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(),
        DimList(m, n),
    )

    matmul_dynamic_scaled_fp8[
        input_scale_granularity="colwise",
        weight_scale_granularity="rowwise",
        m_scale_granularity=1,
        n_scale_granularity=1,
        k_scale_granularity=k_dim,
        transpose_b=transpose_b,
        target="gpu",
    ](
        c_ndbuffer,
        a_ndbuffer,
        b_ndbuffer,
        a_scales_ndbuffer,
        b_scales_ndbuffer,
        ctx,
    )
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk = Index(1, 1, k_dim),
    ](
        c_ref_ndbuffer,
        a_ndbuffer,
        b_ndbuffer,
        a_scales_ndbuffer,
        b_scales_ndbuffer,
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    for i in range(m):
        for j in range(n):
            assert_almost_equal(
                c_host[i, j][0].cast[DType.float32](),
                c_host_ref[i, j][0],
                msg="At [" + String(i) + ", " + String(j) + "]",
                atol=1.5e-2,
                rtol=1.5e-2,
            )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = a_scales_device^
    _ = b_scales_device^
    _ = c_device_ref^


def main():
    with DeviceContext() as ctx:
        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
            M=None,
            N = Int(256 + 256),
            K = Int(256),
        ](ctx, 17, 256 + 256, 256)

        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
            M=None,
            N = Int(512),
            K = Int(512),
        ](ctx, 124, 512, 512)

        # these tests are guaranteed to hit a mojo fp8 kernel in the dispatch table.
        # if the fp8 kernel is not registered, these tests will fail.
        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
            M=None,
            N = Int(5376),
            K = Int(4096),
        ](ctx, 3000, 5376, 4096)

        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
            M=None,
            N = Int(43008),
            K = Int(5376),
        ](ctx, 224, 43008, 5376)
