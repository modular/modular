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


import linalg.matmul.vendor.blas as vendor_blas
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import grid_dim
from gpu.host import DeviceContext, FuncAttribute
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from internal_utils import assert_almost_equal
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout import *
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import MatmulKernels


fn test_fp8_multistage_gemm[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    /,
    *,
    transpose_b: Bool = False,
](ctx: DeviceContext) raises:
    print("test fp8 multistage matmul")

    comptime static_a_shape = DimList(M, K)
    comptime static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    comptime static_c_shape = DimList(M, N)

    comptime a_size = M * K
    comptime b_size_0 = N if transpose_b else K
    comptime b_size_1 = K if transpose_b else N
    comptime b_size = b_size_0 * b_size_1
    comptime c_size = M * N

    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[DType.float32]].alloc(c_size)

    var a_host = NDBuffer[dtype, 2, _, static_a_shape](a_host_ptr)
    var b_host = NDBuffer[dtype, 2, _, static_b_shape](b_host_ptr)
    var c_host = NDBuffer[DType.float32, 2, _, static_c_shape](c_host_ptr)
    var c_host_ref = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_host_ref_ptr
    )

    comptime for i in range(M):
        comptime for j in range(K):
            a_host[i, j] = i + j

    comptime for i in range(static_b_shape.get[0]()):
        comptime for j in range(static_b_shape.get[1]()):
            b_host[i, j] = i + j

    c_host.zero()
    c_host_ref.zero()

    var a_device = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device = ctx.enqueue_create_buffer[DType.float32](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](c_size)

    var a_device_nd = NDBuffer[dtype, 2, _, static_a_shape](
        a_device.unsafe_ptr()
    )
    var b_device_nd = NDBuffer[dtype, 2, _, static_b_shape](
        b_device.unsafe_ptr()
    )
    var c_device_nd = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_device.unsafe_ptr()
    )
    var c_device_ref_nd = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr()
    )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    var c_tensor = from_ndbuffer_row_major(c_device_nd)
    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var b_tensor = from_ndbuffer_row_major(b_device_nd)

    comptime kernels = MatmulKernels[dtype, dtype, DType.float32, transpose_b]()
    comptime config = kernels.hopper_128x128_4

    comptime kernel = multistage_gemm_kernel[
        DType.float32,  # c_type
        c_tensor.layout,
        dtype,  # a_type
        a_tensor.layout,
        dtype,  # b_type
        b_tensor.layout,
        transpose_b,
        c_layout_int_type = c_tensor.layout_int_type,
        c_linear_idx_type = c_tensor.linear_idx_type,
        a_layout_int_type = a_tensor.layout_int_type,
        a_linear_idx_type = a_tensor.linear_idx_type,
        b_layout_int_type = b_tensor.layout_int_type,
        b_linear_idx_type = b_tensor.linear_idx_type,
        config=config,
    ]

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]

    ctx.enqueue_function_experimental[kernel](
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=config.grid_dim(UInt(M), UInt(N)),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    comptime if transpose_b:
        vendor_blas.matmul(
            ctx,
            c_device_ref_nd,
            a_device_nd,
            b_device_nd,
            c_row_major=True,
            transpose_b=transpose_b,
        )

    else:
        # TODO: Matrix B should always be in col-major layout for cublasLt to work
        comptime b_col_major_size = N * K
        var b_host_col_major_ptr = UnsafePointer[Scalar[dtype]].alloc(
            b_col_major_size
        )
        var b_host_col_major = NDBuffer[dtype, 2, _, DimList(N, K)](
            b_host_col_major_ptr
        )

        for i in range(N):
            for j in range(K):
                b_host_col_major[i, j] = b_host[j, i]

        var b_device_col_major = ctx.enqueue_create_buffer[dtype](
            b_col_major_size
        )
        var b_device_col_major_nd = NDBuffer[dtype, 2, _, DimList(N, K)](
            b_device_col_major.unsafe_ptr()
        )
        ctx.enqueue_copy(b_device_col_major, b_host_col_major_ptr)

        vendor_blas.matmul(
            ctx,
            c_device_ref_nd,
            a_device_nd,
            b_device_col_major_nd,
            c_row_major=False,
            transpose_b=True,
        )

        b_host_col_major_ptr.free()
        _ = b_device_col_major^

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.0001,
        rtol=0.01,
    )

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor


def main():
    with DeviceContext() as ctx:
        test_fp8_multistage_gemm[
            DType.float8_e4m3fn, 128, 128, 64, transpose_b=True
        ](ctx)
        test_fp8_multistage_gemm[
            DType.float8_e4m3fn, 128, 128, 128, transpose_b=True
        ](ctx)

    # FIXME: KERN-1480
    # test_fp8_multistage_gemm[
    # DType.float8_e4m3fn, 128, 128, 64, transpose_b=False
    # ](ctx)
    # test_fp8_multistage_gemm[
    # DType.float8_e4m3fn, 128, 128, 128, transpose_b=False
    # ](ctx)
