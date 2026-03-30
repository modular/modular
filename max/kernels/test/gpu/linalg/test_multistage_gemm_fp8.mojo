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
from std.gpu import grid_dim
from std.gpu.host import DeviceContext, FuncAttribute
from internal_utils import assert_almost_equal
from layout import LTToTTLayout, lt_to_tt
from layout.layout import *
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import MatmulKernels
from layout import TileTensor, row_major


def test_fp8_multistage_gemm[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    /,
    *,
    transpose_b: Bool = False,
](ctx: DeviceContext) raises:
    print("test fp8 multistage matmul")

    comptime a_size = M * K
    comptime b_size_0 = N if transpose_b else K
    comptime b_size_1 = K if transpose_b else N
    comptime b_size = b_size_0 * b_size_1
    comptime c_size = M * N

    var a_host_ptr = alloc[Scalar[dtype]](a_size)
    var b_host_ptr = alloc[Scalar[dtype]](b_size)
    var c_host_ptr = alloc[Scalar[DType.float32]](c_size)
    var c_host_ref_ptr = alloc[Scalar[DType.float32]](c_size)

    var a_host = TileTensor(a_host_ptr, row_major[M, K]())
    var b_host = TileTensor(
        b_host_ptr,
        row_major[N if transpose_b else K, K if transpose_b else N](),
    )
    var c_host = TileTensor(c_host_ptr, row_major[M, N]())
    var c_host_ref = TileTensor(c_host_ref_ptr, c_host.layout)

    comptime for i in range(M):
        comptime for j in range(K):
            a_host[i, j] = Scalar[dtype](i + j)

    comptime for i in range(b_host.static_shape[0]):
        comptime for j in range(b_host.static_shape[1]):
            b_host[i, j] = Scalar[dtype](i + j)

    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    var a_device = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device = ctx.enqueue_create_buffer[DType.float32](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](c_size)

    var a_device_nd = TileTensor(a_device, a_host.layout)
    var b_device_nd = TileTensor(b_device, b_host.layout)
    var c_device_nd = TileTensor(c_device, c_host.layout)
    var c_device_ref_nd = TileTensor(c_device_ref, c_host.layout)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    var c_tensor = c_device_nd.to_layout_tensor()
    var a_tensor = a_device_nd.to_layout_tensor()
    var b_tensor = b_device_nd.to_layout_tensor()

    var c_tt = lt_to_tt(c_tensor)
    var a_tt = lt_to_tt(a_tensor).as_immut()
    var b_tt = lt_to_tt(b_tensor).as_immut()

    comptime kernels = MatmulKernels[dtype, dtype, DType.float32, transpose_b]()
    comptime config = kernels.hopper_128x128_4

    comptime kernel = multistage_gemm_kernel[
        DType.float32,  # c_type
        LTToTTLayout[c_tensor.layout],
        dtype,  # a_type
        LTToTTLayout[a_tensor.layout],
        dtype,  # b_type
        LTToTTLayout[b_tensor.layout],
        transpose_b,
        c_linear_idx_type=c_tensor.linear_idx_type,
        a_linear_idx_type=a_tensor.linear_idx_type,
        b_linear_idx_type=b_tensor.linear_idx_type,
        config=config,
    ]

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]

    ctx.enqueue_function_experimental[kernel](
        c_tt,
        a_tt,
        b_tt,
        grid_dim=config.grid_dim(UInt(M), UInt(N)),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
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
        var b_host_col_major_ptr = alloc[Scalar[dtype]](b_col_major_size)
        var b_host_col_major = TileTensor(
            b_host_col_major_ptr, row_major[N, K]()
        )

        for i in range(N):
            for j in range(K):
                b_host_col_major[i, j] = b_host[j, i]

        var b_device_col_major = ctx.enqueue_create_buffer[dtype](
            b_col_major_size
        )
        var b_device_col_major_nd = TileTensor(
            b_device_col_major, row_major[N, K]()
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
        c_host.ptr,
        c_host_ref.ptr,
        c_host.num_elements(),
        atol=0.0001,
        rtol=0.01,
    )

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()


def main() raises:
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
