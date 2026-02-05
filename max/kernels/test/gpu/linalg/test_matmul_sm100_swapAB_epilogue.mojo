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
from collections import Optional
from random import random_si64, shuffle, seed
from sys import align_of, size_of, argv

import linalg.matmul.vendor.blas as vendor_blas
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from internal_utils import assert_almost_equal
from random import rand
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from linalg.utils import elementwise_compute_lambda_type

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test_matmul_sm100_epilogue[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
    test_lambda_fn: Bool = False,
    register_based_epilogue: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
](
    ctx: DeviceContext,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    is_benchmark: Bool = False,
):
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        String(
            "in/out dtypes=(",
            a_type,
            ", ",
            b_type,
            ", ",
            c_type,
            ") ",
            " problem shape=(",
            M,
            ", ",
            N,
            ", ",
            K,
            ") ",
            "mma_shape=",
            mma_shape,
            " block_tile_shape=",
            block_tile_shape,
            " register_based_epilogue=",
            register_based_epilogue,
            " swapAB=",
            swapAB,
            " k_group_size=",
            k_group_size,
        )
    )

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_size = m.value * k.value
    var b_size = n.value * k.value
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_copy_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)

    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )
    var c_host_copy = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_copy_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)

    var a_device_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    var c_tensor = c_device_nd

    @parameter
    @always_inline
    @__copy_capture(c_tensor)
    fn test_lambda_add_coords_prod[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        # this function helps us determine if the provided indexes are correct
        # while also testing arithmetic operations
        var x = c_tensor.load[width=width](idx).cast[_dtype]()
        var y = val * x
        return y

    seed(1234)
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

    var scales: List[Int32] = [-2, -1, 0, 1, 2]

    for i in range(M):
        for j in range(N):
            shuffle(scales)
            c_host[i, j] = Scalar[c_type](scales[0])
            c_host_copy[i, j] = c_host[i, j]

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
    )

    comptime optional_lambda_fn = Optional[elementwise_compute_lambda_type](
        test_lambda_add_coords_prod
    ) if test_lambda_fn else None

    var c_dev = from_ndbuffer_row_major(c_device_nd)
    var a_dev = from_ndbuffer_row_major(a_device_nd)
    var b_dev = from_ndbuffer_row_major(b_device_nd)

    @parameter
    @always_inline
    @__copy_capture(c_dev, a_dev, b_dev)
    fn kernel_launch(ctx: DeviceContext) raises:
        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=matmul_config,
            elementwise_compute_lambda_fn=optional_lambda_fn,
            register_based_epilogue=register_based_epilogue,
        ](c_dev, a_dev, b_dev, ctx)

    if is_benchmark:
        comptime nrun = 50

        # Warmup
        kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time[kernel_launch](nrun)) / Float64(
            nrun
        )
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")
    else:
        kernel_launch(ctx)

    if not is_benchmark:
        comptime assert a_type != DType.float8_e4m3fn or transpose_b, (
            "Testing is only supported for transposed_b==True when"
            " a_type==float8_e4m3fn. Add the non-transposed case if needed."
        )

        vendor_blas.matmul(
            ctx,
            c_device_ref_nd,
            a_device_nd,
            b_device_nd,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host_ptr, c_device)
        ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
        ctx.synchronize()

        var c_tensor_host = c_host_copy

        @parameter
        @always_inline
        @__copy_capture(c_tensor_host)
        fn test_lambda_add_coords_prod_local[
            _dtype: DType,
            width: Int,
            *,
            alignment: Int = align_of[SIMD[_dtype, width]](),
        ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
            _dtype, width
        ]:
            return val * c_tensor_host.load[width=width](idx).cast[_dtype]()

        @parameter
        if optional_lambda_fn:
            # Apply the compute lambda directly on the reference tensor
            # alias compute_lambda = elementwise_compute_lambda_fn.value()
            for i in range(M):
                for j in range(N):
                    c_host_ref[Index(i, j)] = test_lambda_add_coords_prod_local(
                        IndexList[2](i, j),
                        c_host_ref[Index(i, j)],
                    )

        comptime rtol = 1e-2
        assert_almost_equal(
            c_host.data,
            c_host_ref.data,
            c_host.num_elements(),
            atol=0.0001,
            rtol=rtol,
        )

        print("\n=== TEST PASSED ===\n")

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    c_host_copy_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main():
    comptime dtype = DType.bfloat16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]())
    comptime MMA_K = 16
    var is_bench = is_benchmark()

    with DeviceContext() as ctx:

        @parameter
        for register_based_epilogue in [True, False]:
            # swapAB with epilogue tests (2SM)
            @parameter
            for mma_m_scale in range(1, 3):

                @parameter
                for mma_n_scale in range(1, 17):
                    comptime block_tile_shape = Index(
                        64 * mma_m_scale, 8 * mma_n_scale, BK
                    )
                    comptime umma_shape = Index(
                        128 * mma_m_scale, 16 * mma_n_scale, MMA_K
                    )

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=2,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                        k_group_size=2,
                    ](
                        ctx,
                        dynamic(100),
                        static[2560](),
                        static[8192](),
                        is_benchmark=is_bench,
                    )

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=2,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                    ](
                        ctx,
                        dynamic(17),
                        static[1024](),
                        static[1024](),
                        is_benchmark=is_bench,
                    )

            # swapAB with epilogue tests (1SM)
            # we support all range of mma_n_scales in range(1, 33) but the test will time out so we only test a subset
            @parameter
            for mma_m in [64, 128]:

                @parameter
                for mma_n in [8, 16, 32, 40, 48, 64, 88, 104, 128]:
                    comptime block_tile_shape = Index(mma_m, mma_n, BK)
                    comptime umma_shape = Index(mma_m, mma_n, MMA_K)

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=1,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                        k_group_size=2,
                    ](
                        ctx,
                        dynamic(1000),
                        static[1024](),
                        static[1024](),
                        is_benchmark=is_bench,
                    )

                    test_matmul_sm100_epilogue[
                        dtype,
                        dtype,
                        DType.bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 2, 1),
                        cta_group=1,
                        test_lambda_fn=True,
                        register_based_epilogue=register_based_epilogue,
                        swapAB=True,
                    ](
                        ctx,
                        dynamic(512),
                        static[4096](),
                        static[1024 + 16](),
                        is_benchmark=is_bench,
                    )
