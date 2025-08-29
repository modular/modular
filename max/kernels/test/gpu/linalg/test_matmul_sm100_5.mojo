# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from math import align_up
from sys import size_of, argv
from hashlib import default_comp_time_hasher
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from linalg import vendor_blas
from linalg.matmul_sm100 import blackwell_matmul_tma_umma_warp_specialized

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from linalg.utils_gpu import MatmulConfig
from internal_utils._utils import ValOrDim, dynamic, static


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


fn simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_matmul_tma_umma_warp_specialized[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var M = m.value
    var N = n.value
    var K = k.value

    if not benchmark:
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
            )
        )

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    if simple_init():
        var at = a_host.tensor
        var bt = b_host.tensor
        for m in range(M):
            for k in range(K):
                at[m, k] = Float32(k).cast[a_type]()
        for n in range(N):
            for k in range(K):
                bt[n, k] = Float32(1 if n == k else 0).cast[b_type]()
    else:
        random(a_host.tensor)
        random(b_host.tensor)

    # Move operands to the Device
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    alias matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        mma_shape=mma_shape,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
    )

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        cta_group=2,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        ctx,
    )

    if benchmark:
        alias num_runs = 100
        alias num_warmup = 100

        @always_inline
        @parameter
        fn run_kernel(ctx: DeviceContext) raises:
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=matmul_config,
                cta_group=2,
            ](
                c_device.tensor,
                a_device.tensor,
                b_device.tensor,
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()

        var nstime = ctx.execution_time[run_kernel](num_runs) / num_runs
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        # Round TFLOPS to two decimal places for cleaner output
        var tflops = TFlop / sectime
        var tflops_rounded = round(tflops, 2)
        print(
            String(a_type, "x", M, "x", N, "x", K),
            sectime * 1000,
            tflops_rounded,
        )
    else:
        constrained[
            a_type != DType.float8_e4m3fn or transpose_b,
            (
                "Testing is only supported for transposed_b==True when"
                " a_type==float8_e4m3fn. Add the non-transposed case if needed."
            ),
        ]()

        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
        ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
        ctx.synchronize()

        alias rtol = 1e-2
        assert_almost_equal(
            c_host.tensor,
            c_host_ref.tensor,
            atol=0.0001,
            rtol=rtol,
        )
        print("\n=== TEST PASSED ===\n")

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


fn get_shapes_dict(
    index: Int, shapes_dict: Dict[Int, Tuple[Int, Int, Int], *_, **_]
) -> Tuple[Int, Int, Int]:
    try:
        return shapes_dict[index]
    except error:
        print("error")
        return (128, 128, 128)


fn make_shapes_dict() -> (
    Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]
):
    var dic: Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher] = {
        0: (4096, 4096, 4096),
        1: (512, 2560, 8192),
        2: (512, 8192, 2048),
        3: (512, 14336, 8192),
        4: (512, 8192, 7168),
        5: (4096, 2560, 8192),
        6: (4096, 8192, 2048),
        7: (4096, 14336, 8192),
        8: (4096, 8192, 7168),
        9: (8192, 2560, 8192),
        10: (8192, 8192, 2048),
        11: (8192, 14336, 8192),
        12: (8192, 8192, 8192),
    }
    return dic


fn benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    @parameter
    for dtype in [DType.bfloat16, DType.float8_e4m3fn]:

        @parameter
        for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
            print("Benchmarking blackwell_matmul_tma_umma_kernel")
            print("============================================")
            print("dtype, M, N, K, time(ms), TFLOPS")

            alias BK = (swizzle.bytes() // size_of[dtype]())
            alias block_tile_shape = Index(128, 128, BK)
            alias MMA_K = 32 if dtype == DType.float8_e4m3fn else 16
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )

            alias c_type = DType.bfloat16
            alias shapes_dict = make_shapes_dict()

            @parameter
            for i in range(len(shapes_dict)):
                alias shape = get_shapes_dict(i, shapes_dict)
                with vendor_blas.Handle[
                    vendor_blas.Backend.CUBLAS
                ]() as cublas_handle:
                    try:
                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            c_type,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            benchmark=True,
                        ](
                            ctx,
                            dynamic(shape[0]),
                            static[shape[1]](),
                            static[shape[2]](),
                        )
                    except error:
                        print("error")


def main():
    with DeviceContext() as ctx:
        if is_benchmark():
            benchmark_blackwell_matmul(ctx)
            return

        @parameter
        for dtype in [DType.bfloat16, DType.float8_e4m3fn]:

            @parameter
            for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
                alias BK = (swizzle.bytes() // size_of[dtype]())
                alias MMA_K = 32 if dtype == DType.float8_e4m3fn else 16

                @parameter
                for mma_m_scale in range(1, 3):

                    @parameter
                    for mma_n_scale in range(1, 17):
                        # from 16*1 till 16*16 which is 256
                        # basically, if MMA_M is 64, then BN must be multiple of 16 (mma_n_scale must be even)
                        @parameter
                        if mma_m_scale == 1 and mma_n_scale % 2 != 0:
                            continue
                        # TODO: support the increments of 8 for float 8 dtype at a later point
                        # currently it works with increments of BN = 32
                        if (
                            dtype == DType.float8_e4m3fn
                            and mma_n_scale % 4 != 0
                        ):
                            continue
                        alias block_tile_shape = Index(
                            64 * mma_m_scale, 8 * mma_n_scale, BK
                        )
                        alias umma_shape = Index(
                            128 * mma_m_scale, 16 * mma_n_scale, MMA_K
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                        ](
                            ctx,
                            dynamic(1000),
                            static[1024](),
                            static[1024](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                        ](
                            ctx,
                            dynamic(512),
                            static[4096](),
                            static[1024](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                        ](
                            ctx,
                            dynamic(500),
                            static[2048](),
                            static[4096](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](8, 2, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                        ](
                            ctx,
                            dynamic(1024),
                            static[256](),
                            static[128](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](2, 2, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                        ](
                            ctx,
                            static[1024](),
                            static[1024](),
                            static[2048](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                        ](
                            ctx,
                            dynamic(8192),
                            static[2560](),
                            static[8192](),
                        )
