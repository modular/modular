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

from math import align_up, ceildiv
from sys import size_of, argv
from hashlib import default_comp_time_hasher
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from linalg import vendor_blas
from linalg.matmul_sm100_warp_specialized_blockwise_fp8 import (
    sm100_warp_specialized_blockwise_fp8,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    assert_with_measure,
    random,
    zero,
)
from linalg.utils_gpu import MatmulConfig
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils._measure import relative_difference


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


fn test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    scales_type: DType = DType.float32,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises -> Float64:
    var M = m.value
    var N = n.value
    var K = k.value

    alias BLOCK_SCALE_K = 128

    if M * size_of[DType.float32]() % 16 != 0:
        raise Error("TMA expects M to be divisible by 16 bytes")

    if not benchmark:
        if N % BLOCK_SCALE_K != 0:
            raise Error("N must be divisible by BLOCK_SCALE_K")

    if K % BLOCK_SCALE_K != 0:
        raise Error("K must be divisible by BLOCK_SCALE_K")

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

    alias static_a_scales_shape = DimList(
        ceildiv(Int(k.dim), BLOCK_SCALE_K), m.dim
    )
    alias static_b_scales_shape = DimList(
        ceildiv(Int(n.dim), BLOCK_SCALE_K), ceildiv(Int(k.dim), BLOCK_SCALE_K)
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(k.value, BLOCK_SCALE_K), m.value
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, BLOCK_SCALE_K), ceildiv(k.value, BLOCK_SCALE_K)
    )

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

    var a_scales_host = HostNDBuffer[scales_type, 2, static_a_scales_shape](
        dynamic_a_scales_shape
    )
    var b_scales_host = HostNDBuffer[scales_type, 2, static_b_scales_shape](
        dynamic_b_scales_shape
    )

    var a_scales_device = DeviceNDBuffer[scales_type, 2, static_a_scales_shape](
        dynamic_a_scales_shape, ctx=ctx
    )
    var b_scales_device = DeviceNDBuffer[scales_type, 2, static_b_scales_shape](
        dynamic_b_scales_shape, ctx=ctx
    )

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Initialize matmul operands
    if simple_init():
        var at = a_host.tensor
        var bt = b_host.tensor
        for m in range(M):
            for k in range(K):
                at[m, k] = Scalar[a_type](1.0)
        for n in range(N):
            for k in range(K):
                bt[n, k] = Scalar[b_type](1.0)

        for m in range(M):
            for k in range(K):
                a_scales_host.tensor[k // BLOCK_SCALE_K, m] = Scalar[
                    scales_type
                ](0.5)
        for n in range(N):
            for k in range(K):
                b_scales_host.tensor[
                    n // BLOCK_SCALE_K, k // BLOCK_SCALE_K
                ] = Scalar[scales_type](0.5)

    else:
        random(a_host.tensor)
        random(b_host.tensor)
        random(a_scales_host.tensor)
        random(b_scales_host.tensor)

    # Move operands to the Device
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)
    var a_scales = from_ndbuffer_row_major(a_scales_device.tensor)
    var b_scales = from_ndbuffer_row_major(b_scales_device.tensor)

    alias matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        mma_shape=mma_shape,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
    )

    sm100_warp_specialized_blockwise_fp8[
        transpose_b=transpose_b,
        config=matmul_config,
        cta_group=2,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        ctx,
    )

    var tflops_rounded: Float64 = 0.0
    if benchmark:
        alias num_runs = 100
        alias num_warmup = 100

        @always_inline
        @parameter
        fn run_kernel(ctx: DeviceContext) raises:
            sm100_warp_specialized_blockwise_fp8[
                transpose_b=transpose_b,
                config=matmul_config,
                cta_group=2,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
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
        tflops_rounded = round(tflops, 2)

    else:
        constrained[
            a_type != DType.float8_e4m3fn or transpose_b,
            (
                "Testing is only supported for transposed_b==True when"
                " a_type==float8_e4m3fn. Add the non-transposed case if needed."
            ),
        ]()

        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
        ](
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            a_scales_device.tensor,
            b_scales_device.tensor,
            ctx,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
        ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
        ctx.synchronize()

        assert_with_measure[relative_difference](
            c_host.tensor, c_host_ref.tensor, threshold=0.001
        )

        assert_almost_equal(
            c_host.tensor,
            c_host_ref.tensor,
            atol=1e-2,
            rtol=1e-2,
        )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device
    _ = a_scales_device
    _ = b_scales_device
    _ = a_scales_host
    _ = b_scales_host

    return tflops_rounded


fn get_shapes_dict(
    index: Int, shapes_dict: Dict[Int, Tuple[Int, Int], *_, **_]
) -> Tuple[Int, Int]:
    try:
        return shapes_dict[index]
    except error:
        print("error")
        return (128, 128)


# DeepSeek Test Shapes n, k in [(2112, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]:
fn make_shapes_dict() -> Dict[Int, Tuple[Int, Int], default_comp_time_hasher]:
    var dic: Dict[Int, Tuple[Int, Int], default_comp_time_hasher] = {
        0: (2112, 7168),
        1: (24576, 1536),
        2: (32768, 512),
        3: (7168, 16384),
        4: (4096, 7168),
        5: (7168, 2048),
    }
    return dic^


fn benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    alias dtype = DType.float8_e4m3fn
    alias swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias BK = (swizzle.bytes() // size_of[dtype]())
    alias MMA_K = 32

    alias c_type = DType.bfloat16
    alias shapes_dict = make_shapes_dict()

    print("Benchmarking warp-specialized blockwise fp8 (fp32 scalers)")
    print("============================================")
    print("dtype, M, N, K, time(ms), TFLOPS")

    @parameter
    for i in range(len(shapes_dict)):
        alias shape_nk = get_shapes_dict(i, shapes_dict)

        for m in [128, 4096]:
            var max_flops: Float64 = 0.0

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
                    if mma_n_scale % 4 != 0:
                        continue
                    alias block_tile_shape = Index(
                        64 * mma_m_scale, 8 * mma_n_scale, BK
                    )
                    alias umma_shape = Index(
                        128 * mma_m_scale, 16 * mma_n_scale, MMA_K
                    )

                    try:
                        max_flops = max(
                            max_flops,
                            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
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
                                dynamic(m),
                                static[shape_nk[0]](),
                                static[shape_nk[1]](),
                            ),
                        )
                    except error:
                        print("error")

            print(
                String(m, " x ", shape_nk[0], " x ", shape_nk[1]),
                max_flops,
            )


def main():
    with DeviceContext() as ctx:
        if is_benchmark():
            benchmark_blackwell_matmul(ctx)
            return

        alias swizzle = TensorMapSwizzle.SWIZZLE_128B
        alias in_dtype = DType.float8_e4m3fn
        alias BK = (swizzle.bytes() // size_of[in_dtype]())
        alias MMA_K = 32
        alias out_dtype = DType.bfloat16

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
                if mma_n_scale % 4 != 0:
                    continue

                alias block_tile_shape = Index(
                    64 * mma_m_scale, 8 * mma_n_scale, BK
                )
                alias umma_shape = Index(
                    128 * mma_m_scale, 16 * mma_n_scale, MMA_K
                )

                print(
                    "block_tile_shape",
                    block_tile_shape,
                    "umma_shape",
                    umma_shape,
                )

                _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                    in_dtype,
                    in_dtype,
                    out_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    scales_type = DType.bfloat16,
                ](
                    ctx,
                    dynamic(1000),
                    static[512](),
                    static[512 + 128](),
                )

                _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                    in_dtype,
                    in_dtype,
                    out_dtype,
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

                _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                    in_dtype,
                    in_dtype,
                    out_dtype,
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

                _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                    in_dtype,
                    in_dtype,
                    out_dtype,
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

                _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                    in_dtype,
                    in_dtype,
                    out_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](2, 2, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    scales_type = DType.bfloat16,
                ](
                    ctx,
                    static[1024](),
                    static[1024](),
                    static[2048](),
                )

                _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                    in_dtype,
                    in_dtype,
                    out_dtype,
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
