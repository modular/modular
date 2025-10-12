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

from hashlib import default_comp_time_hasher
from math import align_up
from sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.matmul.gpu.sm100.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.utils_gpu import MatmulConfig

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


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
    block_swizzle_size: Int = 0,
    benchmark: Bool = False,
    swapAB: Bool = False,
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
                " swapAB=",
                swapAB,
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
        block_swizzle_size=block_swizzle_size,
        swapAB=swapAB,
    ](
        c_device.to_layout_tensor(),
        a_device.to_layout_tensor(),
        b_device.to_layout_tensor(),
        ctx,
    )

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


def main():
    with DeviceContext() as ctx:
        alias dtype = DType.float8_e4m3fn

        @parameter
        for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
            alias BK = (swizzle.bytes() // size_of[dtype]())
            alias MMA_K = 32

            @parameter
            for mma_m_scale in range(1, 3):

                @parameter
                for mma_n_scale in range(1, 17):
                    # from 16*1 till 16*16 which is 256
                    # basically, if MMA_M is 64, then BN must be multiple of 16 (mma_n_scale must be even)
                    @parameter
                    if mma_m_scale == 1 and mma_n_scale % 2 != 0:
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
                        block_swizzle_size=8,
                    ](
                        ctx,
                        dynamic(1000),
                        static[1024](),
                        static[1024 + 16](),
                    )

                    @parameter
                    for swapAB in [False, True]:

                        @parameter
                        if swapAB and mma_m_scale != 2:
                            continue

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=4,
                            swapAB=swapAB,
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
                            block_swizzle_size=0,
                            swapAB=swapAB,
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
                        block_swizzle_size=2,
                    ](
                        ctx,
                        dynamic(999),
                        static[256](),
                        static[128](),
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
                        block_swizzle_size=1,
                    ](
                        ctx,
                        dynamic(777),
                        static[2560](),
                        static[8192](),
                    )
