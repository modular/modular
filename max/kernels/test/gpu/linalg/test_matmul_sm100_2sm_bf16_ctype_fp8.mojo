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

from std.hashlib import default_comp_time_hasher
from std.math import align_up
from std.sys import argv, size_of
import std.itertools
import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from internal_utils import assert_almost_equal
from std.random import rand
from internal_utils._utils import ValOrDim, dynamic, static
from layout.tile_tensor import TileTensor
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from std.testing import assert_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


def simple_init() -> Bool:
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
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    accum_dtype: DType = DType.float32,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        t"in/out dtypes=({a_type}, {b_type}, {c_type})  problem shape=({M},"
        t" {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape} cta_group={cta_group} cluster_shape=({cluster_shape[0]},"
        t" {cluster_shape[1]}, {cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size}"
    )

    comptime static_a_shape = DimList[m.dim, k.dim]()
    comptime static_b_shape = DimList[
        n.dim if transpose_b else k.dim, k.dim if transpose_b else n.dim
    ]()
    comptime static_c_shape = DimList[m.dim, n.dim]()
    var dynamic_a_shape = IndexList[2](m.value, k.value)
    var dynamic_b_shape = IndexList[2](
        n.value, k.value
    ) if transpose_b else IndexList[2](k.value, n.value)
    var dynamic_c_shape = IndexList[2](m.value, n.value)
    var a_size = m.value * k.value
    var b_size = n.value * k.value if transpose_b else k.value * n.value
    var c_size = m.value * n.value

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_host = NDBuffer[rank=2, a_type, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_host = NDBuffer[rank=2, b_type, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_host = NDBuffer[rank=2, c_type, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = alloc[Scalar[accum_dtype]](c_size)
    var c_host_ref = NDBuffer[rank=2, accum_dtype, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[rank=2, a_type, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[rank=2, b_type, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[rank=2, c_type, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[accum_dtype](c_size)
    var c_device_ref_nd = NDBuffer[rank=2, accum_dtype, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    # Initialize matmul operands
    if simple_init():
        for m_idx in range(M):
            for k_idx in range(K):
                a_host[m_idx, k_idx] = Float32(m_idx + k_idx).cast[a_type]()
        for n_idx in range(N):
            for k_idx in range(K):
                b_host[n_idx, k_idx] = Float32(n_idx + k_idx).cast[b_type]()
    else:
        rand(a_host.data, a_host.num_elements(), min=-1.0, max=1.0)
        rand(b_host.data, b_host.num_elements(), min=-1.0, max=1.0)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
    )

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        TileTensor(c_device_nd),
        TileTensor(a_device_nd),
        TileTensor(b_device_nd),
        ctx,
    )

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

    var c_host_tensor = TileTensor(c_host)
    comptime assert c_host_tensor.flat_rank == 2

    var c_host_ref_tensor = TileTensor(c_host_ref)
    comptime assert c_host_ref_tensor.flat_rank == 2

    for i in range(c_host_ref_tensor.dim[0]()):
        for j in range(c_host_ref_tensor.dim[1]()):
            assert_equal(
                c_host_tensor[i, j].cast[DType.float64](),
                c_host_ref_tensor[i, j].cast[c_type]().cast[DType.float64](),
                msg="At [" + String(i) + ", " + String(j) + "]",
            )

    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16

        comptime for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
            comptime BK = (swizzle.bytes() // size_of[dtype]())
            comptime MMA_K = 16
            comptime cta_group = 2

            # we support all range of mma_n_scales in range(1, 33) but the test will time out so we only test a subset
            comptime for bm in [64, 128]:
                comptime for bn in range(8, 128 + 1, 8):
                    comptime block_tile_shape = Index(bm, bn, BK)
                    comptime umma_shape = Index(
                        cta_group * bm, cta_group * bn, MMA_K
                    )

                    # Output TMA expects mma_n to be divisible by 16 for FP8 output
                    comptime if umma_shape[0] == 128 and umma_shape[
                        1
                    ] % 32 != 0:
                        continue

                    comptime for swapAB in [True, False]:
                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.float8_e4m3fn,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=8,
                            swapAB=swapAB,
                        ](
                            ctx,
                            dynamic(64),
                            static[64](),
                            static[1024 + 16](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.float8_e4m3fn,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=4,
                            swapAB=swapAB,
                        ](
                            ctx,
                            dynamic(512),
                            static[4096](),
                            static[1024 + 16](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.float8_e4m3fn,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=0,
                            k_group_size=2,
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
                            DType.float8_e4m3fn,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=2,
                            swapAB=swapAB,
                        ](
                            ctx,
                            dynamic(999),
                            static[256](),
                            static[128](),
                        )

                        test_blackwell_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            DType.float8_e4m3fn,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=1,
                            swapAB=swapAB,
                        ](
                            ctx,
                            dynamic(777),
                            static[2560](),
                            static[8192](),
                        )
