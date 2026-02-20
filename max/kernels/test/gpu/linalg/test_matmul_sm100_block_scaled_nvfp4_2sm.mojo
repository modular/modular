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
from hashlib import default_comp_time_hasher
from math import align_up
from sys import argv, size_of
import itertools
import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from memory import LegacyUnsafePointer
from random import rand

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import assert_almost_equal
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    lt_to_tt,
)
from linalg.matmul.gpu.sm100_structured.block_scaled.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from math import ceildiv, align_up
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    set_scale_factor,
)
from random import random_ui64
from builtin.simd import _convert_f32_to_float8_ue8m0
from layout import LayoutTensor, Layout, RuntimeLayout
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


fn simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
](
    ctx: DeviceContext,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    alpha: Float32 = 1.0,
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
            ", ",
            scales_dtype,
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
            " cta_group=",
            cta_group,
            " cluster_shape=(",
            cluster_shape[0],
            ", ",
            cluster_shape[1],
            ", ",
            cluster_shape[2],
            ")",
            " swapAB=",
            swapAB,
            " k_group_size=",
            k_group_size,
            " SF_VECTOR_SIZE=",
            SF_VECTOR_SIZE,
            " alpha=",
            alpha,
        )
    )

    comptime static_a_shape = DimList(m.dim, k.dim // 2)
    comptime static_b_shape = DimList(n.dim, k.dim // 2)
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value // 2)
    var dynamic_b_shape = DimList(n.value, k.value // 2)
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_size = m.value * (k.value // 2)
    var b_size = n.value * (k.value // 2)
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    comptime static_a_scales_shape = DimList(
        ceildiv(m.dim, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    comptime static_b_scales_shape = DimList(
        ceildiv(n.dim, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(m.value, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var a_scales_total = (
        ceildiv(m.value, SF_MN_GROUP_SIZE)
        * ceildiv(k.value, SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_total = (
        ceildiv(n.value, SF_MN_GROUP_SIZE)
        * ceildiv(k.value, SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        a_scales_total
    )
    var a_scales_host = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        a_scales_host_ptr, dynamic_a_scales_shape
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        b_scales_total
    )
    var b_scales_host = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        b_scales_host_ptr, dynamic_b_scales_shape
    )

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_device_nd = NDBuffer[
        scales_dtype, 5, _, static_a_scales_shape
    ](a_scales_device.unsafe_ptr(), dynamic_a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_device_nd = NDBuffer[
        scales_dtype, 5, _, static_b_scales_shape
    ](b_scales_device.unsafe_ptr(), dynamic_b_scales_shape)

    # LayoutTensors for reference matmul and scale factor reshaping
    var a_lt = from_ndbuffer_row_major(a_device_nd)
    var b_lt = from_ndbuffer_row_major(b_device_nd)
    var c_lt = from_ndbuffer_row_major(c_device_nd)
    var a_scales_lt = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales_lt = from_ndbuffer_row_major(b_scales_device_nd)
    var c_ref_tensor = from_ndbuffer_row_major(c_device_ref_nd)

    # TileTensors for the kernel under test
    var a_tensor = lt_to_tt(a_lt)
    var b_tensor = lt_to_tt(b_lt)
    var c_tensor = lt_to_tt(c_lt)
    var a_scales_tensor = lt_to_tt(a_scales_lt)
    var b_scales_tensor = lt_to_tt(b_scales_lt)

    # Initialize matmul operands
    if simple_init():
        for m in range(M):
            for k in range(K // 2):
                a_host[m, k] = UInt8(m).cast[a_type]()
        for n in range(N):
            for k in range(K // 2):
                b_host[n, k] = UInt8(n).cast[b_type]()
    else:
        rand(a_host.data, a_host.num_elements(), min=0, max=255)
        rand(b_host.data, b_host.num_elements(), min=0, max=255)

    comptime scales_5d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime a_scales_5d_layout = scales_5d_layout[a_scales_lt.layout]
    comptime b_scales_5d_layout = scales_5d_layout[b_scales_lt.layout]

    var a_scales_tensor_host = LayoutTensor[
        scales_dtype, a_scales_5d_layout, MutAnyOrigin
    ](
        a_scales_host_ptr,
        RuntimeLayout[a_scales_5d_layout].row_major(
            IndexList[5](
                a_scales_host.dim(0),
                a_scales_host.dim(1),
                a_scales_host.dim(2),
                a_scales_host.dim(3),
                a_scales_host.dim(4),
            ),
        ),
    )

    var b_scales_tensor_host = LayoutTensor[
        scales_dtype, b_scales_5d_layout, MutAnyOrigin
    ](
        b_scales_host_ptr,
        RuntimeLayout[b_scales_5d_layout].row_major(
            IndexList[5](
                b_scales_host.dim(0),
                b_scales_host.dim(1),
                b_scales_host.dim(2),
                b_scales_host.dim(3),
                b_scales_host.dim(4),
            ),
        ),
    )

    rand(a_scales_host.data, a_scales_host.num_elements())
    rand(b_scales_host.data, b_scales_host.num_elements())
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for idx0 in range(align_up(m.value, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(k.value, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 >= m.value or idx1 >= k.value:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_tensor_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(n.value, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(k.value, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 >= n.value or idx1 >= k.value:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_tensor_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
    )

    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
        alpha=alpha,
    )

    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_lt,
        b_lt,
        a_scales=a_scales_lt.get_immutable(),
        b_scales=b_scales_lt.get_immutable(),
        transpose_b=transpose_b,
        c_row_major=True,
        alpha=alpha,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def main():
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8  # TODO: (KERN-2238): Replace with float4-e2m1fn
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = NVFP4_SF_DTYPE
        comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B

        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        comptime for bm in [128]:
            comptime for bn in [64, 128]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(2 * bm, 2 * bn, MMA_K)

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(1000),
                    static[1024](),
                    static[1024 + 32](),
                    alpha=0.5,
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=4,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(512),
                    static[4096](),
                    static[1024 + 32](),
                    alpha=0.135,
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 2, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=0,
                    k_group_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(500),
                    static[2048](),
                    static[4096](),
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](8, 2, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=2,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(999),
                    static[256](),
                    static[128](),
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(777),
                    static[2560](),
                    static[8192](),
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(1),
                    static[576](),
                    static[7168](),
                )

                # swapAB tests
                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(16),
                    static[1024](),
                    static[1024 + 32](),
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    dynamic(100),
                    static[2560](),
                    static[8192](),
                )
