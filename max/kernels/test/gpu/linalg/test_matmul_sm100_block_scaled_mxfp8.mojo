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

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import assert_almost_equal
from random import rand
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from math import ceildiv, align_up
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
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
    SF_VECTOR_SIZE: Int = MXFP8_SF_VECTOR_SIZE,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
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
    var b_size = n.value * k.value if transpose_b else k.value * n.value
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

    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var b_tensor = from_ndbuffer_row_major(b_device_nd)
    var c_tensor = from_ndbuffer_row_major(c_device_nd)
    var a_scales_tensor = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales_tensor = from_ndbuffer_row_major(b_scales_device_nd)
    var c_ref_tensor = from_ndbuffer_row_major(c_device_ref_nd)

    # Initialize matmul operands
    if simple_init():
        for m in range(M):
            for k in range(K):
                a_host[m, k] = random_ui64(0, 1).cast[a_type]()
        for n in range(N):
            for k in range(K):
                b_host[n, k] = random_ui64(0, 1).cast[b_type]()
    else:
        rand(a_host.data, a_host.num_elements())
        rand(b_host.data, b_host.num_elements())

    comptime scales_5d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime a_scales_5d_layout = scales_5d_layout[a_scales_tensor.layout]
    comptime b_scales_5d_layout = scales_5d_layout[b_scales_tensor.layout]

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

    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for idx0 in range(align_up(m.value, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(k.value, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < m.value and idx1 < k.value:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[DType.float32]()
                    .cast[scales_dtype]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_tensor_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_tensor_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(n.value, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(k.value, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < n.value and idx1 < k.value:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[DType.float32]()
                    .cast[scales_dtype]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_tensor_host, idx0, idx1, scale_value
                )
            else:
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
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=1 if mma_shape[1] in (192, 256) else 2,
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
    )

    comptime assert a_type != DType.float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        a_scales=a_scales_tensor.get_immutable(),
        b_scales=b_scales_tensor.get_immutable(),
        transpose_b=transpose_b,
        c_row_major=True,
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
        comptime dtype = DType.float8_e4m3fn
        comptime out_dtype = DType.bfloat16
        comptime scale_dtype = MXFP8_SF_DTYPE
        comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        comptime for cta_group in [1, 2]:
            comptime for bm in [128]:
                comptime for mma_n in [64, 128, 192, 256]:
                    comptime block_tile_shape = Index(
                        bm, mma_n // cta_group, BK
                    )
                    comptime umma_shape = Index(cta_group * bm, mma_n, MMA_K)

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](
                            Int32(cta_group), 1, 1
                        ),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=8,
                    ](
                        ctx,
                        dynamic(1000),
                        static[1024](),
                        static[1024 + 16](),
                    )

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 2, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=0,
                        k_group_size=2,
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
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](8, 2, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=2,
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
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=1,
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
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=1,
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
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](
                            Int32(cta_group), 1, 1
                        ),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        swapAB=True,
                    ](
                        ctx,
                        dynamic(16),
                        static[1024](),
                        static[1024 + 16](),
                    )

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        swapAB=True,
                        k_group_size=2,
                    ](
                        ctx,
                        dynamic(100),
                        static[2560](),
                        static[8192](),
                    )
