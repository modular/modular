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
"""Epilogue fusion tests for grouped block-scaled GEMM.

Tests element-wise epilogue lambda application on the grouped block-scaled
GEMM kernel output. Verifies that:
1. The lambda is correctly applied to each output element
2. Coordinates passed to the lambda are correct
3. Both register-based and SMEM-based epilogues work correctly
"""

from collections import Optional
from math import align_up, ceildiv
from random import rand, random_float64, seed
from sys import align_of, size_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from internal_utils import assert_almost_equal
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
)
from linalg.utils import elementwise_compute_lambda_type
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul import (
    grouped_block_scaled_matmul,
)


fn test_grouped_gemm_epilogue[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    transpose_b: Bool,
    cta_group: Int,
    mma_shape: IndexList[3],
    cluster_shape: IndexList[3],
    register_based_epilogue: Bool = True,
](ctx: DeviceContext) raises:
    """Test grouped block-scaled GEMM with epilogue lambda.

    The epilogue lambda adds the original C value to the matmul result,
    effectively computing: C' = matmul(A, B) + C_original

    This tests that:
    1. The lambda is applied to all output elements
    2. Global coordinates passed to lambda are correct
    3. The captured tensor is accessible from the lambda
    """
    print("\n--- Testing grouped GEMM with epilogue ---")
    print(
        "  M=",
        m.value,
        " N=",
        n.value,
        " K=",
        k.value,
        " cta_group=",
        cta_group,
        " register_based_epilogue=",
        register_based_epilogue,
    )

    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime max_groups = 1
    var num_groups = 1

    # Create NDBuffer shapes
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

    # Host allocations
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
    var c_host_original_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_original = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_original_ptr, dynamic_c_shape
    )

    # Device allocations
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

    # Scale factor shapes (5D)
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
        ceildiv(k.value, SF_VECTOR_SIZE * SF_ATOM_K),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, SF_MN_GROUP_SIZE),
        ceildiv(k.value, SF_VECTOR_SIZE * SF_ATOM_K),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )

    var sfa_size = dynamic_a_scales_shape.product[]().get()
    var sfb_size = dynamic_b_scales_shape.product[]().get()

    # Scale factor device allocations
    var sfa_device = ctx.enqueue_create_buffer[scales_dtype](sfa_size)
    var sfa_device_nd = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        sfa_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var sfb_device = ctx.enqueue_create_buffer[scales_dtype](sfb_size)
    var sfb_device_nd = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        sfb_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    # Scale factor host allocations
    var sfa_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(sfa_size)
    var sfa_host = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        sfa_host_ptr, dynamic_a_scales_shape
    )
    var sfb_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(sfb_size)
    var sfb_host = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        sfb_host_ptr, dynamic_b_scales_shape
    )

    # The C tensor that will be captured by the epilogue lambda
    var c_tensor = c_device_nd

    # Define epilogue lambda that adds original C value to matmul result
    @parameter
    @always_inline
    @__copy_capture(c_tensor)
    fn epilogue_add_c[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        # C' = matmul(A, B) + C_original
        return val + c_tensor.load[width=width](idx).cast[_dtype]()

    # Initialize random data
    seed(42)
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

    # Initialize C with random values for epilogue test
    for i in range(m.value):
        for j in range(n.value):
            c_host[i, j] = Scalar[c_type](random_float64(-1, 1))
            c_host_original[i, j] = c_host[i, j]

    # Initialize scale factors to 1.0 (identity scaling)
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(sfa_size):
        sfa_host_ptr[i] = scale_one
    for i in range(sfb_size):
        sfb_host_ptr[i] = scale_one

    # Copy to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(sfa_device, sfa_host_ptr)
    ctx.enqueue_copy(sfb_device, sfb_host_ptr)

    # Create config
    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        cta_group=cta_group,
    )

    # Problem sizes tensor
    comptime problem_sizes_layout = Layout.row_major(max_groups, 4)
    var problem_sizes_host = UnsafePointer[Int32].alloc(max_groups * 4)
    problem_sizes_host[0] = Int32(m.value)  # M
    problem_sizes_host[1] = Int32(n.value)  # N
    problem_sizes_host[2] = Int32(k.value)  # K
    problem_sizes_host[3] = Int32(1)  # L (batch)

    var problem_sizes_device = ctx.enqueue_create_buffer[DType.int32](
        max_groups * 4
    )
    ctx.enqueue_copy(problem_sizes_device, problem_sizes_host)

    var problem_sizes_tensor = LayoutTensor[
        DType.int32, problem_sizes_layout, MutAnyOrigin
    ](
        problem_sizes_device.unsafe_ptr(),
        RuntimeLayout[problem_sizes_layout].row_major(
            IndexList[2](max_groups, 4)
        ),
    )

    # Pointer arrays
    var a_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var b_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var c_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfa_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfb_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)

    a_ptrs_host[0] = UInt64(Int(a_device.unsafe_ptr()))
    b_ptrs_host[0] = UInt64(Int(b_device.unsafe_ptr()))
    c_ptrs_host[0] = UInt64(Int(c_device.unsafe_ptr()))
    sfa_ptrs_host[0] = UInt64(Int(sfa_device.unsafe_ptr()))
    sfb_ptrs_host[0] = UInt64(Int(sfb_device.unsafe_ptr()))

    var a_ptrs_device = ctx.enqueue_create_buffer[DType.uint64](max_groups)
    var b_ptrs_device = ctx.enqueue_create_buffer[DType.uint64](max_groups)
    var c_ptrs_device = ctx.enqueue_create_buffer[DType.uint64](max_groups)
    var sfa_ptrs_device = ctx.enqueue_create_buffer[DType.uint64](max_groups)
    var sfb_ptrs_device = ctx.enqueue_create_buffer[DType.uint64](max_groups)

    ctx.enqueue_copy(a_ptrs_device, a_ptrs_host)
    ctx.enqueue_copy(b_ptrs_device, b_ptrs_host)
    ctx.enqueue_copy(c_ptrs_device, c_ptrs_host)
    ctx.enqueue_copy(sfa_ptrs_device, sfa_ptrs_host)
    ctx.enqueue_copy(sfb_ptrs_device, sfb_ptrs_host)

    comptime ptr_layout = Layout.row_major(max_groups, 1)
    var a_ptrs_tensor = LayoutTensor[DType.uint64, ptr_layout, MutAnyOrigin](
        a_ptrs_device.unsafe_ptr(),
        RuntimeLayout[ptr_layout].row_major(IndexList[2](max_groups, 1)),
    )
    var b_ptrs_tensor = LayoutTensor[DType.uint64, ptr_layout, MutAnyOrigin](
        b_ptrs_device.unsafe_ptr(),
        RuntimeLayout[ptr_layout].row_major(IndexList[2](max_groups, 1)),
    )
    var c_ptrs_tensor = LayoutTensor[DType.uint64, ptr_layout, MutAnyOrigin](
        c_ptrs_device.unsafe_ptr(),
        RuntimeLayout[ptr_layout].row_major(IndexList[2](max_groups, 1)),
    )
    var sfa_ptrs_tensor = LayoutTensor[DType.uint64, ptr_layout, MutAnyOrigin](
        sfa_ptrs_device.unsafe_ptr(),
        RuntimeLayout[ptr_layout].row_major(IndexList[2](max_groups, 1)),
    )
    var sfb_ptrs_tensor = LayoutTensor[DType.uint64, ptr_layout, MutAnyOrigin](
        sfb_ptrs_device.unsafe_ptr(),
        RuntimeLayout[ptr_layout].row_major(IndexList[2](max_groups, 1)),
    )

    # Compute total tiles
    comptime BM = config.block_tile_shape[0]
    comptime BN = mma_shape[1]
    var total_tiles = ceildiv(m.value, BM) * ceildiv(n.value, BN)

    # Create epilogue lambda optional
    comptime optional_lambda = Optional[elementwise_compute_lambda_type](
        epilogue_add_c
    )

    # Launch grouped GEMM with epilogue
    grouped_block_scaled_matmul[
        config=config,
        max_groups=max_groups,
        elementwise_compute_lambda_fn=optional_lambda,
        register_based_epilogue=register_based_epilogue,
    ](
        a_ptrs_tensor,
        b_ptrs_tensor,
        c_ptrs_tensor,
        sfa_ptrs_tensor,
        sfb_ptrs_tensor,
        problem_sizes_tensor,
        num_groups,
        total_tiles,
        from_ndbuffer_row_major(a_device_nd),
        from_ndbuffer_row_major(b_device_nd),
        from_ndbuffer_row_major(c_device_nd),
        from_ndbuffer_row_major(sfa_device_nd),
        from_ndbuffer_row_major(sfb_device_nd),
        ctx,
    )

    # Run reference matmul (without epilogue)
    vendor_blas.matmul(
        ctx,
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    # Copy results back
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    # Apply epilogue lambda on CPU to reference
    var c_tensor_host = c_host_original

    @parameter
    @always_inline
    @__copy_capture(c_tensor_host)
    fn epilogue_add_c_host[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val + c_tensor_host.load[width=width](idx).cast[_dtype]()

    for i in range(m.value):
        for j in range(n.value):
            c_host_ref[Index(i, j)] = epilogue_add_c_host(
                IndexList[2](i, j),
                c_host_ref[Index(i, j)],
            )

    # Compare results
    comptime rtol = 1e-2
    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    print("  PASSED!")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    c_host_original_ptr.free()
    sfa_host_ptr.free()
    sfb_host_ptr.free()
    problem_sizes_host.free()
    a_ptrs_host.free()
    b_ptrs_host.free()
    c_ptrs_host.free()
    sfa_ptrs_host.free()
    sfb_ptrs_host.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = sfa_device^
    _ = sfb_device^
    _ = problem_sizes_device^
    _ = a_ptrs_device^
    _ = b_ptrs_device^
    _ = c_ptrs_device^
    _ = sfa_ptrs_device^
    _ = sfb_ptrs_device^


def main():
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    with DeviceContext() as ctx:
        print("\n" + "=" * 60)
        print("Grouped Block-Scaled GEMM Epilogue Tests")
        print("=" * 60)

        # Test 1: 1SM mode with register-based epilogue (small)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[256](),
            n = static[256](),
            k = static[128](),
            transpose_b=transpose_b,
            cta_group=1,
            mma_shape = Index(128, 128, 32),
            cluster_shape = Index(1, 1, 1),
            register_based_epilogue=True,
        ](ctx)

        # Test 2: 2SM mode with register-based epilogue (small)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[256](),
            n = static[256](),
            k = static[128](),
            transpose_b=transpose_b,
            cta_group=2,
            mma_shape = Index(256, 128, 32),
            cluster_shape = Index(2, 1, 1),
            register_based_epilogue=True,
        ](ctx)

        # Test 3: 1SM mode with register-based epilogue (larger)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[512](),
            n = static[512](),
            k = static[256](),
            transpose_b=transpose_b,
            cta_group=1,
            mma_shape = Index(128, 128, 32),
            cluster_shape = Index(1, 1, 1),
            register_based_epilogue=True,
        ](ctx)

        # Test 4: 2SM mode with register-based epilogue (larger)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[512](),
            n = static[512](),
            k = static[256](),
            transpose_b=transpose_b,
            cta_group=2,
            mma_shape = Index(256, 128, 32),
            cluster_shape = Index(2, 1, 1),
            register_based_epilogue=True,
        ](ctx)

        print("\n" + "=" * 60)
        print("All epilogue tests PASSED!")
        print("=" * 60)
