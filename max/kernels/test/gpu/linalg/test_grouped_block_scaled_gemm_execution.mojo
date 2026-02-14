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
"""Execution test for grouped block-scaled GEMM with cuBLAS comparison.

This test incrementally verifies:
1. The existing block-scaled kernel works per group (baseline)
2. The grouped kernel works for a single group (identity test)
3. The grouped kernel works for multiple groups
"""

from math import align_up, ceildiv
from sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from memory import LegacyUnsafePointer, bitcast

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from internal_utils import assert_almost_equal
from random import rand, random_ui64, seed
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    lt_to_tt,
)
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    set_scale_factor,
)

from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
)
from linalg.matmul.gpu.sm100_structured.block_scaled.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul import (
    grouped_block_scaled_matmul,
)


fn test_existing_kernel_single_group[
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
](ctx: DeviceContext) raises:
    """Test existing block-scaled kernel with a single problem.

    This establishes that the kernel and tensor setup work correctly.
    """
    print("\n--- Testing existing kernel (single group baseline) ---")
    print(
        "  M=",
        m.value,
        " N=",
        n.value,
        " K=",
        k.value,
        " cta_group=",
        cta_group,
    )

    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

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

    # Create LayoutTensors
    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var b_tensor = from_ndbuffer_row_major(b_device_nd)
    var c_tensor = from_ndbuffer_row_major(c_device_nd)
    var a_scales_tensor = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales_tensor = from_ndbuffer_row_major(b_scales_device_nd)
    var c_ref_tensor = from_ndbuffer_row_major(c_device_ref_nd)

    # Initialize with random data
    rand(a_host_ptr, a_size)
    rand(b_host_ptr, b_size)
    for i in range(c_size):
        c_host_ptr[i] = 0
        c_host_ref_ptr[i] = 0

    # Set scale factors to 1.0 (using cast from float32)
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        a_scales_host_ptr[i] = scale_one
    for i in range(b_scales_total):
        b_scales_host_ptr[i] = scale_one

    # Copy to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.synchronize()

    # Run cuBLAS reference
    print("  Running cuBLAS reference...")
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

    # Run our kernel
    print("  Running block-scaled kernel...")

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
        k_group_size=1,
        num_accum_pipeline_stages=2,
    )

    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=config,
    ](
        lt_to_tt(c_tensor),
        lt_to_tt(a_tensor),
        lt_to_tt(b_tensor),
        lt_to_tt(a_scales_tensor),
        lt_to_tt(b_scales_tensor),
        ctx,
    )
    ctx.synchronize()

    # Copy results back
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    # Compare
    var max_diff: Float32 = 0.0
    var sum_diff: Float32 = 0.0
    for i in range(c_size):
        var kernel_val = c_host_ptr[i].cast[DType.float32]()
        var ref_val = c_host_ref_ptr[i].cast[DType.float32]()
        var diff = abs(kernel_val - ref_val)
        max_diff = max(max_diff, diff)
        sum_diff += diff

    var avg_diff = sum_diff / Float32(c_size)

    if max_diff < 0.1:
        print(
            "  PASSED (max_diff=",
            max_diff,
            ", avg_diff=",
            avg_diff,
            ")",
        )
    else:
        print(
            "  FAILED (max_diff=",
            max_diff,
            ", avg_diff=",
            avg_diff,
            ")",
        )
        raise Error("Test failed - kernel output does not match cuBLAS")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()


fn test_grouped_kernel_single_group[
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
](ctx: DeviceContext) raises:
    """Test grouped kernel with a single group (should behave like non-grouped).

    This tests the grouped infrastructure with trivial grouping.
    """
    print("\n--- Testing grouped kernel (single group) ---")
    print(
        "  M=",
        m.value,
        " N=",
        n.value,
        " K=",
        k.value,
        " cta_group=",
        cta_group,
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
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)

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
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        b_scales_total
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

    # Create LayoutTensors for cuBLAS
    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var b_tensor = from_ndbuffer_row_major(b_device_nd)
    var c_ref_tensor = from_ndbuffer_row_major(c_device_ref_nd)
    var a_scales_tensor = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales_tensor = from_ndbuffer_row_major(b_scales_device_nd)

    # Test with random data
    rand(a_host_ptr, a_size)
    rand(b_host_ptr, b_size)
    for i in range(c_size):
        c_host_ptr[i] = 0
        c_host_ref_ptr[i] = 0

    # Set scale factors to 1.0
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        a_scales_host_ptr[i] = scale_one
    for i in range(b_scales_total):
        b_scales_host_ptr[i] = scale_one

    # Copy to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.synchronize()

    # Run cuBLAS reference
    print("  Running cuBLAS reference...")
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

    # Setup grouped kernel inputs
    print("  Setting up grouped kernel inputs...")

    # Problem sizes tensor: (max_groups, 4) with [M, N, K, L=1]
    var problem_sizes_host = UnsafePointer[Int32].alloc(max_groups * 4)
    problem_sizes_host[0] = Int32(m.value)  # M
    problem_sizes_host[1] = Int32(n.value)  # N
    problem_sizes_host[2] = Int32(k.value)  # K
    problem_sizes_host[3] = 1  # L (batch=1)

    var problem_sizes_device = ctx.enqueue_create_buffer[DType.int32](
        max_groups * 4
    )
    ctx.enqueue_copy(problem_sizes_device, problem_sizes_host)
    ctx.synchronize()

    # Create HOST-based problem_sizes tensor for host-side computations
    # The grouped_block_scaled_matmul function reads this on the host
    comptime problem_sizes_layout = Layout.row_major(max_groups, 4)
    var problem_sizes_tensor_host = LayoutTensor[
        DType.int32, problem_sizes_layout, MutAnyOrigin
    ](
        problem_sizes_host,  # Use HOST pointer for host-side access
        RuntimeLayout[problem_sizes_layout].row_major(
            IndexList[2](max_groups, 4)
        ),
    )

    # Create DEVICE-based problem_sizes tensor for kernel
    var problem_sizes_tensor_device = LayoutTensor[
        DType.int32, problem_sizes_layout, MutAnyOrigin
    ](
        problem_sizes_device.unsafe_ptr(),
        RuntimeLayout[problem_sizes_layout].row_major(
            IndexList[2](max_groups, 4)
        ),
    )

    # Compute total tiles on HOST
    # For 2SM (cta_group=2), tile size is mma_shape // cta_group
    comptime tile_m = mma_shape[0] // cta_group  # BM
    comptime tile_n = mma_shape[1] // cta_group  # BN
    var total_tiles = 0
    for g in range(num_groups):
        var m_val = Int(problem_sizes_host[g * 4 + 0])
        var n_val = Int(problem_sizes_host[g * 4 + 1])
        var m_tiles_count = ceildiv(m_val, tile_m)
        var n_tiles_count = ceildiv(n_val, tile_n)
        total_tiles += m_tiles_count * n_tiles_count
    print("  Computed total_tiles on host:", total_tiles)

    # Pointer arrays: (max_groups, 1)
    var a_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var b_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var c_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfa_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfb_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)

    a_ptrs_host[0] = UInt64(Int(a_device.unsafe_ptr()))
    b_ptrs_host[0] = UInt64(Int(b_device.unsafe_ptr()))
    c_ptrs_host[0] = UInt64(Int(c_device.unsafe_ptr()))
    sfa_ptrs_host[0] = UInt64(Int(a_scales_device.unsafe_ptr()))
    sfb_ptrs_host[0] = UInt64(Int(b_scales_device.unsafe_ptr()))

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
    ctx.synchronize()

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

    # Template tensors - use the 2D tensors directly (like non-grouped kernel)
    # The grouped_block_scaled_matmul function will convert them to 3D internally
    # This preserves compile-time dimensions for proper TMA descriptor creation
    # a_tensor, b_tensor, a_scales_tensor, b_scales_tensor already created above

    # Create c_tensor for grouped output (reuse c_device_nd from above)
    var c_tensor_for_grouped = from_ndbuffer_row_major(c_device_nd)

    # Run the grouped kernel with 2D template tensors
    print("  Running grouped kernel...")

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
        k_group_size=1,
        num_accum_pipeline_stages=2,
    )

    grouped_block_scaled_matmul[
        transpose_b=transpose_b,
        max_groups=max_groups,
        config=config,
    ](
        a_ptrs_tensor,
        b_ptrs_tensor,
        c_ptrs_tensor,
        sfa_ptrs_tensor,
        sfb_ptrs_tensor,
        problem_sizes_tensor_device,
        num_groups,
        total_tiles,
        a_tensor,
        b_tensor,
        c_tensor_for_grouped,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
    )
    ctx.synchronize()

    print("  Kernel completed")

    # Copy results back
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    # Compare
    var max_diff: Float32 = 0.0
    var sum_diff: Float32 = 0.0
    for i in range(c_size):
        var kernel_val = c_host_ptr[i].cast[DType.float32]()
        var ref_val = c_host_ref_ptr[i].cast[DType.float32]()
        var diff = abs(kernel_val - ref_val)
        max_diff = max(max_diff, diff)
        sum_diff += diff

    var avg_diff = sum_diff / Float32(c_size)

    if max_diff < 0.1:
        print(
            "  PASSED (max_diff=",
            max_diff,
            ", avg_diff=",
            avg_diff,
            ")",
        )
    else:
        print(
            "  FAILED (max_diff=",
            max_diff,
            ", avg_diff=",
            avg_diff,
            ")",
        )
        raise Error("Grouped kernel output does not match cuBLAS")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    problem_sizes_host.free()
    a_ptrs_host.free()
    b_ptrs_host.free()
    c_ptrs_host.free()
    sfa_ptrs_host.free()
    sfb_ptrs_host.free()


fn test_grouped_kernel_multi_group_same_ptr[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    num_groups: Int,
    transpose_b: Bool,
    cta_group: Int,
    mma_shape: IndexList[3],
    cluster_shape: IndexList[3],
](ctx: DeviceContext) raises:
    """Test grouped kernel with multiple groups using same pointers.

    This tests group iteration logic without needing TMA base pointer updates.
    All groups point to the same tensors - the result will be computed
    num_groups times (overwriting each time).
    """
    print(
        "\n--- Testing grouped kernel (", num_groups, " groups, same ptr) ---"
    )
    print(
        "  M=",
        m.value,
        " N=",
        n.value,
        " K=",
        k.value,
        " num_groups=",
        num_groups,
        " cta_group=",
        cta_group,
    )

    # Allocate matrices (same as single-group)
    var a_size = m.value * k.value
    var b_size = k.value * n.value if not transpose_b else n.value * k.value
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)

    # Static shapes
    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(
        k.dim, n.dim
    ) if not transpose_b else DimList(n.dim, k.dim)
    comptime static_c_shape = DimList(m.dim, n.dim)

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(
        k.value, n.value
    ) if not transpose_b else DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)

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
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
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
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        b_scales_total
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

    # Create LayoutTensors
    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var b_tensor = from_ndbuffer_row_major(b_device_nd)
    var c_tensor = from_ndbuffer_row_major(c_device_nd)
    var c_ref_tensor = from_ndbuffer_row_major(c_device_ref_nd)
    var a_scales_tensor = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales_tensor = from_ndbuffer_row_major(b_scales_device_nd)

    # Test with random data
    rand(a_host_ptr, a_size)
    rand(b_host_ptr, b_size)
    for i in range(c_size):
        c_host_ptr[i] = 0
        c_host_ref_ptr[i] = 0

    # Set scale factors to 1.0
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        a_scales_host_ptr[i] = scale_one
    for i in range(b_scales_total):
        b_scales_host_ptr[i] = scale_one

    # Copy to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.synchronize()

    # Run cuBLAS reference (single GEMM - result should match grouped)
    print("  Running cuBLAS reference...")
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

    # Setup grouped kernel inputs for num_groups groups
    print("  Setting up grouped kernel inputs...")

    comptime max_groups = num_groups

    # Problem sizes tensor: (max_groups, 4) with [M, N, K, L=1]
    var problem_sizes_host = UnsafePointer[Int32].alloc(max_groups * 4)
    for g in range(max_groups):
        problem_sizes_host[g * 4 + 0] = Int32(m.value)  # M
        problem_sizes_host[g * 4 + 1] = Int32(n.value)  # N
        problem_sizes_host[g * 4 + 2] = Int32(k.value)  # K
        problem_sizes_host[g * 4 + 3] = 1  # L (batch=1)

    var problem_sizes_device = ctx.enqueue_create_buffer[DType.int32](
        max_groups * 4
    )
    ctx.enqueue_copy(problem_sizes_device, problem_sizes_host)

    # Pointer tensors - ALL groups point to the SAME tensors
    var a_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var b_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var c_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfa_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfb_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)

    for g in range(max_groups):
        a_ptrs_host[g] = UInt64(Int(a_device.unsafe_ptr()))
        b_ptrs_host[g] = UInt64(Int(b_device.unsafe_ptr()))
        c_ptrs_host[g] = UInt64(Int(c_device.unsafe_ptr()))
        sfa_ptrs_host[g] = UInt64(Int(a_scales_device.unsafe_ptr()))
        sfb_ptrs_host[g] = UInt64(Int(b_scales_device.unsafe_ptr()))

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
    ctx.synchronize()

    # Create tensors for dispatch
    comptime problem_sizes_layout = Layout.row_major(max_groups, 4)
    var problem_sizes_tensor_host = LayoutTensor[
        DType.int32, problem_sizes_layout, MutAnyOrigin
    ](
        problem_sizes_host,
        RuntimeLayout[problem_sizes_layout].row_major(
            IndexList[2](max_groups, 4)
        ),
    )
    var problem_sizes_tensor_device = LayoutTensor[
        DType.int32, problem_sizes_layout, MutAnyOrigin
    ](
        problem_sizes_device.unsafe_ptr(),
        RuntimeLayout[problem_sizes_layout].row_major(
            IndexList[2](max_groups, 4)
        ),
    )

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

    # Calculate total tiles across all groups
    comptime BM = mma_shape[0] // cta_group
    comptime BN = mma_shape[1] // cta_group
    var tiles_per_group = ceildiv(m.value, BM) * ceildiv(n.value, BN)
    var total_tiles = tiles_per_group * num_groups

    print("  Total tiles across", num_groups, "groups:", total_tiles)

    # Run grouped kernel
    print("  Running grouped kernel...")

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
        k_group_size=1,
        num_accum_pipeline_stages=2,
    )

    grouped_block_scaled_matmul[
        transpose_b=transpose_b,
        config=config,
        max_groups=max_groups,
    ](
        a_ptrs_tensor,
        b_ptrs_tensor,
        c_ptrs_tensor,
        sfa_ptrs_tensor,
        sfb_ptrs_tensor,
        problem_sizes_tensor_host,
        num_groups,
        total_tiles,
        a_tensor,  # Template for TMA creation
        b_tensor,
        c_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
    )
    ctx.synchronize()
    print("  Kernel completed")

    # Copy results back
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    # Compare - result should match cuBLAS since we computed the same GEMM num_groups times
    var max_diff: Float32 = 0.0
    var sum_diff: Float32 = 0.0
    for i in range(c_size):
        var kernel_val = c_host_ptr[i].cast[DType.float32]()
        var ref_val = c_host_ref_ptr[i].cast[DType.float32]()
        var diff = abs(kernel_val - ref_val)
        max_diff = max(max_diff, diff)
        sum_diff += diff

    var avg_diff = sum_diff / Float32(c_size)

    if max_diff < 0.1:
        print("  PASSED (max_diff=", max_diff, ", avg_diff=", avg_diff, ")")
    else:
        print("  FAILED (max_diff=", max_diff, ", avg_diff=", avg_diff, ")")
        raise Error(
            "Grouped kernel (multi-group same ptr) does not match cuBLAS"
        )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    problem_sizes_host.free()
    a_ptrs_host.free()
    b_ptrs_host.free()
    c_ptrs_host.free()
    sfa_ptrs_host.free()
    sfb_ptrs_host.free()


fn test_grouped_kernel_two_groups_different_ptrs[
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
](ctx: DeviceContext) raises:
    """Test grouped kernel with 2 groups using different pointers per group.

    This tests the CuTe DSL TMA update pattern - each group has its own
    A, B, C, SFA, SFB tensors with different base addresses.
    """
    print("\n--- Testing grouped kernel (2 groups, different ptrs) ---")
    print(
        "  M=",
        m.value,
        " N=",
        n.value,
        " K=",
        k.value,
        " cta_group=",
        cta_group,
    )

    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime max_groups = 2
    var num_groups = 2

    # Compute sizes
    var a_size = m.value * k.value
    var b_size = n.value * k.value if transpose_b else k.value * n.value
    var c_size = m.value * n.value

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

    # Static shapes for NDBuffer
    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    comptime static_c_shape = DimList(m.dim, n.dim)
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

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)
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

    # ========== Group 0 allocations ==========
    var a0_host = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b0_host = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c0_host = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c0_ref_host = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var sfa0_host = UnsafePointer[Scalar[scales_dtype]].alloc(a_scales_total)
    var sfb0_host = UnsafePointer[Scalar[scales_dtype]].alloc(b_scales_total)

    var a0_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b0_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c0_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c0_ref_device = ctx.enqueue_create_buffer[c_type](c_size)
    var sfa0_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_total)
    var sfb0_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_total)

    # ========== Group 1 allocations ==========
    var a1_host = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b1_host = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c1_host = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c1_ref_host = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var sfa1_host = UnsafePointer[Scalar[scales_dtype]].alloc(a_scales_total)
    var sfb1_host = UnsafePointer[Scalar[scales_dtype]].alloc(b_scales_total)

    var a1_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b1_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c1_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c1_ref_device = ctx.enqueue_create_buffer[c_type](c_size)
    var sfa1_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_total)
    var sfb1_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_total)

    # Initialize with DIFFERENT random data for each group
    seed(42)  # Reproducible
    rand(a0_host, a_size)
    rand(b0_host, b_size)
    seed(123)  # Different seed for group 1
    rand(a1_host, a_size)
    rand(b1_host, b_size)

    # Zero C outputs
    for i in range(c_size):
        c0_host[i] = 0
        c0_ref_host[i] = 0
        c1_host[i] = 0
        c1_ref_host[i] = 0

    # Set scale factors to 1.0
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        sfa0_host[i] = scale_one
        sfa1_host[i] = scale_one
    for i in range(b_scales_total):
        sfb0_host[i] = scale_one
        sfb1_host[i] = scale_one

    # Copy to device
    ctx.enqueue_copy(a0_device, a0_host)
    ctx.enqueue_copy(b0_device, b0_host)
    ctx.enqueue_copy(c0_device, c0_host)
    ctx.enqueue_copy(c0_ref_device, c0_ref_host)
    ctx.enqueue_copy(sfa0_device, sfa0_host)
    ctx.enqueue_copy(sfb0_device, sfb0_host)

    ctx.enqueue_copy(a1_device, a1_host)
    ctx.enqueue_copy(b1_device, b1_host)
    ctx.enqueue_copy(c1_device, c1_host)
    ctx.enqueue_copy(c1_ref_device, c1_ref_host)
    ctx.enqueue_copy(sfa1_device, sfa1_host)
    ctx.enqueue_copy(sfb1_device, sfb1_host)
    ctx.synchronize()

    # Create NDBuffers for cuBLAS
    var a0_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a0_device.unsafe_ptr(), dynamic_a_shape
    )
    var b0_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b0_device.unsafe_ptr(), dynamic_b_shape
    )
    var c0_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c0_ref_device.unsafe_ptr(), dynamic_c_shape
    )
    var sfa0_nd = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        sfa0_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var sfb0_nd = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        sfb0_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    var a1_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a1_device.unsafe_ptr(), dynamic_a_shape
    )
    var b1_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b1_device.unsafe_ptr(), dynamic_b_shape
    )
    var c1_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c1_ref_device.unsafe_ptr(), dynamic_c_shape
    )
    var sfa1_nd = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        sfa1_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var sfb1_nd = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        sfb1_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    # Create LayoutTensors
    var a0_tensor = from_ndbuffer_row_major(a0_nd)
    var b0_tensor = from_ndbuffer_row_major(b0_nd)
    var c0_ref_tensor = from_ndbuffer_row_major(c0_ref_nd)
    var sfa0_tensor = from_ndbuffer_row_major(sfa0_nd)
    var sfb0_tensor = from_ndbuffer_row_major(sfb0_nd)

    var a1_tensor = from_ndbuffer_row_major(a1_nd)
    var b1_tensor = from_ndbuffer_row_major(b1_nd)
    var c1_ref_tensor = from_ndbuffer_row_major(c1_ref_nd)
    var sfa1_tensor = from_ndbuffer_row_major(sfa1_nd)
    var sfb1_tensor = from_ndbuffer_row_major(sfb1_nd)

    # Run cuBLAS for each group separately
    print("  Running cuBLAS for group 0...")
    vendor_blas.matmul(
        ctx,
        c0_ref_tensor,
        a0_tensor,
        b0_tensor,
        a_scales=sfa0_tensor.get_immutable(),
        b_scales=sfb0_tensor.get_immutable(),
        transpose_b=transpose_b,
        c_row_major=True,
    )

    print("  Running cuBLAS for group 1...")
    vendor_blas.matmul(
        ctx,
        c1_ref_tensor,
        a1_tensor,
        b1_tensor,
        a_scales=sfa1_tensor.get_immutable(),
        b_scales=sfb1_tensor.get_immutable(),
        transpose_b=transpose_b,
        c_row_major=True,
    )
    ctx.synchronize()

    # Setup grouped kernel inputs
    print("  Setting up grouped kernel inputs...")

    # Problem sizes: both groups have same size
    var problem_sizes_host = UnsafePointer[Int32].alloc(max_groups * 4)
    for g in range(max_groups):
        problem_sizes_host[g * 4 + 0] = Int32(m.value)
        problem_sizes_host[g * 4 + 1] = Int32(n.value)
        problem_sizes_host[g * 4 + 2] = Int32(k.value)
        problem_sizes_host[g * 4 + 3] = 1

    var problem_sizes_device = ctx.enqueue_create_buffer[DType.int32](
        max_groups * 4
    )
    ctx.enqueue_copy(problem_sizes_device, problem_sizes_host)

    # Pointer arrays - DIFFERENT pointers per group
    var a_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var b_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var c_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfa_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfb_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)

    # Group 0 pointers
    a_ptrs_host[0] = UInt64(Int(a0_device.unsafe_ptr()))
    b_ptrs_host[0] = UInt64(Int(b0_device.unsafe_ptr()))
    c_ptrs_host[0] = UInt64(Int(c0_device.unsafe_ptr()))
    sfa_ptrs_host[0] = UInt64(Int(sfa0_device.unsafe_ptr()))
    sfb_ptrs_host[0] = UInt64(Int(sfb0_device.unsafe_ptr()))

    # Group 1 pointers - DIFFERENT from group 0
    a_ptrs_host[1] = UInt64(Int(a1_device.unsafe_ptr()))
    b_ptrs_host[1] = UInt64(Int(b1_device.unsafe_ptr()))
    c_ptrs_host[1] = UInt64(Int(c1_device.unsafe_ptr()))
    sfa_ptrs_host[1] = UInt64(Int(sfa1_device.unsafe_ptr()))
    sfb_ptrs_host[1] = UInt64(Int(sfb1_device.unsafe_ptr()))

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
    ctx.synchronize()

    # Create tensors for dispatch
    comptime problem_sizes_layout = Layout.row_major(max_groups, 4)
    var problem_sizes_tensor_host = LayoutTensor[
        DType.int32, problem_sizes_layout, MutAnyOrigin
    ](
        problem_sizes_host,
        RuntimeLayout[problem_sizes_layout].row_major(
            IndexList[2](max_groups, 4)
        ),
    )

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

    # Calculate total tiles
    comptime BM = mma_shape[0] // cta_group
    comptime BN = mma_shape[1] // cta_group
    var tiles_per_group = ceildiv(m.value, BM) * ceildiv(n.value, BN)
    var total_tiles = tiles_per_group * num_groups

    print("  Total tiles across 2 groups:", total_tiles)

    # Run grouped kernel
    print("  Running grouped kernel with different pointers per group...")

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
        k_group_size=1,
        num_accum_pipeline_stages=2,
    )

    # Use group 0 tensors as templates for TMA descriptor creation
    var c0_nd_for_template = NDBuffer[c_type, 2, _, static_c_shape](
        c0_device.unsafe_ptr(), dynamic_c_shape
    )
    var c0_tensor_for_template = from_ndbuffer_row_major(c0_nd_for_template)

    grouped_block_scaled_matmul[
        transpose_b=transpose_b,
        max_groups=max_groups,
        config=config,
    ](
        a_ptrs_tensor,
        b_ptrs_tensor,
        c_ptrs_tensor,
        sfa_ptrs_tensor,
        sfb_ptrs_tensor,
        problem_sizes_tensor_host,
        num_groups,
        total_tiles,
        a0_tensor,  # Template tensors from group 0
        b0_tensor,
        c0_tensor_for_template,
        sfa0_tensor,
        sfb0_tensor,
        ctx,
    )
    ctx.synchronize()
    print("  Kernel completed")

    # Copy results back
    ctx.enqueue_copy(c0_host, c0_device)
    ctx.enqueue_copy(c0_ref_host, c0_ref_device)
    ctx.enqueue_copy(c1_host, c1_device)
    ctx.enqueue_copy(c1_ref_host, c1_ref_device)
    ctx.synchronize()

    # Compare group 0
    var max_diff0: Float32 = 0.0
    var sum_diff0: Float32 = 0.0
    for i in range(c_size):
        var kernel_val = c0_host[i].cast[DType.float32]()
        var ref_val = c0_ref_host[i].cast[DType.float32]()
        var diff = abs(kernel_val - ref_val)
        max_diff0 = max(max_diff0, diff)
        sum_diff0 += diff
    var avg_diff0 = sum_diff0 / Float32(c_size)

    # Compare group 1
    var max_diff1: Float32 = 0.0
    var sum_diff1: Float32 = 0.0
    for i in range(c_size):
        var kernel_val = c1_host[i].cast[DType.float32]()
        var ref_val = c1_ref_host[i].cast[DType.float32]()
        var diff = abs(kernel_val - ref_val)
        max_diff1 = max(max_diff1, diff)
        sum_diff1 += diff
    var avg_diff1 = sum_diff1 / Float32(c_size)

    print("  Group 0: max_diff=", max_diff0, ", avg_diff=", avg_diff0)
    print("  Group 1: max_diff=", max_diff1, ", avg_diff=", avg_diff1)

    var passed = max_diff0 < 0.1 and max_diff1 < 0.1
    if passed:
        print("  PASSED (both groups match cuBLAS)")
    else:
        print("  FAILED (group outputs do not match cuBLAS)")
        raise Error("Multi-group different pointers test failed")

    # Cleanup
    a0_host.free()
    b0_host.free()
    c0_host.free()
    c0_ref_host.free()
    sfa0_host.free()
    sfb0_host.free()
    a1_host.free()
    b1_host.free()
    c1_host.free()
    c1_ref_host.free()
    sfa1_host.free()
    sfb1_host.free()
    problem_sizes_host.free()
    a_ptrs_host.free()
    b_ptrs_host.free()
    c_ptrs_host.free()
    sfa_ptrs_host.free()
    sfb_ptrs_host.free()


def main():
    print("\n" + "=" * 60)
    print("Test: Incremental Grouped Block-Scaled GEMM Execution")
    print("=" * 60)

    var ctx = DeviceContext()

    # Test 1: Verify the existing kernel works
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    test_existing_kernel_single_group[
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
    ](ctx)

    print("\n--- Baseline test passed ---")

    # Test 2: Test grouped kernel with SAME dimensions as baseline
    test_grouped_kernel_single_group[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),  # Same as baseline
        n = static[256](),  # Same as baseline
        k = static[128](),
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    print("\n--- Additional test cases ---")

    # Test 3: Larger M dimension (more M tiles)
    test_grouped_kernel_single_group[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[512](),
        n = static[256](),
        k = static[128](),
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    # Test 4: Larger N dimension (more N tiles)
    test_grouped_kernel_single_group[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),
        n = static[512](),
        k = static[128](),
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    # Test 5: Larger K dimension (more K iterations)
    test_grouped_kernel_single_group[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),
        n = static[256](),
        k = static[256](),
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    # Test 6: All dimensions larger
    test_grouped_kernel_single_group[
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
    ](ctx)

    # Test 7: Minimum tile size (single tile per dimension)
    test_grouped_kernel_single_group[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[128](),
        n = static[128](),
        k = static[128](),
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    # Test 8: Many K iterations
    test_grouped_kernel_single_group[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),
        n = static[256](),
        k = static[512](),
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    print("\n--- Multi-group test (same pointers) ---")

    # Test 9: Two groups with same size and same pointers
    # This tests group iteration without needing TMA base pointer updates
    test_grouped_kernel_multi_group_same_ptr[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),
        n = static[256](),
        k = static[128](),
        num_groups=2,
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    # Test 10: Four groups with same size and same pointers
    test_grouped_kernel_multi_group_same_ptr[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),
        n = static[256](),
        k = static[128](),
        num_groups=4,
        transpose_b=transpose_b,
        cta_group=1,
        mma_shape = Index(128, 128, 32),
        cluster_shape = Index(1, 1, 1),
    ](ctx)

    print("\n--- Multi-group test (different pointers) ---")

    # Test 11: Two groups with different pointers per group
    # This tests the CuTe DSL TMA update pattern
    test_grouped_kernel_two_groups_different_ptrs[
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
    ](ctx)

    # 2SM tests - using run_2sm() production kernel
    print("\n--- 2SM (cta_group=2) tests ---")
    # Note: 2SM requires mma_shape[0]=256 so that BM = 256/2 = 128 >= SF_MN_GROUP_SIZE
    # Also requires mma_shape[2]=32 for block_tile_shape[2]=128 constraint

    # Test 12: 2SM single group (256x256 with 2SM cluster)
    # Note: mma_shape must match run_simple_2sm: (256, 128, 32)
    test_grouped_kernel_single_group[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),
        n = static[256](),
        k = static[128](),
        transpose_b=transpose_b,
        cta_group=2,  # 2SM mode
        mma_shape = Index(
            256, 128, 32
        ),  # mma[0]=256 for BM=128, mma[1]=128 for BN=128
        cluster_shape = Index(2, 1, 1),  # 2 CTAs per cluster
    ](ctx)

    # Test 13: 2SM with multiple tiles (512x512)
    test_grouped_kernel_single_group[
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
    ](ctx)

    # Test 14: 2SM multi-group (same pointers)
    test_grouped_kernel_multi_group_same_ptr[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        m = static[256](),
        n = static[256](),
        k = static[128](),
        num_groups=2,
        transpose_b=transpose_b,
        cta_group=2,
        mma_shape = Index(256, 128, 32),
        cluster_shape = Index(2, 1, 1),
    ](ctx)

    # Test 15: 2SM multi-group (different pointers)
    test_grouped_kernel_two_groups_different_ptrs[
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
    ](ctx)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
