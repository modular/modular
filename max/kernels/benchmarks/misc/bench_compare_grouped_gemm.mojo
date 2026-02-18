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
"""Apples-to-apples comparison: cuBLAS (per-group) vs Structured Grouped GEMM.

Both sides use identical block-scaled data formats (MXFP8 or NVFP4):
- MXFP8: float8_e4m3fn A/B with UE8M0 scale factors
- NVFP4: uint8 (packed FP4) A/B with float8_e4m3fn scale factors
- C: bfloat16

The cuBLAS baseline calls vendor_blas.matmul once per group (sequentially),
which is what you'd do without a grouped/persistent kernel. The structured
kernel processes all groups in a single persistent launch.
"""

from math import ceildiv
from memory import LegacyUnsafePointer
from time import perf_counter_ns

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from random import rand, seed
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout import Layout, LayoutTensor, RuntimeLayout

from utils.index import Index, IndexList

import linalg.matmul.vendor.blas as vendor_blas
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
)

# Structured kernel
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul import (
    grouped_block_scaled_matmul,
)

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


fn bench_cublas_per_group[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    num_groups: Int,
    sf_vector_size: Int = MXFP8_SF_VECTOR_SIZE,
](ctx: DeviceContext, mut bench: Bench) raises:
    """Benchmark cuBLAS called once per group (sequential baseline).

    This represents the naive approach without a grouped/persistent kernel:
    iterate over groups and call cuBLAS matmul for each one.
    """

    comptime SF_VECTOR_SIZE = sf_vector_size
    comptime transpose_b = True
    comptime is_fp4 = (a_type == DType.uint8)
    comptime k_pack = 2 if is_fp4 else 1
    comptime k_array_dim = k.dim // k_pack
    var k_array_val = k.value // k_pack

    var a_size = m.value * k_array_val
    var b_size = n.value * k_array_val
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

    comptime static_a_shape = DimList(m.dim, k_array_dim)
    comptime static_b_shape = DimList(n.dim, k_array_dim)
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

    var a_host = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var sfa_host = UnsafePointer[Scalar[scales_dtype]].alloc(a_scales_total)
    var sfb_host = UnsafePointer[Scalar[scales_dtype]].alloc(b_scales_total)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var sfa_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_total)
    var sfb_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_total)

    seed(42)
    rand(a_host, a_size)
    rand(b_host, b_size)
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        sfa_host[i] = scale_one
    for i in range(b_scales_total):
        sfb_host[i] = scale_one

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(sfa_device, sfa_host)
    ctx.enqueue_copy(sfb_device, sfb_host)
    ctx.synchronize()

    var dynamic_a_shape = DimList(m.value, k_array_val)
    var dynamic_b_shape = DimList(n.value, k_array_val)
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

    var a_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var sfa_nd = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        sfa_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var sfb_nd = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        sfb_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    var a_tensor = from_ndbuffer_row_major(a_nd)
    var b_tensor = from_ndbuffer_row_major(b_nd)
    var c_tensor = from_ndbuffer_row_major(c_nd)
    var sfa_tensor = from_ndbuffer_row_major(sfa_nd)
    var sfb_tensor = from_ndbuffer_row_major(sfb_nd)

    # FLOPs use logical K (not packed)
    var total_flops = 2 * m.value * n.value * k.value * num_groups

    @parameter
    @__copy_capture(a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor)
    @always_inline
    fn bench_func(mut bencher: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            # Call cuBLAS once per group (sequential)
            for _g in range(num_groups):
                vendor_blas.matmul(
                    ctx,
                    c_tensor,
                    a_tensor,
                    b_tensor,
                    a_scales=sfa_tensor.get_immutable(),
                    b_scales=sfb_tensor.get_immutable(),
                    transpose_b=transpose_b,
                    c_row_major=True,
                )

        bencher.iter_custom[kernel_launch](ctx)

    var fmt = String("NVFP4") if is_fp4 else String("MXFP8")

    bench.bench_function[bench_func](
        BenchId(
            String(
                "cuBLAS(",
                fmt,
                ",per-group) : ",
                num_groups,
                " x ",
                m.value,
                " x ",
                n.value,
                " x ",
                k.value,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    # Cleanup
    a_host.free()
    b_host.free()
    sfa_host.free()
    sfb_host.free()


fn bench_structured_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    num_groups: Int,
    mma_m: Int = 128,
    mma_n: Int = 128,
    k_grp_size: Int = 1,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF8F6F4,
    sf_vector_size: Int = MXFP8_SF_VECTOR_SIZE,
](ctx: DeviceContext, mut bench: Bench) raises:
    """Benchmark structured grouped_block_scaled_matmul."""

    comptime SF_VECTOR_SIZE = sf_vector_size
    comptime max_groups = num_groups
    comptime transpose_b = True
    comptime mma_shape = Index(mma_m, mma_n, 32)
    comptime cluster_shape = Index(1, 1, 1)
    comptime is_fp4 = (a_type == DType.uint8)
    comptime k_pack = 2 if is_fp4 else 1
    comptime k_array_dim = k.dim // k_pack
    var k_array_val = k.value // k_pack

    var a_size = m.value * k_array_val
    var b_size = n.value * k_array_val
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

    comptime static_a_shape = DimList(m.dim, k_array_dim)
    comptime static_b_shape = DimList(n.dim, k_array_dim)
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

    var a_host = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var sfa_host = UnsafePointer[Scalar[scales_dtype]].alloc(a_scales_total)
    var sfb_host = UnsafePointer[Scalar[scales_dtype]].alloc(b_scales_total)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var sfa_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_total)
    var sfb_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_total)

    seed(42)
    rand(a_host, a_size)
    rand(b_host, b_size)
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        sfa_host[i] = scale_one
    for i in range(b_scales_total):
        sfb_host[i] = scale_one

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(sfa_device, sfa_host)
    ctx.enqueue_copy(sfb_device, sfb_host)
    ctx.synchronize()

    var dynamic_a_shape = DimList(m.value, k_array_val)
    var dynamic_b_shape = DimList(n.value, k_array_val)
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

    var a_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var sfa_nd = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        sfa_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var sfb_nd = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        sfb_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    var a_tensor = from_ndbuffer_row_major(a_nd)
    var b_tensor = from_ndbuffer_row_major(b_nd)
    var c_tensor = from_ndbuffer_row_major(c_nd)
    var sfa_tensor = from_ndbuffer_row_major(sfa_nd)
    var sfb_tensor = from_ndbuffer_row_major(sfb_nd)

    var problem_sizes_host = UnsafePointer[Int32].alloc(max_groups * 4)
    for g in range(max_groups):
        problem_sizes_host[g * 4 + 0] = Int32(m.value)
        problem_sizes_host[g * 4 + 1] = Int32(n.value)
        problem_sizes_host[g * 4 + 2] = Int32(k.value)  # Logical K
        problem_sizes_host[g * 4 + 3] = 1

    var a_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var b_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var c_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfa_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)
    var sfb_ptrs_host = UnsafePointer[UInt64].alloc(max_groups)

    for g in range(max_groups):
        a_ptrs_host[g] = UInt64(Int(a_device.unsafe_ptr()))
        b_ptrs_host[g] = UInt64(Int(b_device.unsafe_ptr()))
        c_ptrs_host[g] = UInt64(Int(c_device.unsafe_ptr()))
        sfa_ptrs_host[g] = UInt64(Int(sfa_device.unsafe_ptr()))
        sfb_ptrs_host[g] = UInt64(Int(sfb_device.unsafe_ptr()))

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

    comptime problem_sizes_layout = Layout.row_major(max_groups, 4)
    var problem_sizes_tensor = LayoutTensor[
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

    comptime BM = mma_shape[0]
    comptime BN = mma_shape[1]
    var tiles_per_group = ceildiv(m.value, BM) * ceildiv(n.value, BN)
    var total_tiles = tiles_per_group * num_groups

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=8,
        cta_group=1,
        k_group_size=k_grp_size,
        num_accum_pipeline_stages=2,
    )

    # FLOPs use logical K (not packed)
    var total_flops = 2 * m.value * n.value * k.value * num_groups

    @parameter
    @__copy_capture(
        a_ptrs_tensor,
        b_ptrs_tensor,
        c_ptrs_tensor,
        sfa_ptrs_tensor,
        sfb_ptrs_tensor,
        problem_sizes_tensor,
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        total_tiles,
    )
    @always_inline
    fn bench_func(mut bencher: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
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
                problem_sizes_tensor,
                num_groups,
                total_tiles,
                a_tensor,
                b_tensor,
                c_tensor,
                sfa_tensor,
                sfb_tensor,
                ctx,
            )

        bencher.iter_custom[kernel_launch](ctx)

    var fmt = String("NVFP4") if is_fp4 else String("MXFP8")

    bench.bench_function[bench_func](
        BenchId(
            String(
                "STRUCTURED(",
                fmt,
                ",",
                mma_m,
                "x",
                mma_n,
                ",k_grp=",
                k_grp_size,
                ") : ",
                num_groups,
                " x ",
                m.value,
                " x ",
                n.value,
                " x ",
                k.value,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    # Cleanup
    a_host.free()
    b_host.free()
    sfa_host.free()
    sfb_host.free()
    problem_sizes_host.free()
    a_ptrs_host.free()
    b_ptrs_host.free()
    c_ptrs_host.free()
    sfa_ptrs_host.free()
    sfb_ptrs_host.free()


def main():
    print("=" * 70)
    print("Comparison: cuBLAS (per-group) vs Structured Grouped GEMM")
    print("=" * 70)
    print()
    print("Both sides use identical block-scaled formats (MXFP8 and NVFP4).")
    print("cuBLAS baseline: vendor_blas.matmul called once per group")
    print("Structured:      grouped_block_scaled_matmul (persistent)")
    print()

    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scales_dtype = MXFP8_SF_DTYPE

    comptime fp4_a_type = DType.uint8
    comptime fp4_b_type = DType.uint8
    comptime fp4_c_type = DType.bfloat16
    comptime fp4_scales_dtype = NVFP4_SF_DTYPE

    var b = Bench()

    with DeviceContext() as ctx:
        # =====================================================================
        # MXFP8: DeepSeek-V2 Prefill: 32 groups x 4096 x 4096 x 7168
        # =====================================================================
        print("=== MXFP8 Prefill: 32 groups x 4096 x 4096 x 7168 ===")

        bench_cublas_per_group[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[4096](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
        ](ctx, b)

        bench_structured_kernel[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[4096](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
        ](ctx, b)

        # =====================================================================
        # MXFP8: DeepSeek-V2 Decode: 32 groups x 128 x 4096 x 7168
        # =====================================================================
        print("\n=== MXFP8 Decode: 32 groups x 128 x 4096 x 7168 ===")

        bench_cublas_per_group[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[128](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
        ](ctx, b)

        bench_structured_kernel[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            m = static[128](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
        ](ctx, b)

        # =====================================================================
        # NVFP4: DeepSeek-V2 Prefill: 32 groups x 4096 x 4096 x 7168
        # =====================================================================
        print("\n=== NVFP4 Prefill: 32 groups x 4096 x 4096 x 7168 ===")

        bench_cublas_per_group[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            m = static[4096](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b)

        bench_structured_kernel[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            m = static[4096](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
            scaling_kind = UMMAKind.KIND_MXF4NVF4,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b)

        # =====================================================================
        # NVFP4: DeepSeek-V2 Decode: 32 groups x 128 x 4096 x 7168
        # =====================================================================
        print("\n=== NVFP4 Decode: 32 groups x 128 x 4096 x 7168 ===")

        bench_cublas_per_group[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            m = static[128](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b)

        bench_structured_kernel[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            m = static[128](),
            n = static[4096](),
            k = static[7168](),
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
            scaling_kind = UMMAKind.KIND_MXF4NVF4,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b)

    b.dump_report()
    print()
    print("=" * 70)
