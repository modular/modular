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
"""Performance benchmarking for grouped block-scaled GEMM (sm100_structured).

Benchmarks the grouped_block_scaled_matmul kernel with realistic MoE shapes
from DeepSeek-V2 and other models.

Usage:
    # Basic benchmark (standalone)
    mojo bench_grouped_block_scaled_gemm.mojo

    # With kbench (autotuning)
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
        max/kernels/benchmarks/autotune/bench_grouped_block_scaled_gemm.yaml
"""

from math import ceildiv
from memory import LegacyUnsafePointer
from sys import env_get_int, size_of
from time import perf_counter_ns

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from random import rand, seed
from internal_utils import arg_parse
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul import (
    grouped_block_scaled_matmul,
)

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


# =============================================================================
# Benchmark Configuration
# =============================================================================


fn _get_run_name[
    in_type: DType,
    out_type: DType,
](num_groups: Int, m_per_group: Int, n: Int, k: Int, cta_group: Int,) -> String:
    var mode_str = "1SM" if cta_group == 1 else "2SM"
    return String(
        "grouped_block_scaled_gemm(",
        in_type,
        "->",
        out_type,
        ") : ",
        num_groups,
        " x ",
        m_per_group,
        " x ",
        n,
        " x ",
        k,
        " [",
        mode_str,
        "]",
    )


# =============================================================================
# Main Benchmark Function
# =============================================================================


fn bench_grouped_block_scaled_gemm[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    num_groups: Int,
    transpose_b: Bool,
    cta_group: Int = 1,
    k_group_size: Int = 1,
    block_swizzle_size: Int = 8,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF8F6F4,
    sf_vector_size: Int = MXFP8_SF_VECTOR_SIZE,
](ctx: DeviceContext, mut bench: Bench, m_override: Int = 0) raises:
    """Benchmark grouped block-scaled GEMM with production-style configuration.

    Args:
        ctx: Device context.
        bench: Benchmark instance.
        m_override: Runtime M value. When > 0, overrides m.value for
            allocations and problem sizes. Used by kbench with dynamic M.
    """
    comptime SF_VECTOR_SIZE = sf_vector_size
    comptime max_groups = num_groups
    # FP4 packs 2 values per byte, so K dimension arrays are halved
    comptime is_fp4 = (a_type == DType.uint8)

    # MMA shape and cluster shape
    comptime mma_shape = Index(256, 256, 32) if cta_group == 2 else Index(
        128, 128, 32
    )
    comptime cluster_shape = Index(2, 1, 1) if cta_group == 2 else Index(
        1, 1, 1
    )

    # Use m_override if provided, otherwise m.value (compile-time)
    var M = m_override if m_override > 0 else m.value

    # For FP4, K dimension in arrays is halved (2 values packed per byte)
    comptime k_pack = 2 if is_fp4 else 1
    comptime k_array_dim = k.dim // k_pack
    var k_array_val = k.value // k_pack

    # Compute sizes (using packed K for FP4)
    var a_size = M * k_array_val
    var b_size = n.value * k_array_val if transpose_b else k_array_val * n.value
    var c_size = M * n.value

    # Scale factors always use logical K
    var a_scales_total = (
        ceildiv(M, SF_MN_GROUP_SIZE)
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

    # Static shapes (using packed K for A/B arrays)
    comptime static_a_shape = DimList(m.dim, k_array_dim)
    comptime static_b_shape = DimList(
        n.dim, k_array_dim
    ) if transpose_b else DimList(k_array_dim, n.dim)
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

    var dynamic_a_shape = DimList(M, k_array_val)
    var dynamic_b_shape = DimList(
        n.value, k_array_val
    ) if transpose_b else DimList(k_array_val, n.value)
    var dynamic_c_shape = DimList(M, n.value)
    var dynamic_a_scales_shape = DimList(
        ceildiv(M, SF_MN_GROUP_SIZE),
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

    # Allocate tensors
    var a_host = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var sfa_host = UnsafePointer[Scalar[scales_dtype]].alloc(a_scales_total)
    var sfb_host = UnsafePointer[Scalar[scales_dtype]].alloc(b_scales_total)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var sfa_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_total)
    var sfb_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_total)

    # Initialize
    seed(42)
    rand(a_host, a_size)
    rand(b_host, b_size)
    for i in range(c_size):
        c_host[i] = 0
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        sfa_host[i] = scale_one
    for i in range(b_scales_total):
        sfb_host[i] = scale_one

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)
    ctx.enqueue_copy(sfa_device, sfa_host)
    ctx.enqueue_copy(sfb_device, sfb_host)
    ctx.synchronize()

    # Create NDBuffers and LayoutTensors
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

    # Setup pointer arrays
    var problem_sizes_host = UnsafePointer[Int32].alloc(max_groups * 4)
    for g in range(max_groups):
        problem_sizes_host[g * 4 + 0] = Int32(M)
        problem_sizes_host[g * 4 + 1] = Int32(n.value)
        problem_sizes_host[g * 4 + 2] = Int32(k.value)
        problem_sizes_host[g * 4 + 3] = 1

    var problem_sizes_device = ctx.enqueue_create_buffer[DType.int32](
        max_groups * 4
    )
    ctx.enqueue_copy(problem_sizes_device, problem_sizes_host)

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

    comptime BM = mma_shape[0] // cta_group
    comptime BN = mma_shape[1] // cta_group
    var tiles_per_group = ceildiv(M, BM) * ceildiv(n.value, BN)
    var total_tiles = tiles_per_group * num_groups

    # Configuration matching production benchmark
    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=2,
    )

    # Total FLOPs for all groups
    var total_flops = 2 * M * n.value * k.value * num_groups

    @parameter
    @__copy_capture(
        a_ptrs_tensor,
        b_ptrs_tensor,
        c_ptrs_tensor,
        sfa_ptrs_tensor,
        sfb_ptrs_tensor,
        problem_sizes_tensor_host,
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
                problem_sizes_tensor_host,
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

    bench.bench_function[bench_func](
        BenchId(
            _get_run_name[a_type, c_type](
                num_groups, M, n.value, k.value, cta_group
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    # Cleanup
    a_host.free()
    b_host.free()
    c_host.free()
    sfa_host.free()
    sfb_host.free()
    problem_sizes_host.free()
    a_ptrs_host.free()
    b_ptrs_host.free()
    c_ptrs_host.free()
    sfa_ptrs_host.free()
    sfb_ptrs_host.free()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    # Compile-time parameters (from kbench YAML or defaults)
    comptime N = env_get_int["N", 0]()
    comptime K = env_get_int["K", 0]()
    comptime num_groups = env_get_int["num_groups", 0]()
    comptime cta_group = env_get_int["cta_group", 1]()
    comptime k_group_size = env_get_int["k_group_size", 1]()
    comptime block_swizzle_size = env_get_int["block_swizzle_size", 8]()

    # Runtime parameters (from kbench YAML $-prefixed or defaults)
    var M = Int(arg_parse("M", 0))

    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    var b = Bench()

    with DeviceContext() as ctx:

        @parameter
        if N > 0 and K > 0 and num_groups > 0:
            # kbench mode: use env parameters
            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                m = dynamic(0),
                n = static[N](),
                k = static[K](),
                num_groups=num_groups,
                transpose_b=transpose_b,
                cta_group=cta_group,
                k_group_size=k_group_size,
                block_swizzle_size=block_swizzle_size,
            ](ctx, b, M)
        else:
            # Standalone mode: run default shapes
            print("=" * 70)
            print("Benchmark: Grouped Block-Scaled GEMM (sm100_structured)")
            print("=" * 70)

            print("\n=== DeepSeek-V2 MoE Decode Shapes (small M) ===")

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                m = static[128](),
                n = static[4096](),
                k = static[7168](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b)

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                m = static[128](),
                n = static[7168](),
                k = static[2048](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b)

            print("\n=== DeepSeek-V2 MoE Prefill Shapes (large M) ===")

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                m = static[4096](),
                n = static[4096](),
                k = static[7168](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b)

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                m = static[4096](),
                n = static[7168](),
                k = static[2048](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b)

            # === NVFP4 Benchmarks ===
            comptime fp4_a_type = DType.uint8
            comptime fp4_b_type = DType.uint8
            comptime fp4_c_type = DType.bfloat16
            comptime fp4_scales_dtype = NVFP4_SF_DTYPE

            print("\n=== NVFP4 DeepSeek-V2 MoE Decode Shapes (small M) ===")

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                m = static[128](),
                n = static[4096](),
                k = static[7168](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind = UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b)

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                m = static[128](),
                n = static[7168](),
                k = static[2048](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind = UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b)

            print("\n=== NVFP4 DeepSeek-V2 MoE Prefill Shapes (large M) ===")

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                m = static[4096](),
                n = static[4096](),
                k = static[7168](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind = UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b)

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                m = static[4096](),
                n = static[7168](),
                k = static[2048](),
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind = UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b)

    b.dump_report()
    print("\n" + "=" * 70)
