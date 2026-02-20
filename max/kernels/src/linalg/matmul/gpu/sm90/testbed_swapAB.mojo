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
"""Testbed for comparing swapAB vs normal matmul execution.

swapAB is an internal optimization where:
- A and B operands are swapped inside the kernel
- The output C tile is transposed on write-out
- The final result C[M,N] should be identical to the normal kernel

Both kernels compute: C[M,N] = A[M,K] @ B[N,K]^T
The swapAB version just does it via: (B @ A^T)^T stored transposed = A @ B^T
"""

from math import ceildiv
from sys import align_of

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from random import rand
from internal_utils._utils import ValOrDim
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from utils.index import Index, IndexList

from .config import MatmulConfig as MatmulConfigSM90
from ....utils_gpu import MatmulConfig as BaseMatmulConfig
from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ..tile_scheduler import MatmulSchedule
from .matmul import warp_specialize_gemm_with_multicasting
from ...vendor.blas import matmul as vendor_matmul


fn test_matmul_sm90_swapAB_comparison[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    # Compile-time configs
    config: MatmulConfigSM90[a_type, b_type, c_type, True],
    config_swapAB: MatmulConfigSM90[a_type, b_type, c_type, True],
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    """Compare matmul results between normal execution and swapAB execution.

    Both compute: C[M,N] = A[M,K] @ B[N,K]^T
    swapAB internally swaps A/B and transposes C on store, but result should match.

    Args:
        ctx: The device context.
        m: The M dimension (can be static or dynamic).
        n: The N dimension (can be static or dynamic).
        k: The K dimension (can be static or dynamic).
    """
    comptime transpose_b = True
    var M = m.value
    var N = n.value
    var K = k.value

    # Convert SM90 configs to base configs for the kernel
    comptime base_config = config.to_base_config()
    comptime base_config_swapAB = config_swapAB.to_base_config()

    # Extract block tile shapes for assertions
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime CLUSTER_M = config.cluster_shape[1]
    comptime CLUSTER_N = config.cluster_shape[0]

    comptime BM_SWAPAB = config_swapAB.block_tile_shape[0]
    comptime BN_SWAPAB = config_swapAB.block_tile_shape[1]
    comptime CLUSTER_M_SWAPAB = config_swapAB.cluster_shape[1]
    comptime CLUSTER_N_SWAPAB = config_swapAB.cluster_shape[0]

    # Static shapes - C shape is the same for both kernels
    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim)
    comptime static_c_shape = DimList(m.dim, n.dim)  # [M, N] for both

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)  # [M, N] for both

    # Calculate sizes
    var a_size = M * K
    var b_size = N * K
    var c_size = M * N

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_normal_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_swapAB_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)

    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_normal_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_normal_host_ptr, dynamic_c_shape
    )
    var c_swapAB_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_swapAB_host_ptr, dynamic_c_shape
    )

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_normal_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_swapAB_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    var a_device = NDBuffer[a_type, 2, _, static_a_shape](
        a_dev_buffer.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = NDBuffer[b_type, 2, _, static_b_shape](
        b_dev_buffer.unsafe_ptr(), dynamic_b_shape
    )
    var c_normal_device = NDBuffer[c_type, 2, _, static_c_shape](
        c_normal_dev_buffer.unsafe_ptr(), dynamic_c_shape
    )
    var c_swapAB_device = NDBuffer[c_type, 2, _, static_c_shape](
        c_swapAB_dev_buffer.unsafe_ptr(), dynamic_c_shape
    )

    # Initialize matmul operands with random values
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

    # Move operands to the Device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)

    # Extract more config values for printing
    comptime BK = config.block_tile_shape[2]
    comptime BK_SWAPAB = config_swapAB.block_tile_shape[2]

    print(
        "=== SwapAB Comparison Test ===\n",
        "PROBLEM SHAPE (M,N,K): (",
        M,
        "x",
        N,
        "x",
        K,
        ")\n",
        "Types:",
        a_type,
        "x",
        b_type,
        "->",
        c_type,
        "\n\n--- Normal Kernel Config ---",
        "\nwgmma_shape:",
        config.mma_shape,
        "\nBLOCK SHAPE (BM,BN,BK): (",
        BM,
        "x",
        BN,
        "x",
        BK,
        ")",
        "\nCLUSTER DIMS (M,N): (",
        CLUSTER_M,
        "x",
        CLUSTER_N,
        ")",
        "\nNUM CONSUMERS:",
        config.num_consumer,
        "NUM PIPELINE STAGES:",
        config.num_pipeline_stages,
        "\n\n--- SwapAB Kernel Config ---",
        "\nwgmma_shape:",
        config_swapAB.mma_shape,
        "\nBLOCK SHAPE (BM,BN,BK): (",
        BM_SWAPAB,
        "x",
        BN_SWAPAB,
        "x",
        BK_SWAPAB,
        ")",
        "\nCLUSTER DIMS (M,N): (",
        CLUSTER_M_SWAPAB,
        "x",
        CLUSTER_N_SWAPAB,
        ")",
        "\nNUM CONSUMERS:",
        config_swapAB.num_consumer,
        "NUM PIPELINE STAGES:",
        config_swapAB.num_pipeline_stages,
        "\n",
    )

    # Assertions for normal kernel
    debug_assert(
        (ceildiv(M, BM) % CLUSTER_M) == 0,
        String(
            (
                "Normal: Number of blocks on M axis should be multiple of"
                " cluster dim. M"
            ),
            "(M // BM=",
            String(M // BM),
            ") CLUSTER SIZE:",
            String(CLUSTER_M),
        ),
    )

    debug_assert(
        (ceildiv(N, BN) % CLUSTER_N) == 0,
        String(
            (
                "Normal: Number of blocks on N axis should be multiple of"
                " cluster dim. N"
            ),
            "(N // BN=",
            String(N // BN),
            ") CLUSTER SIZE:",
            String(CLUSTER_N),
        ),
    )

    # Assertions for swapAB kernel
    debug_assert(
        (ceildiv(M, BM_SWAPAB) % CLUSTER_M_SWAPAB) == 0,
        String(
            (
                "SwapAB: Number of blocks on M axis should be multiple of"
                " cluster dim. M"
            ),
            "(M // BM=",
            String(M // BM_SWAPAB),
            ") CLUSTER SIZE:",
            String(CLUSTER_M_SWAPAB),
        ),
    )

    debug_assert(
        (ceildiv(N, BN_SWAPAB) % CLUSTER_N_SWAPAB) == 0,
        String(
            (
                "SwapAB: Number of blocks on N axis should be multiple of"
                " cluster dim. N"
            ),
            "(N // BN=",
            String(N // BN_SWAPAB),
            ") CLUSTER SIZE:",
            String(CLUSTER_N_SWAPAB),
        ),
    )

    # =========================================================================
    # Run NORMAL matmul: C[M,N] = A[M,K] @ B[N,K]^T
    # =========================================================================
    print("Running normal matmul (swapAB=False)...")
    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=base_config,
        schedule = MatmulSchedule.NONE,
        swapAB=False,
    ](
        c_normal_device,
        a_device,
        b_device,
        ctx,
    )

    ctx.synchronize()

    # =========================================================================
    # Run SWAPAB matmul: same C[M,N] = A[M,K] @ B[N,K]^T
    # Internally swaps A/B and transposes C tile on store
    # =========================================================================
    print("Running swapAB matmul (swapAB=True)...")
    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=base_config_swapAB,
        schedule = MatmulSchedule.NONE,
        swapAB=True,
    ](
        c_swapAB_device,
        a_device,
        b_device,
        ctx,
    )

    # Copy results back to host
    ctx.enqueue_copy(c_normal_host_ptr, c_normal_dev_buffer)
    ctx.enqueue_copy(c_swapAB_host_ptr, c_swapAB_dev_buffer)
    ctx.synchronize()

    # =========================================================================
    # Compare results: Both should be identical
    # =========================================================================
    print("Comparing results (should be identical)...")

    # Print both results for debugging
    # print("\n=== Normal Result (C = A @ B) ===")
    # for i in range(M):
    #     for j in range(N):
    #         print(c_normal_host[Index(i, j)], end=" ")
    #     print()

    # print("\n=== SwapAB Result (C = (B^T @ A^T)^T) ===")
    # for i in range(M):
    #     for j in range(N):
    #         print(c_swapAB_host[Index(i, j)], end=" ")
    #     print()

    var max_diff: Float64 = 0.0
    var total_diff: Float64 = 0.0
    var num_mismatches = 0

    for i in range(M):
        for j in range(N):
            var val_swapAB = c_swapAB_host[Index(i, j)].cast[DType.float64]()
            var val_normal = c_normal_host[Index(i, j)].cast[DType.float64]()
            var diff = abs(val_swapAB - val_normal)

            if diff > 0.01:  # Threshold for counting mismatches
                num_mismatches += 1
                if num_mismatches <= 10:  # Print first 10 mismatches
                    print(
                        "  Mismatch at [",
                        i,
                        ",",
                        j,
                        "]: swapAB=",
                        val_swapAB,
                        "normal=",
                        val_normal,
                        "diff=",
                        diff,
                    )

            if diff > max_diff:
                max_diff = diff
            total_diff += diff

    var avg_diff = total_diff / Float64(M * N)
    print(
        "\n=== Comparison Results ===",
        "\nTotal elements:",
        M * N,
        "\nMismatches (diff > 0.01):",
        num_mismatches,
        "\nMax difference:",
        max_diff,
        "\nAvg difference:",
        avg_diff,
    )

    # Formal assertion
    comptime rtol = 1e-2
    assert_almost_equal(
        c_swapAB_host.data,
        c_normal_host.data,
        c_swapAB_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    print("=== SwapAB comparison test PASSED ===\n")

    # Cleanup host pointers
    a_host_ptr.free()
    b_host_ptr.free()
    c_normal_host_ptr.free()
    c_swapAB_host_ptr.free()

    # Consume device buffers
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_normal_dev_buffer^
    _ = c_swapAB_dev_buffer^


fn test_matmul_sm90_swapAB_comparison_v2[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    # Config parameters for normal kernel (compile-time)
    BM: Int,
    BN: Int,
    BK: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    num_pipeline_stages: UInt,
    num_consumer: UInt,
    k_group_size: UInt = 1,
    num_k_partitions: UInt = 1,
    partitioned_multicast: Bool = False,
    # Config parameters for swapAB kernel (compile-time, defaults to normal values)
    BM_SWAPAB: Int = BM,
    BN_SWAPAB: Int = BN,
    BK_SWAPAB: Int = BK,
    MMA_M_SWAPAB: Int = MMA_M,
    MMA_N_SWAPAB: Int = MMA_N,
    MMA_K_SWAPAB: Int = MMA_K,
    num_pipeline_stages_swapAB: UInt = num_pipeline_stages,
    num_consumer_swapAB: UInt = num_consumer,
    k_group_size_swapAB: UInt = k_group_size,
    num_k_partitions_swapAB: UInt = num_k_partitions,
    partitioned_multicast_swapAB: Bool = partitioned_multicast,
    # Use vendor matmul (cuBLAS) as reference instead of normal kernel
    use_vendor_reference: Bool = False,
    # Epilogue support
    default_epilogue: Bool = False,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    """Compare matmul results between normal execution and swapAB execution.

    This version accepts config parameters directly as compile-time values
    and builds configs internally.

    Both compute: C[M,N] = A[M,K] @ B[N,K]^T
    swapAB internally swaps A/B and transposes C on store, but result should match.

    Parameters:
        a_type: Data type of matrix A.
        b_type: Data type of matrix B.
        c_type: Data type of output matrix C.
        BM: Block tile M dimension for normal kernel.
        BN: Block tile N dimension for normal kernel.
        BK: Block tile K dimension for normal kernel.
        MMA_M: MMA M dimension for normal kernel.
        MMA_N: MMA N dimension for normal kernel.
        MMA_K: MMA K dimension for normal kernel.
        num_pipeline_stages: Number of pipeline stages for normal kernel.
        num_consumer: Number of consumers for normal kernel.
        k_group_size: K group size for normal kernel.
        num_k_partitions: Number of K partitions for normal kernel.
        partitioned_multicast: Partitioned multicast for normal kernel.
        BM_SWAPAB: Block tile M dimension for swapAB kernel.
        BN_SWAPAB: Block tile N dimension for swapAB kernel.
        BK_SWAPAB: Block tile K dimension for swapAB kernel.
        MMA_M_SWAPAB: MMA M dimension for swapAB kernel.
        MMA_N_SWAPAB: MMA N dimension for swapAB kernel.
        MMA_K_SWAPAB: MMA K dimension for swapAB kernel.
        num_pipeline_stages_swapAB: Number of pipeline stages for swapAB kernel.
        num_consumer_swapAB: Number of consumers for swapAB kernel.
        k_group_size_swapAB: K group size for swapAB kernel.
        num_k_partitions_swapAB: Number of K partitions for swapAB kernel.
        partitioned_multicast_swapAB: Partitioned multicast for swapAB kernel.
        use_vendor_reference: If True, use vendor matmul (cuBLAS) as reference
            instead of normal kernel.
        default_epilogue: If True, use default epilogue function that stores
            directly to output tensor.
        elementwise_compute_lambda_fn: Optional compute lambda function to apply
            to each element before storing.

    Args:
        ctx: The device context.
        m: The M dimension (can be static or dynamic).
        n: The N dimension (can be static or dynamic).
        k: The K dimension (can be static or dynamic).
    """
    comptime transpose_b = True
    var M = m.value
    var N = n.value
    var K = k.value

    # Build compile-time configs directly using BaseMatmulConfig
    comptime base_config = BaseMatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=Index(BM, BN, BK),
        warp_tile_shape=Index(MMA_M, MMA_N, BK),
        mma_shape=Index(MMA_M, MMA_N, MMA_K),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=num_pipeline_stages,
        num_k_partitions=num_k_partitions,
        k_group_size=k_group_size,
        num_warp_k_partitions=1,
        num_consumer=num_consumer,
        partitioned_multicast=partitioned_multicast,
    )

    comptime base_config_swapAB = BaseMatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=Index(BM_SWAPAB, BN_SWAPAB, BK_SWAPAB),
        warp_tile_shape=Index(MMA_M_SWAPAB, MMA_N_SWAPAB, BK_SWAPAB),
        mma_shape=Index(MMA_M_SWAPAB, MMA_N_SWAPAB, MMA_K_SWAPAB),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=num_pipeline_stages_swapAB,
        num_k_partitions=num_k_partitions_swapAB,
        k_group_size=k_group_size_swapAB,
        num_warp_k_partitions=1,
        num_consumer=num_consumer_swapAB,
        partitioned_multicast=partitioned_multicast_swapAB,
    )

    # Extract cluster shapes for assertions (hardcoded to 1 for now)
    comptime CLUSTER_M = 1
    comptime CLUSTER_N = 1
    comptime CLUSTER_M_SWAPAB = 1
    comptime CLUSTER_N_SWAPAB = 1

    # Static shapes - C shape is the same for both kernels
    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim)
    comptime static_c_shape = DimList(m.dim, n.dim)  # [M, N] for both

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)  # [M, N] for both

    # Calculate sizes
    var a_size = M * K
    var b_size = N * K
    var c_size = M * N

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_normal_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_swapAB_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)

    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_normal_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_normal_host_ptr, dynamic_c_shape
    )
    var c_swapAB_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_swapAB_host_ptr, dynamic_c_shape
    )

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_normal_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_swapAB_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    var a_device = NDBuffer[a_type, 2, _, static_a_shape](
        a_dev_buffer.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = NDBuffer[b_type, 2, _, static_b_shape](
        b_dev_buffer.unsafe_ptr(), dynamic_b_shape
    )
    var c_normal_device = NDBuffer[c_type, 2, _, static_c_shape](
        c_normal_dev_buffer.unsafe_ptr(), dynamic_c_shape
    )
    var c_swapAB_device = NDBuffer[c_type, 2, _, static_c_shape](
        c_swapAB_dev_buffer.unsafe_ptr(), dynamic_c_shape
    )

    # Initialize matmul operands with random values
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

    # Move operands to the Device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)

    print(
        "=== SwapAB Comparison Test V2 ===\n",
        "PROBLEM SHAPE (M,N,K): (",
        M,
        "x",
        N,
        "x",
        K,
        ")\n",
        "Types:",
        a_type,
        "x",
        b_type,
        "->",
        c_type,
        "\n\n--- Normal Kernel Config ---",
        "\nwgmma_shape: (",
        MMA_M,
        "x",
        MMA_N,
        "x",
        MMA_K,
        ")",
        "\nBLOCK SHAPE (BM,BN,BK): (",
        BM,
        "x",
        BN,
        "x",
        BK,
        ")",
        "\nCLUSTER DIMS (M,N): (",
        CLUSTER_M,
        "x",
        CLUSTER_N,
        ")",
        "\nNUM CONSUMERS:",
        num_consumer,
        "NUM PIPELINE STAGES:",
        num_pipeline_stages,
        "\n\n--- SwapAB Kernel Config ---",
        "\nwgmma_shape: (",
        MMA_M_SWAPAB,
        "x",
        MMA_N_SWAPAB,
        "x",
        MMA_K_SWAPAB,
        ")",
        "\nBLOCK SHAPE (BM,BN,BK): (",
        BM_SWAPAB,
        "x",
        BN_SWAPAB,
        "x",
        BK_SWAPAB,
        ")",
        "\nCLUSTER DIMS (M,N): (",
        CLUSTER_M_SWAPAB,
        "x",
        CLUSTER_N_SWAPAB,
        ")",
        "\nNUM CONSUMERS:",
        num_consumer_swapAB,
        "NUM PIPELINE STAGES:",
        num_pipeline_stages_swapAB,
        "\n",
    )

    # Assertions for normal kernel
    debug_assert(
        (ceildiv(M, BM) % CLUSTER_M) == 0,
        String(
            (
                "Normal: Number of blocks on M axis should be multiple of"
                " cluster dim. M"
            ),
            "(M // BM=",
            String(M // BM),
            ") CLUSTER SIZE:",
            String(CLUSTER_M),
        ),
    )

    debug_assert(
        (ceildiv(N, BN) % CLUSTER_N) == 0,
        String(
            (
                "Normal: Number of blocks on N axis should be multiple of"
                " cluster dim. N"
            ),
            "(N // BN=",
            String(N // BN),
            ") CLUSTER SIZE:",
            String(CLUSTER_N),
        ),
    )

    # Assertions for swapAB kernel
    debug_assert(
        (ceildiv(M, BM_SWAPAB) % CLUSTER_M_SWAPAB) == 0,
        String(
            (
                "SwapAB: Number of blocks on M axis should be multiple of"
                " cluster dim. M"
            ),
            "(M // BM=",
            String(M // BM_SWAPAB),
            ") CLUSTER SIZE:",
            String(CLUSTER_M_SWAPAB),
        ),
    )

    debug_assert(
        (ceildiv(N, BN_SWAPAB) % CLUSTER_N_SWAPAB) == 0,
        String(
            (
                "SwapAB: Number of blocks on N axis should be multiple of"
                " cluster dim. N"
            ),
            "(N // BN=",
            String(N // BN_SWAPAB),
            ") CLUSTER SIZE:",
            String(CLUSTER_N_SWAPAB),
        ),
    )

    # =========================================================================
    # Set up epilogue functions if requested
    # =========================================================================
    @parameter
    @always_inline
    @__copy_capture(c_normal_device)
    fn epilogue_fn_normal[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_normal_device.store[alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    @parameter
    @always_inline
    @__copy_capture(c_swapAB_device)
    fn epilogue_fn_swapAB[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_swapAB_device.store[alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    comptime elf_normal = Optional[elementwise_epilogue_type](
        epilogue_fn_normal
    ) if default_epilogue and elementwise_compute_lambda_fn is None else None

    comptime elf_swapAB = Optional[elementwise_epilogue_type](
        epilogue_fn_swapAB
    ) if default_epilogue and elementwise_compute_lambda_fn is None else None

    # =========================================================================
    # Run REFERENCE matmul: C[M,N] = A[M,K] @ B[N,K]^T
    # Either normal kernel or vendor matmul (cuBLAS)
    # =========================================================================
    comptime if use_vendor_reference:
        print("Running vendor matmul (cuBLAS) as reference...")
        vendor_matmul(
            ctx,
            c_normal_device,
            a_device,
            b_device,
            c_row_major=True,
            transpose_b=transpose_b,
        )
    else:
        comptime if default_epilogue or elementwise_compute_lambda_fn:
            print(
                "Running normal matmul (swapAB=False) with epilogue as"
                " reference..."
            )
        else:
            print("Running normal matmul (swapAB=False) as reference...")
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            config=base_config,
            schedule = MatmulSchedule.NONE,
            swapAB=False,
            elementwise_lambda_fn=elf_normal,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ](
            c_normal_device,
            a_device,
            b_device,
            ctx,
        )

    ctx.synchronize()

    # =========================================================================
    # Run SWAPAB matmul: same C[M,N] = A[M,K] @ B[N,K]^T
    # Internally swaps A/B and transposes C tile on store
    # =========================================================================
    comptime if default_epilogue or elementwise_compute_lambda_fn:
        print("Running swapAB matmul (swapAB=True) with epilogue...")
    else:
        print("Running swapAB matmul (swapAB=True)...")
    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=base_config_swapAB,
        schedule = MatmulSchedule.NONE,
        swapAB=True,
        elementwise_lambda_fn=elf_swapAB,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    ](
        c_swapAB_device,
        a_device,
        b_device,
        ctx,
    )

    # Copy results back to host
    ctx.enqueue_copy(c_normal_host_ptr, c_normal_dev_buffer)
    ctx.enqueue_copy(c_swapAB_host_ptr, c_swapAB_dev_buffer)
    ctx.synchronize()

    # =========================================================================
    # Apply compute lambda to vendor reference if needed
    # (vendor matmul doesn't apply epilogue, so we apply it on CPU)
    # =========================================================================
    comptime if use_vendor_reference and elementwise_compute_lambda_fn:
        print("Applying compute lambda to vendor reference on CPU...")
        comptime compute_lambda = elementwise_compute_lambda_fn.value()
        for i in range(M):
            for j in range(N):
                c_normal_host[Index(i, j)] = compute_lambda(
                    IndexList[2](i, j),
                    c_normal_host[Index(i, j)],
                )

    # =========================================================================
    # Compare results: Both should be identical
    # =========================================================================
    comptime if use_vendor_reference:
        comptime if elementwise_compute_lambda_fn:
            print(
                "Comparing swapAB results against vendor matmul + CPU"
                " epilogue..."
            )
        else:
            print("Comparing swapAB results against vendor matmul (cuBLAS)...")
    else:
        print("Comparing swapAB results against normal kernel...")

    # Print both results for debugging
    # print("\n=== Reference Result ===")
    # for i in range(M):
    #     for j in range(N):
    #         print(c_normal_host[Index(i, j)], end=" ")
    #     print()

    # print("\n=== SwapAB Result (C = (B^T @ A^T)^T) ===")
    # for i in range(M):
    #     for j in range(N):
    #         print(c_swapAB_host[Index(i, j)], end=" ")
    #     print()

    var max_diff: Float64 = 0.0
    var total_diff: Float64 = 0.0
    var num_mismatches = 0

    comptime ref_name = "vendor" if use_vendor_reference else "normal"

    for i in range(M):
        for j in range(N):
            var val_swapAB = c_swapAB_host[Index(i, j)].cast[DType.float64]()
            var val_ref = c_normal_host[Index(i, j)].cast[DType.float64]()
            var diff = abs(val_swapAB - val_ref)

            if diff > 0.01:  # Threshold for counting mismatches
                num_mismatches += 1
                if num_mismatches <= 10:  # Print first 10 mismatches
                    print(
                        "  Mismatch at [",
                        i,
                        ",",
                        j,
                        "]: swapAB=",
                        val_swapAB,
                        ref_name,
                        "=",
                        val_ref,
                        "diff=",
                        diff,
                    )

            if diff > max_diff:
                max_diff = diff
            total_diff += diff

    var avg_diff = total_diff / Float64(M * N)
    print(
        "\n=== Comparison Results ===",
        "\nReference:",
        "vendor (cuBLAS)" if use_vendor_reference else "normal kernel",
        "\nTotal elements:",
        M * N,
        "\nMismatches (diff > 0.01):",
        num_mismatches,
        "\nMax difference:",
        max_diff,
        "\nAvg difference:",
        avg_diff,
    )

    # Formal assertion
    comptime rtol = 1e-2
    assert_almost_equal(
        c_swapAB_host.data,
        c_normal_host.data,
        c_swapAB_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    print("=== SwapAB comparison test V2 PASSED ===\n")

    # Cleanup host pointers
    a_host_ptr.free()
    b_host_ptr.free()
    c_normal_host_ptr.free()
    c_swapAB_host_ptr.free()

    # Consume device buffers
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_normal_dev_buffer^
    _ = c_swapAB_dev_buffer^
