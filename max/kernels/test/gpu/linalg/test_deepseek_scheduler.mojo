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

from collections import OptionalReg
from math import ceildiv

import linalg.vendor_blas
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    assert_with_measure,
    random,
    zero,
)
from internal_utils._measure import relative_difference
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.matmul_sm90 import warp_specialize_gemm_with_multicasting
from linalg.matmul_tile_scheduler import MatmulSchedule
from linalg.utils_gpu import MatmulConfig

from utils.index import Index, IndexList

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4


fn test_warp_specialize_gemm_with_multicasting[
    block_tile_shape: IndexList[3],
    a_type: DType,
    b_type: DType,
    c_type: DType,
    cluster_shape: IndexList[3],
    num_pipeline_stages: Int = 4,
    transpose_b: Bool = True,
    partitioned_multicast: Bool = False,
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
](
    ctx: DeviceContext,
    handle: linalg.vendor_blas.Handle,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
) raises:
    var M = m.value
    var N = n.value
    var K = k.value

    alias CLUSTER_N = cluster_shape[0]
    alias CLUSTER_M = cluster_shape[1]

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
    random(a_host.tensor)
    random(b_host.tensor)

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    # alias block_tile_shape = Index(
    #     BM, BN, 128
    # ) if a_type is DType.float8_e4m3fn else Index(BM, BN, 64)
    alias wgmma_shape = Index(
        64, BN, 32
    ) if a_type is DType.float8_e4m3fn else Index(64, BN, 16)

    alias num_consumer: Int = 1 if BM == 64 else 2

    print(
        "wgmma_n",
        BN,
        a_type,
        "x",
        b_type,
        "x",
        c_type,
        " : PROBLEM SHAPE (M,N,K): (",
        M,
        "x",
        N,
        "x",
        K,
        ") - ",
        "BLOCKS SHAPE (BM,BN,BK): (",
        BM,
        "x",
        BN,
        "x",
        BK,
        ") - ",
        "CLUSTER DIMS (M,N): (",
        CLUSTER_M,
        "x",
        CLUSTER_N,
        ") NUM CONSUMERS: ",
        num_consumer,
        " NUM PIPELINE STAGES: ",
        num_pipeline_stages,
        " MULTICAST MODE: ",
        "PARTITIONED" if partitioned_multicast else "BROADCAST",
    )

    debug_assert(
        (ceildiv(M, BM) % (CLUSTER_M)) == 0,
        String(
            "Number of blocks on M axis should be multiple of cluster dim. M",
            "(M // BM=",
            String(M // BM),
            ") CLUSTER SIZE:",
            String(CLUSTER_M),
        ),
    )

    debug_assert(
        (ceildiv(N, BN) % (CLUSTER_N)) == 0,
        String(
            "Number of blocks on M axis should be multiple of cluster dim. N",
            "N // BN=(",
            String(N // BN),
            ") CLUSTER SIZE:",
            String(CLUSTER_N),
        ),
    )

    alias matmul_config = MatmulConfig[
        a_type, b_type, c_type, transpose_b, mma_shape=wgmma_shape
    ](
        block_tile_shape=block_tile_shape,
        cluster_shape=cluster_shape,
        num_pipeline_stages=num_pipeline_stages,
        num_consumer=num_consumer,
        partitioned_multicast=partitioned_multicast,
    )

    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=matmul_config,
        schedule=schedule,
        grid_shape=grid_shape,
        use_tma_store=use_tma_store,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
    )

    ctx.synchronize()

    @parameter
    if a_type is DType.float8_e4m3fn:
        vendor_blas.matmul(
            ctx,
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )
    else:
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

    assert_with_measure[relative_difference](
        c_host.tensor, c_host_ref.tensor, threshold=0.001
    )

    alias rtol = 1e-2
    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=rtol,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


fn main() raises:
    with DeviceContext() as ctx:
        # NOTE: please note that cublaslt handle should be used for fp8-e4m3fn and cublas handle for bfloat16
        # because cublas does not support float8-e4m3fn. Also, fp8 tests should be run first and then bfloat16 tests
        # otherwise we will get unhandled exception error.
        print("FP8-E4M3FN GEMM TESTS")
        with vendor_blas.Handle[
            vendor_blas.Backend.CUBLASLT
        ]() as cublaslt_handle:
            test_warp_specialize_gemm_with_multicasting[
                Index(128, 80, 128),
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=6,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[2560](),
                static[8192](),
            )
            test_warp_specialize_gemm_with_multicasting[
                Index(64, 128, 128),
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=8,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[16384](),
                static[2048](),
            )
            test_warp_specialize_gemm_with_multicasting[
                Index(64, 48, 128),
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=8,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[2304](),
                static[16384](),
            )
            test_warp_specialize_gemm_with_multicasting[
                Index(128, 128, 128),
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=4,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[13312](),
                static[16384](),
            )
            test_warp_specialize_gemm_with_multicasting[
                Index(128, 128, 128),
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=4,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[16384](),
                static[6656](),
            )

        print("BFLOAT16 GEMM TESTS")
        with vendor_blas.Handle[vendor_blas.Backend.CUBLAS]() as cublas_handle:
            test_warp_specialize_gemm_with_multicasting[
                Index(64, 32, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=8,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(64), static[2560](), static[8192]())
            test_warp_specialize_gemm_with_multicasting[
                Index(64, 40, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=8,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(128), static[2560](), static[8192]())
            test_warp_specialize_gemm_with_multicasting[
                Index(64, 80, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=8,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(256), static[2560](), static[8192]())
            test_warp_specialize_gemm_with_multicasting[
                Index(64, 160, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=7,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(512), static[2560](), static[8192]())
            test_warp_specialize_gemm_with_multicasting[
                Index(64, 160, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=5,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(1024), static[2560](), static[8192]())
            test_warp_specialize_gemm_with_multicasting[
                Index(128, 160, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=5,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(2048), static[2560](), static[8192]())
            test_warp_specialize_gemm_with_multicasting[
                Index(128, 256, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=4,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(4096), static[2560](), static[8192]())
            test_warp_specialize_gemm_with_multicasting[
                Index(128, 256, 64),
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_pipeline_stages=4,
                partitioned_multicast=False,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](ctx, cublas_handle, dynamic(8192), static[2560](), static[8192]())
