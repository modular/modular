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
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug %s
# ALLOW_RETRIES: 5

from collections import OptionalReg
from math import ceildiv
from sys import alignof, simdwidthof, sizeof
from sys._assembly import inlined_assembly

import linalg.vendor_blas
from buffer.dimlist import Dim, DimList, _make_tuple
from gpu import WARP_SIZE, barrier
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
    elect_one_sync,
)
from gpu.host import DeviceContext
from gpu.host import Dim as ClusterDim
from gpu.host import FuncAttribute
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.id import block_dim, block_idx, thread_idx
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    fill,
    random,
    zero,
    assert_with_measure,
)
from internal_utils._utils import ValOrDim, dynamic, static
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import LayoutTensorIter, copy_local_to_dram
from layout.tensor_core_async import (
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)
from layout.tma_async import PipelineState, TMATensorTile, create_tma_tile
from linalg.matmul_sm90 import warp_specialize_gemm_with_multicasting
from linalg.matmul_tile_scheduler import MatmulSchedule
from linalg.utils_gpu import MatmulConfig
from memory import stack_allocation
from internal_utils._measure import relative_difference
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4


fn test_warp_specialize_gemm_with_multicasting[
    wgmma_n: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    cluster_shape: IndexList[3],
    num_consumer: Int = 1,
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

    alias block_tile_shape = Index(
        128, wgmma_n, 128
    ) if a_type is DType.float8_e4m3fn else Index(128, wgmma_n, 64)
    alias wgmma_shape = Index(
        64, wgmma_n, 32
    ) if a_type is DType.float8_e4m3fn else Index(64, wgmma_n, 16)

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    print(
        "wgmma_n",
        wgmma_n,
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
        "PARTITIONED" if partitioned_multicast else StaticString("BROADCAST"),
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
                80,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                num_pipeline_stages=6,
                partitioned_multicast=False,
                grid_shape = Index(32, 4),
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[2560](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                num_pipeline_stages=6,
                grid_shape = Index(10, 13),
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublaslt_handle,
                static[8192](),
                static[2560](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                num_pipeline_stages=6,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublaslt_handle,
                static[4096](),
                static[2560](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                num_pipeline_stages=6,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[8192](),
                static[2048](),
            )

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                num_pipeline_stages=6,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[14336](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                num_pipeline_stages=6,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublaslt_handle,
                static[4096](),
                static[8192](),
                static[7168](),
            )

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                num_pipeline_stages=6,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublaslt_handle,
                static[512](),
                static[8192](),
                static[7168](),
            )

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 2, 1),
                partitioned_multicast=True,
            ](ctx, cublaslt_handle, dynamic(199), static[512](), static[1024]())

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 2, 1),
                partitioned_multicast=False,
            ](ctx, cublaslt_handle, dynamic(200), static[256](), static[256]())

            test_warp_specialize_gemm_with_multicasting[
                128,
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                DType.bfloat16,
                Index(1, 2, 1),
                num_consumer=2,
                partitioned_multicast=True,
            ](ctx, cublaslt_handle, dynamic(201), static[384](), static[256]())

        print("BFLOAT16 GEMM TESTS")
        with vendor_blas.Handle[vendor_blas.Backend.CUBLAS]() as cublas_handle:
            test_warp_specialize_gemm_with_multicasting[
                80,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                num_pipeline_stages=8,
                partitioned_multicast=False,
                grid_shape = Index(32, 4),
                schedule = MatmulSchedule.TILE2D,
            ](ctx, cublas_handle, dynamic(512), static[2560](), static[8192]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                grid_shape = Index(10, 13),
                schedule = MatmulSchedule.TILE2D,
            ](ctx, cublas_handle, dynamic(8192), static[2560](), static[8192]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](ctx, cublas_handle, dynamic(4096), static[2560](), static[8192]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                grid_shape = Index(4, 33),
                schedule = MatmulSchedule.TILE2D,
            ](ctx, cublas_handle, dynamic(8192), static[8192](), static[2048]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](ctx, cublas_handle, dynamic(4096), static[8192](), static[2048]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                use_tma_store=True,
                schedule = MatmulSchedule.TILE2D,
            ](ctx, cublas_handle, dynamic(4096), static[8192](), static[2048]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                grid_shape = Index(8, 16),
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                dynamic(8192),
                static[14336](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                grid_shape = Index(8, 16),
                use_tma_store=True,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                dynamic(8192),
                static[14336](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                dynamic(4096),
                static[14336](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                use_tma_store=True,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                dynamic(4096),
                static[14336](),
                static[8192](),
            )

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                grid_shape = Index(4, 33),
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                static[8192](),
                static[8192](),
                static[7168](),
            )

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                grid_shape = Index(4, 33),
                use_tma_store=True,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                static[8192](),
                static[8192](),
                static[7168](),
            )

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                static[4096](),
                static[8192](),
                static[7168](),
            )

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                use_tma_store=True,
                schedule = MatmulSchedule.TILE2D,
            ](
                ctx,
                cublas_handle,
                static[4096](),
                static[8192](),
                static[7168](),
            )

            @parameter
            for multicast_mode in range(2):
                test_warp_specialize_gemm_with_multicasting[
                    80,
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](
                    ctx,
                    cublas_handle,
                    static[256](),
                    static[80](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    256,
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](
                    ctx,
                    cublas_handle,
                    static[256](),
                    static[256](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    64,
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](
                    ctx,
                    cublas_handle,
                    static[256](),
                    static[64](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    256,
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 1, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](
                    ctx,
                    cublas_handle,
                    static[128](),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    64,
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 1, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](
                    ctx,
                    cublas_handle,
                    static[128](),
                    static[128](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    256,
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 2, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](
                    ctx,
                    cublas_handle,
                    static[256](),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    64,
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](
                    ctx,
                    cublas_handle,
                    static[256](),
                    static[128](),
                    static[128](),
                )

            alias wgmma_n = List[Int](64, 128, 256)

            print("# 2x1 warp specialized gemm with multicasting tests")

            @parameter
            for i in range(len(wgmma_n)):
                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    static[1024](),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(1024),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(199),
                    static[1024](),
                    static[1024](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(200),
                    static[512](),
                    static[256](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 2, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(201),
                    static[2048](),
                    static[256](),
                )

            print("# 1x2 warp specialized gemm with multicasting tests")

            @parameter
            for i in range(len(wgmma_n)):
                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 1, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    static[1024](),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 1, 1),
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(1024),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 1, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(99),
                    static[1024](),
                    static[1024](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 1, 1),
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(100),
                    static[512](),
                    static[256](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 1, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(201),
                    static[2048](),
                    static[256](),
                )

            print("# 2x2 warp specialized gemm with multicasting tests")

            @parameter
            for i in range(len(wgmma_n)):
                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 2, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    static[1024](),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 2, 1),
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(1024),
                    static[512](),
                    static[128](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 2, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(199),
                    static[1024](),
                    static[1024](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 2, 1),
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(200),
                    static[512](),
                    static[256](),
                )

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(2, 2, 1),
                    num_consumer=2,
                    partitioned_multicast=True,
                ](
                    ctx,
                    cublas_handle,
                    dynamic(201),
                    static[2048](),
                    static[256](),
                )
