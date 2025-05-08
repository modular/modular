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
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from linalg.matmul_tile_scheduler import MatmulSchedule
from linalg.utils_gpu import MatmulConfig
from memory import stack_allocation

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4


def test_warp_specialize_gemm_with_multicasting[
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
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
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

    alias block_tile_shape = Index(128, wgmma_n, 64)
    alias wgmma_shape = Index(64, wgmma_n, 16)

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    # fmt: off
    print(
        "wgmma_n", wgmma_n,
        " : PROBLEM SHAPE (M,N,K): (", M, "x", N, "x", K, ") - ",
        "BLOCKS SHAPE (BM,BN,BK): (", BM, "x", BN, "x", BK, ") - ",
        "CLUSTER DIMS (M,N): (", CLUSTER_M, "x", CLUSTER_N,
        ") NUM CONSUMERS: ", num_consumer,
        " NUM PIPELINE STAGES: ", num_pipeline_stages,
        " MULTICAST MODE: ",
        "PARTITIONED" if partitioned_multicast else "BROADCAST",
    )
    # fmt: on

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

    var c_tensor = c_device.tensor

    @parameter
    @always_inline
    @__copy_capture(c_tensor)
    fn epilogue_fn[
        _type: DType,
        width: Int,
        *,
        alignment: Int = alignof[SIMD[_type, width]](),
    ](idx: IndexList[2], val: SIMD[_type, width]) capturing -> None:
        c_tensor.store[alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    alias matmul_config = MatmulConfig[
        a_type, b_type, c_type, transpose_b, mma_shape = Index(64, wgmma_n, 16)
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
        # Pass the compute lambda if defined, otherwise use the epilogue lambda.
        elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
            epilogue_fn
        ) if elementwise_compute_lambda_fn
        is None else None,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
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

    @parameter
    if elementwise_compute_lambda_fn:
        # Apply the compute lambda directly on the reference tensor
        alias compute_lambda = elementwise_compute_lambda_fn.value()
        for i in range(M):
            for j in range(N):
                c_host_ref.tensor[Index(i, j)] = compute_lambda(
                    IndexList[2](i, j),
                    c_host_ref.tensor[Index(i, j)],
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


def main():
    with DeviceContext() as ctx:
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
        ](ctx, dynamic(512), static[2560](), static[8192]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            num_consumer=2,
        ](ctx, dynamic(277), static[2560](), static[128]())

        test_warp_specialize_gemm_with_multicasting[
            64,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 2, 1),
        ](ctx, dynamic(393), static[8192](), static[2048]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
        ](ctx, dynamic(532), static[8192](), static[7168]())

        test_warp_specialize_gemm_with_multicasting[
            64,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
        ](ctx, dynamic(604), static[14336](), static[8192]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 2, 1),
        ](ctx, dynamic(2021), static[512](), static[128]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[8192](), static[2560](), static[8192]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[4096](), static[2560](), static[8192]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[8192](), static[8192](), static[2048]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[4096](), static[8192](), static[2048]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[8192](), static[14336](), static[8192]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[4096](), static[14336](), static[8192]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[8192](), static[8192](), static[7168]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
        ](ctx, static[4096](), static[8192](), static[7168]())

        @parameter
        @always_inline
        fn test_lambda_fn_square[
            _type: DType,
            width: Int,
            *,
            alignment: Int = alignof[SIMD[_type, width]](),
        ](idx: IndexList[2], val: SIMD[_type, width]) capturing -> SIMD[
            _type, width
        ]:
            return val * val

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            num_consumer=2,
            elementwise_compute_lambda_fn = OptionalReg[
                elementwise_compute_lambda_type
            ](test_lambda_fn_square),
        ](ctx, dynamic(277), static[2560](), static[128]())

        @parameter
        @always_inline
        fn test_lambda_add_coords[
            _type: DType,
            width: Int,
            *,
            alignment: Int = alignof[SIMD[_type, width]](),
        ](idx: IndexList[2], val: SIMD[_type, width]) capturing -> SIMD[
            _type, width
        ]:
            # Cast indices between 0-1 to avoid accuracy issues
            var i = Float32(idx[0]) / 277.0
            var j = Float32(idx[1] - idx[1] % 8) / 2560.0
            return val + i.cast[_type]() + 2 * j.cast[_type]()

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            num_consumer=2,
            elementwise_compute_lambda_fn = OptionalReg[
                elementwise_compute_lambda_type
            ](test_lambda_add_coords),
        ](ctx, dynamic(277), static[2560](), static[128]())

        @parameter
        for wgmma_n in range(8, 264, 8):
            test_warp_specialize_gemm_with_multicasting[
                wgmma_n,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](ctx, static[1024](), static[wgmma_n * 3](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 2, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](ctx, static[1024](), static[wgmma_n * 2](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](ctx, static[1024](), static[wgmma_n * 4](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 2, 1),
                num_consumer=2,
                partitioned_multicast=False,
                schedule = MatmulSchedule.TILE2D,
            ](ctx, dynamic(1024), static[wgmma_n * 6](), static[128]())
