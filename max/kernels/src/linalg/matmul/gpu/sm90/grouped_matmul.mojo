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
from sys import simd_width_of, size_of

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import cluster_sync, cluster_sync_relaxed, elect_one_sync
from gpu.globals import WARPGROUP_SIZE
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import (
    block_id_in_cluster,
    block_idx,
    grid_dim,
    thread_idx,
)
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout
from layout.tensor_core_async import TensorCoreAsync, tile_layout_k_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .matmul_kernels import HopperMatmulSM90Kernel
from .matmul import _get_c_smem_layout

from ....utils import elementwise_epilogue_type
from ....utils_gpu import MatmulConfig, block_swizzle


@always_inline
fn default_config_sm90[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    wgmma_shape: IndexList[3],
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    alias BN = wgmma_shape[1]
    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(128, BN, 64),
        mma_shape=wgmma_shape,
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
    )


fn grouped_matmul_sm90[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool = True,
    wgmma_shape: IndexList[3] = Index(64, 256, 16),
    config: MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ] = default_config_sm90[a_type, b_type, c_type, transpose_b, wgmma_shape](),
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    alias num_experts = b.shape.get[0]()
    alias N = b.shape.get[1]()
    alias K = b.shape.get[2]()

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )

    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])

    alias c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        Int(config.num_pipeline_stages),
    ]()
    alias c_smem_tile = Index(
        c_smem_layout.shape[0].value(), c_smem_layout.shape[1].value()
    )

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    # Create TMA op for the entire A tensor including all tokens.
    a_tensor = from_ndbuffer_row_major(a)
    a_tma_op = create_tma_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a_tensor
    )

    # Flattne B tensor into a 2D tensor for easier TMA support.
    b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)
    b_tma_op = create_tma_tile[Index(BN, BK), swizzle_mode=b_swizzle](
        ctx, b_tensor
    )

    # Create a dummy TMA op for C, we don't support TMA store for output.
    c_tensor = from_ndbuffer_row_major(c)
    c_tma_op = create_tma_tile[Index(BM, BK), swizzle_mode=c_swizzle](
        ctx, c_tensor
    )

    alias num_threads = WARPGROUP_SIZE * config.num_consumer + WARPGROUP_SIZE
    alias smem_size = Int(config.num_pipeline_stages) * (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + (size_of[Int64]() * 2)
    ) + c_smem_layout.size() * size_of[c_type]()

    alias kernel = HopperMatmulSM90Kernel[
        a_type,
        b_type,
        c_type,
        a_tensor.layout,
        b_tensor.layout,
        c_tensor.layout,
        c_smem_layout,
        config.block_tile_shape,
        wgmma_shape,
        cluster_shape,
        Int(config.num_pipeline_stages),
        Int(num_threads),
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast = config.partitioned_multicast,
        use_tma_store=False,
        promotion_frequency=1,
        pdl_level = config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
    ].run_grouped[
        a_tma_op.layout,
        b_tma_op.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_offsets,
        expert_ids,
        c_tensor,
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )
