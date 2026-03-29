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
from std.collections import Optional
from std.math import ceildiv
from std.sys import size_of

from std.gpu.globals import WARPGROUP_SIZE
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Layout,
    LayoutTensor,
    TileTensor,
)
from layout.tma_async import create_tensor_tile

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from .matmul_kernels import HopperMatmulSM90Kernel
from .matmul import _get_c_smem_layout

from ....utils import elementwise_epilogue_type
from ....utils_gpu import MatmulConfig


@always_inline
def default_config_sm90[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    wgmma_shape: IndexList[3],
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    comptime BN = wgmma_shape[1]
    comptime BK = 128 // size_of[a_type]()
    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(128, BN, BK),
        mma_shape=wgmma_shape,
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
    )


def grouped_matmul_sm90[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool = True,
    wgmma_shape: IndexList[3] = Index(64, 256, 16),
    config: MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ] = default_config_sm90[a_type, b_type, c_type, transpose_b, wgmma_shape](),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=AddressSpace.GENERIC, ...],
    a: TileTensor[a_type, address_space=AddressSpace.GENERIC, ...],
    a_offsets: TileTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    b: TileTensor[b_type, address_space=AddressSpace.GENERIC, ...],
    expert_ids: TileTensor[
        mut=False, DType.int32, address_space=AddressSpace.GENERIC, ...
    ],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    # Early-exit for empty inputs to avoid creating invalid TMA descriptors.
    if num_active_experts == 0 or Int(a.dim[0]()) == 0 or Int(c.dim[0]()) == 0:
        return
    comptime num_experts = b.static_shape[0]
    comptime N = b.static_shape[1]
    comptime K = b.static_shape[2]

    comptime cluster_shape = StaticTuple[Int32, 3](
        Int32(config.cluster_shape[0]),
        Int32(config.cluster_shape[1]),
        Int32(config.cluster_shape[2]),
    )

    comptime k_group_size = config.k_group_size

    comptime c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        Int(config.num_pipeline_stages),
        Int(k_group_size),
    ]()
    comptime c_smem_tile = Index(
        c_smem_layout.shape[0].value(), c_smem_layout.shape[1].value()
    )

    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]

    # Create TMA op for the entire A tensor including all tokens.
    a_tensor = a.to_layout_tensor()
    a_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a_tensor
    )

    # Flattne B tensor into a 2D tensor for easier TMA support.
    b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        address_space=AddressSpace.GENERIC,
    ](b.ptr)
    b_tma_op = create_tensor_tile[Index(BN, BK), swizzle_mode=b_swizzle](
        ctx, b_tensor
    )

    # Create a dummy TMA op for C, we don't support TMA store for output.
    c_tensor = c.to_layout_tensor()
    c_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=c_swizzle](
        ctx, c_tensor
    )

    comptime num_threads = WARPGROUP_SIZE * Int(
        config.num_consumer
    ) + WARPGROUP_SIZE
    comptime smem_size = Int(config.num_pipeline_stages) * (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + (size_of[Int64]() * 2)
    ) + c_smem_layout.size() * size_of[c_type]()

    comptime kernel = HopperMatmulSM90Kernel[
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
        num_threads,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast=config.partitioned_multicast,
        use_tma_store=False,
        promotion_frequency=1,
        pdl_level=config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
    ].run_grouped[
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(c_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        type_of(c_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).desc_shape,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
    ]

    ctx.enqueue_function[kernel, kernel](
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
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
