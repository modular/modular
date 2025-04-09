# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import ceildiv
from sys import sizeof

import linalg.vendor_blas
from buffer.dimlist import Dim, DimList, _make_tuple
from collections import OptionalReg
from gpu import WARP_SIZE, barrier, MAX_THREADS_PER_BLOCK_METADATA
from gpu.grid_controls import (
    pdl_launch_attributes,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.host.info import H100
from gpu.id import (
    block_dim,
    block_idx,
    thread_idx,
    global_idx,
    grid_dim,
    lane_id,
    block_id_in_cluster,
)
from gpu.memory import AddressSpace
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_ldmatrix_swizzle, make_swizzle
from layout.layout_tensor import (
    copy_local_to_dram,
    LayoutTensorIter,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple, UNKNOWN_VALUE
from layout._ndbuffer_stub import from_ndbuffer_row_major
from utils.numerics import get_accum_type
from layout.tensor_core_async import (
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
    st_matrix_n_layout,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from buffer.buffer import NDBuffer
from sys._assembly import inlined_assembly
from sys import alignof, simdwidthof
from gpu.cluster import (
    elect_one_sync,
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
)
from gpu.intrinsics import warpgroup_reg_dealloc, warpgroup_reg_alloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from gpu.sync import cp_async_bulk_wait_group, named_barrier
from pathlib import Path

from .utils import elementwise_epilogue_type
from linalg.matmul_tile_scheduler import TileScheduler, MatmulSchedule
from linalg.matmul_sm90 import (
    warp_specialized_gemm_output,
    _get_c_smem_layout,
    cluster_size,
)
from .utils_gpu import block_swizzle, MatmulConfig
from gpu.warp import broadcast
from gpu.mma import st_matrix
from memory import bitcast
from stdlib.bit import log2_floor

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4

# ===----------------------------------------------------------------------=== #
# Naive grouped matmul
# ===----------------------------------------------------------------------=== #


fn naive_grouped_matmul[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    constrained[transpose_b, "Only support transposed B in grouped matmul."]()

    ctx.enqueue_function[
        naive_grouped_matmul_kernel[
            c_type,
            c_shape,
            a_type,
            a_shape,
            b_type,
            b_shape,
        ]
    ](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        grid_dim=(
            ceildiv(c.dim[1](), 32),
            ceildiv(max_num_tokens_per_expert, 16),
            num_active_experts,
        ),
        block_dim=(32, 16, 1),
    )


fn naive_grouped_matmul_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
):
    # There has to be a better way :(
    var M: UInt = UInt(Int(a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]))
    N = b.dim[1]()
    K = b.dim[2]()

    a_start_row = a_offsets[block_idx.z]
    a_by_expert = a.data + a_start_row * K

    expert = expert_ids[block_idx.z]
    b_by_expert = b.data + expert * N * K

    # indices in current matmul
    n = global_idx.x
    m = global_idx.y

    if n >= N or m >= M:
        return

    alias accum_type = get_accum_type[a_type]()

    var accum = Scalar[accum_type](0.0)

    for k in range(K):
        accum += (
            a_by_expert[m * K + k].cast[accum_type]()
            * b_by_expert[n * K + k].cast[accum_type]()
        )

    c_by_expert = c.data + a_start_row * N
    c_by_expert[m * N + n] = accum.cast[c_type]()
