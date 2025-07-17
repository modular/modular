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
from sys import alignof, simdwidthof, sizeof

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, _make_tuple
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
    elect_one_sync,
)
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.compile import _compile_code_asm
from gpu.host import get_gpu_target
from gpu.host._nvidia_cuda import TensorMapSwizzle, TMADescriptor
from gpu.host.info import H100
from gpu.id import (
    block_dim,
    block_id_in_cluster,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_mbarrier_init,
    tma_store_fence,
)
from gpu.mma import (
    WGMMADescriptor,
    st_matrix,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from gpu.sync import named_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout import coalesce
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.swizzle import make_ldmatrix_swizzle
from layout.tensor_core_async import (
    TensorCoreAsync,
    st_matrix_n_layout,
    tile_layout_k_major,
    wgmma_c_layout,
    tile_to_descriptor,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
    create_tma_tile_template,
)
from linalg.matmul_tile_scheduler import MatmulSchedule, TileScheduler
from memory import bitcast, stack_allocation
from memory.pointer import _GPUAddressSpace
from stdlib.bit import log2_floor

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from .utils_gpu import (
    MatmulConfig,
    block_swizzle,
)
from sys import sizeof
from math import ceildiv
from hashlib import default_comp_time_hasher
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu.mma_sm100 import *
from gpu.tcgen05 import *


# TODO: Add methods to conveniently extract specific modes from a layout.
fn extract_first_2_modes[l: Layout]() -> Layout:
    constrained[l.rank() >= 2]()

    return Layout(
        IntTuple(l.shape[0].value(), l.shape[1].value()),
        IntTuple(l.stride[0].value(), l.stride[1].value()),
    )


@fieldwise_init
@register_passable("trivial")
struct Major:
    var val: Int

    alias K = Major(0)
    alias MN = Major(1)

    fn __eq__(self, rhs: Major) -> Bool:
        return self.val == rhs.val


# TODO: add create method to mma_operand trait and unify this with
# SM90 counter part by abstracting the return type.
fn _create_mma_desc[
    type: DType, //, canonical_layout: Layout, swizzle_mode: TensorMapSwizzle
](
    ptr: UnsafePointer[
        Scalar[type], address_space = AddressSpace.SHARED, *_, **_
    ]
) -> MMASmemDescriptor:
    # Extract the stride values from the canonical layout
    # The canonical layout is expected to have at least 2 dimensions
    alias stride01 = canonical_layout[0].stride[1].value()
    alias stride11 = canonical_layout[1].stride[1].value()
    alias SBO = stride01 * sizeof[type]()
    alias LBO = stride11 * sizeof[type]()

    # Create and return the MMA shared memory descriptor
    # This will be used by the SM100 MMA operations to access shared memory
    return MMASmemDescriptor.create[SBO, LBO, swizzle_mode](ptr)


@register_passable("trivial")
struct MmaOpSM100_SS[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    /,
    *,
    accum_type: DType = DType.float32,
    cta_group: Int = 1,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = False,
](Defaultable):
    var idesc: UMMAInsDescriptor[UMMAKind.KIND_F16]

    @always_inline
    fn __init__(out self):
        constrained[transpose_b, "MmaOpSM100 only supports transposed B"]()

        self.idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
            accum_type,
            a_type,
            b_type,
            Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
            transpose_b=transpose_b,
        ]()

    @always_inline
    fn mma(
        self,
        a: LayoutTensor[address_space = AddressSpace.SHARED, *_, **_],
        b: LayoutTensor[address_space = AddressSpace.SHARED, *_, **_],
        c_tmem: UInt32,
        init_c: Bool,
    ):
        """MMA input tiles.

        The layout assumes that coalesce(A) has shape (bm, sw_k, num_sw_k), we currently
        assumes bm = mma_m. In future, we can tile it to (mma_m, sw_k, num_sw_k, num_mma_m)
        The same logic applies to matrix B.
        """

        # Coalesce a and b
        # A and B are coalesced to rank-2 if it's only one tile or rank-3 if it has
        # multiple canonical layouts in K dim.
        alias a_coalesced_layout = coalesce(a.layout)
        alias b_coalesced_layout = coalesce(b.layout)

        # Canonical layouts are tiled by core matrices.
        alias a_canonical_layout = tile_to_descriptor[
            a.dtype, extract_first_2_modes[a_coalesced_layout]()
        ]()
        alias b_canonical_layout = tile_to_descriptor[
            b.dtype, extract_first_2_modes[b_coalesced_layout]()
        ]()

        var a_desc = _create_mma_desc[a_canonical_layout, a_swizzle](a.ptr)
        var b_desc = _create_mma_desc[b_canonical_layout, b_swizzle](b.ptr)

        @parameter
        for k in range(0, block_tile_shape[2], mma_shape[2]):
            alias a_offset = a.layout(IntTuple(0, k)) * sizeof[a_type]()
            alias b_offset = b.layout(IntTuple(0, k)) * sizeof[b_type]()

            var c_scale: UInt32 = 0 if (init_c and k == 0) else 1

            mma(
                a_desc + a_offset,
                b_desc + b_offset,
                c_tmem,
                self.idesc,
                c_scale=c_scale,
            )

    @always_inline
    fn commit(
        self,
        ptr_mbar: UnsafePointer[address_space = AddressSpace.SHARED, *_, **_],
    ):
        @parameter
        if cta_group == 1:
            mma_arrive(ptr_mbar)

    @always_inline
    fn wait(self):
        pass
