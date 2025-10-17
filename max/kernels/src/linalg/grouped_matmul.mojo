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
from sys import align_of, simd_width_of, size_of
from sys.info import has_amd_gpu_accelerator

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, barrier
from gpu.cluster import cluster_sync, cluster_sync_relaxed, elect_one_sync
from gpu.globals import WARPGROUP_SIZE
from gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200, H100
from gpu.id import (
    block_dim,
    block_id_in_cluster,
    block_idx,
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id as get_warp_id,
)
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
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

from utils.fast_div import FastDiv
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .arch.sm100 import MmaOpSM100_SS
from .matmul.gpu.sm90.dispatch import _find_largest_bn_for_sm90_matmul
from .matmul.gpu.sm90.matmul import _get_c_smem_layout
from .matmul.gpu.sm90.grouped_matmul import grouped_matmul_sm90
from .matmul.vendor.blas import matmul as vendor_matmul
from .utils import elementwise_epilogue_type
from .utils_gpu import MatmulConfig, block_swizzle

from .matmul.gpu.amd import gemm_kernel_amd
from algorithm import vectorize


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
    transpose_b: Bool = True,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    constrained[transpose_b, "Only support transposed B in grouped matmul."]()

    alias kernel = naive_grouped_matmul_kernel[
        c_type,
        c_shape,
        a_type,
        a_shape,
        b_type,
        b_shape,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
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


# grouped matmul computes:
# for i in range(num_active_experts)
#     C[a_offsets[i]:a_offsets[i+1], :] = A[a_offsets[i]:a_offsets[i+1], :] @ B[expert_ids[i], :, :].T


fn naive_grouped_matmul_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
):
    # There has to be a better way :(
    var M: UInt = UInt(
        a_offsets[Int(block_idx.z) + 1] - a_offsets[Int(block_idx.z)]
    )
    N = b.dim[1]()
    K = b.dim[2]()

    a_start_row = a_offsets[Int(block_idx.z)]
    a_by_expert = a.data + a_start_row * K

    expert = expert_ids[Int(block_idx.z)]
    b_by_expert = b.data + expert * N * K

    # indices in current matmul
    n = global_idx.x
    m = global_idx.y

    if n >= UInt(N) or m >= UInt(M):
        return

    alias accum_type = get_accum_type[a_type]()

    var accum = Scalar[accum_type](0.0)

    # avoid doing matmul if expert is -1. We use this value to indicate that
    # the block is not active for LoRA use cases.
    # NOTE: we still call elementwise lambda even if expert is -1
    if expert != -1:
        for k in range(K):
            accum += (
                a_by_expert[m * UInt(K) + UInt(k)].cast[accum_type]()
                * b_by_expert[n * UInt(K) + UInt(k)].cast[accum_type]()
            )

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](
            Index(a_start_row + m, n), accum.cast[c_type]()
        )
    else:
        c_by_expert = c.data + a_start_row * N
        c_by_expert[m * UInt(N) + n] = accum.cast[c_type]()


# ===----------------------------------------------------------------------=== #
# H100 grouped matmul
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn grouped_matmul_kernel_sm100[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = True,
    num_threads: Int = 128,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: Int,
):
    constrained[transpose_b, "Only support transposed B in layout"]()
    constrained[num_threads == 128 or num_threads == 256]()

    M = a_offsets[Int(block_idx.z + 1)] - a_offsets[Int(block_idx.z)]
    alias N = c.layout.shape[1].value()
    alias K = b_layout.shape[1].value()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]  # BM
    alias MMA_N = mma_shape[1]  # BN
    alias MMA_K = mma_shape[2]  # 16
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // MMA_K

    a_start_row = a_offsets[Int(block_idx.z)]
    expert = expert_ids[Int(block_idx.z)]
    b_start_row = expert * N

    m_start = block_idx.y * UInt(BM)
    n_start = block_idx.x * UInt(BN)
    a_m_start = UInt(a_start_row) + m_start
    b_n_start = UInt(b_start_row) + n_start
    if m_start >= UInt(M) or n_start >= UInt(N):
        return

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    alias sub_a_smem_layout = tile_layout_k_major[
        a_type, BM, 64, swizzle_mode=a_swizzle
    ]()
    alias sub_b_smem_layout = tile_layout_k_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]()

    a_smem = rebind[
        UnsafePointer[Scalar[a_type], address_space = AddressSpace.SHARED]
    ](
        external_memory[
            Scalar[a_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )

    # a_smem_layout is a description of how tile is arranged in memory, and LayoutTensor is a pointer to memory + a layout, taking in a_smem as its pointer
    alias a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias sub_a_smem_tile_t = LayoutTensor[
        a_type,
        sub_a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias sub_b_smem_tile_t = LayoutTensor[
        b_type,
        sub_b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias a_size = a_smem_layout.size()
    alias b_size = b_smem_layout.size()

    constrained[
        ((a_size * size_of[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[
        ((b_size * size_of[b_type]()) % 16) == 0, "preserve alignment"
    ]()
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    alias accum_type = get_accum_type[a_type]()

    alias c_frag_size = MMA_M * MMA_N // num_threads  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag = SIMD[
        accum_type, c_frag_size
    ]()  # array of accumulator elements

    alias a_expected_bytes = a_size * size_of[a_type]()
    alias b_expected_bytes = b_size * size_of[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
    var elect_one_thread = thread_idx.x == 0
    alias max_tmem_cols = 512

    # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = ptr_tmem_addr[0]

    # Create MmaOpSM100_SS instance to handle MMA operations
    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=1,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    for i in range(
        num_iters
    ):  # K // BK, which is K // 64 or K // 128 depending on BK
        # so only one thread per CTA does the copy
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)

            @parameter
            for j in range(
                BK // 64
            ):  # so we do the copy in 64 chunks or 64 elements at a time (BK // 64). but hmm, we said that the K atom can only be 32 bytes (16 elements)
                alias k = 64 * j
                alias a_offset = a_smem_layout(IntTuple(0, k))
                alias b_offset = b_smem_layout(IntTuple(0, k))
                constrained[((a_offset * size_of[a_type]()) % 128) == 0]()
                constrained[((b_offset * size_of[b_type]()) % 128) == 0]()
                sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                # the answer to the above comment. # The descriptor layout i.e. data per copy can be smaller than the shared memory
                # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
                # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
                # so the async_copy takes care of that. TMA engine will copy the data from global tensor into smem tile A
                k_start = UInt(i) * UInt(BK) + UInt(k)
                a_tma_op.async_copy(
                    sub_a_smem_tile,
                    tma_mbar[0],
                    (UInt(k_start), UInt(a_m_start)),
                )
                sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)
                b_tma_op.async_copy(
                    sub_b_smem_tile,
                    tma_mbar[0],
                    (UInt(k_start), UInt(b_n_start)),
                )
        # wait for the copy to finish
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        # now we do the mma, again only one thread issues the instruction
        if elect_one_thread:
            # Use MmaOpSM100_SS to perform the MMA operation
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(i == 0),  # Initialize C on first iteration
            )

            mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # eventually all of c has been accumulated, so we load it from tmem_addr into c_frag registers using tcgen05_ld
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()  # wait for the load to finish

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    alias num_warps = num_threads // WARP_SIZE
    warp_id = thread_idx.x // UInt(WARP_SIZE)

    alias c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
    alias c_gmem_type = LayoutTensor[
        c_type,
        c_gmem_layout,
        MutableAnyOrigin,
        layout_int_type = DType.int32,
        address_space = AddressSpace.GENERIC,
    ]

    # FIXME: A list literal initializer should be enough here, but somehow Mojo fails to infer that.
    var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
        Index(M, N), Index(N, 1)
    )

    var c_by_expert = c_gmem_type(
        c.ptr + a_start_row * N, c_gmem_runtime_layout
    )

    ctile, ctile_coords, _ = c_by_expert.tile_with_offset[BM, BN](
        block_idx.y, block_idx.x
    )
    alias c_coord_type = __type_of(ctile_coords)

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // num_warps, MMA_N](
                    4 * m_mma + warp_id, n_mma
                )
            )
            c_gmem_warp_tile_coords = ctile_coords + rebind[c_coord_type](
                _c_gmem_warp_tile_coords
            )

            c_gmem_frag, _c_gmem_frag_coords, _ = c_gmem_warp_tile.vectorize[
                1, 2
            ]().distribute_with_offset[Layout.row_major(8, 4)](lane_id())
            new_c_gmem_frag_coords = rebind[c_coord_type](_c_gmem_frag_coords)
            new_c_gmem_frag_coords[1] *= 2
            c_gmem_frag_coords = (
                c_gmem_warp_tile_coords + new_c_gmem_frag_coords
            )

            alias num_vecs_m = c_gmem_frag.layout.shape[0].value()
            alias num_vecs_n = c_gmem_frag.layout.shape[1].value()

            @parameter
            for n_vec in range(num_vecs_n):

                @parameter
                for m_vec in range(num_vecs_m):
                    alias i_vec = n_vec * num_vecs_m + m_vec
                    alias dst_idx = __type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    alias dst_m_offset = dst_idx // N
                    alias dst_n_offset = dst_idx % N
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < N:
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        @parameter
                        if elementwise_lambda_fn:
                            alias alignment = align_of[SIMD[c_type, 2]]()
                            alias epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (Int(a_start_row + m), Int(n)), c_mn
                            )
                        else:
                            c_gmem_frag[m_vec, n_vec] = rebind[
                                c_gmem_frag.element_type
                            ](c_mn)


fn grouped_matmul_sm100[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool = True,
    mma_shape: IndexList[3] = Index(64, 128, 16),
    block_tile_shape: IndexList[3] = Index(64, 128, 64),
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

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    constrained[K % BK == 0]()
    constrained[BK == 64]()

    # hard coded 64 for BK

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias c_swizzle = TensorMapSwizzle.SWIZZLE_NONE
    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    a_tensor = from_ndbuffer_row_major(a)
    a_tma_op = create_tma_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a_tensor
    )
    b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)
    b_tma_op = create_tma_tile[
        Index(BN, BK) if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b_tensor)
    c_tensor = from_ndbuffer_row_major(c)

    alias block_dim = 128
    alias smem_use = (BM * size_of[a_type]() + BN * size_of[b_type]()) * BK + 24

    alias kernel = grouped_matmul_kernel_sm100[
        a_type,
        b_type,
        c_type,
        __type_of(a_tensor).layout,
        __type_of(b_tensor).layout,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c_tensor).layout,
        block_tile_shape,
        mma_shape,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        a_swizzle,
        b_swizzle,
        c_swizzle,
        transpose_b=transpose_b,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        a_offsets,
        expert_ids,
        c,
        ceildiv(K, BK),
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )


fn grouped_matmul_amd_kernel_launcher[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c_tensor: LayoutTensor[c_type, layout_c, MutableAnyOrigin],
    a_tensor: LayoutTensor[a_type, layout_a, MutableAnyOrigin],
    b_tensor: LayoutTensor[b_type, layout_b, MutableAnyOrigin],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    num_active_experts: Int,
):
    var M = a_offsets[Int(block_idx.z + 1)] - a_offsets[Int(block_idx.z)]
    alias N = c_tensor.shape[1]()
    alias K = b_tensor.shape[1]()

    alias num_experts = b_tensor.shape[0]()
    var expert_id = expert_ids[Int(block_idx.z)]
    var a_start_row = a_offsets[Int(block_idx.z)]

    var a_ptr = a_tensor.ptr + a_start_row * K
    var b_ptr = b_tensor.ptr + expert_id * N * K
    var c_ptr = c_tensor.ptr + a_start_row * N

    alias c_layout = Layout.row_major(UNKNOWN_VALUE, N)
    alias a_layout = Layout.row_major(UNKNOWN_VALUE, K)
    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)

    var c = LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        address_space = c_ptr.address_space,
    ](c_ptr, RuntimeLayout[c_layout](Index(M, N), Index(N, 1)))

    var a = LayoutTensor[
        a_type,
        a_layout,
        MutableAnyOrigin,
        address_space = a_ptr.address_space,
    ](a_ptr, RuntimeLayout[a_layout](Index(M, K), Index(K, 1)))

    var b = LayoutTensor[
        b_type,
        b_layout,
        MutableAnyOrigin,
        address_space = b_ptr.address_space,
    ](b_ptr, RuntimeLayout[b_layout](Index(N, K), Index(K, 1)))

    @always_inline
    @parameter
    fn elementwise_epilogue_fn_wrapper[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        @parameter
        if elementwise_lambda_fn:
            alias elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_idx = IndexList[2](Int(a_start_row + idx[0]), idx[1])
            elementwise_epilogue(batch_idx, val)

    # Only perform matmul if expert_id is not -1
    # AMD matmul kernel performs the epilogue function
    if expert_id != -1:
        gemm_kernel_amd[
            c_type,
            c.layout,
            a_type,
            a.layout,
            b_type,
            b.layout,
            transpose_b,
            c.layout_int_type,
            a.layout_int_type,
            b.layout_int_type,
            c.linear_idx_type,
            a.linear_idx_type,
            b.linear_idx_type,
            config,
            OptionalReg[elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ](c, a, b)

    # Perform the epilogue function separately if expert_id is -1
    else:
        _ = c.fill(0.0)

        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()

            alias BM = config.block_tile_shape[0]
            alias BN = config.block_tile_shape[1]
            alias vec_width = simd_width_of[c_type]()
            alias alignment = align_of[SIMD[c_type, vec_width]]()

            var block_m = Int(block_idx.y)
            var block_n = Int(block_idx.x)

            # Early exit if this block is completely outside the matrix bounds
            if UInt32(block_m * BM) >= UInt32(M):
                return

            alias threads_per_block = 256
            alias elements_per_thread = ceildiv(BM * BN, threads_per_block)

            var tid = Int(thread_idx.x)
            var thread_start = tid * elements_per_thread
            var thread_end = min(thread_start + elements_per_thread, BM * BN)

            var elements_to_process = thread_end - thread_start

            @always_inline
            @parameter
            fn process_elements[width: Int](idx: Int):
                var elem_idx = thread_start + idx
                var tile_row, tile_col = divmod(elem_idx, BN)
                var local_row: UInt32 = block_m * BM + tile_row
                var local_col: UInt32 = block_n * BN + tile_col

                if local_row < M:
                    var remaining_in_row = N - local_col
                    var remaining_in_tile_row = BN - tile_col
                    var actual_width = min(
                        width,
                        min(Int(remaining_in_row), Int(remaining_in_tile_row)),
                    )

                    if actual_width == width and local_col + width <= N:
                        var zero_vec = SIMD[c_type, width](0.0)
                        epilogue[
                            dtype=c_type,
                            width=width,
                            alignment = align_of[SIMD[c_type, width]](),
                        ]((Int(local_row), Int(local_col)), zero_vec)
                    else:
                        for i in range(actual_width):
                            if local_col + i < N:
                                var zero_scalar = SIMD[c_type, 1](0.0)
                                epilogue[dtype=c_type, width=1, alignment=1](
                                    (Int(local_row), Int(local_col + i)),
                                    zero_scalar,
                                )

            vectorize[process_elements, vec_width](elements_to_process)


fn grouped_matmul_amd[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool = True,
    block_tile_shape: IndexList[3] = Index(128, 128, 64),
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

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    constrained[K % BK == 0]()

    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)
    var c_tensor = from_ndbuffer_row_major(c)

    alias block_dim = 256
    alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(BM, BN, BK),
        warp_tile_shape=Index(BM // 2, BN // 2, BK),
        num_pipeline_stages=1,
        num_k_partitions=1,
    )

    alias kernel = grouped_matmul_amd_kernel_launcher[
        c_type,
        a_type,
        b_type,
        __type_of(c_tensor).layout,
        __type_of(a_tensor).layout,
        __type_of(b_tensor).layout,
        transpose_b,
        config,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        c_tensor,
        a_tensor,
        b_tensor,
        a_offsets,
        expert_ids,
        num_active_experts,
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(block_dim),
    )


# ===----------------------------------------------------------------------=== #
# Entry Point and Dispatch
# ===----------------------------------------------------------------------=== #


fn grouped_matmul[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    alias is_expert_shape_static = b_shape.all_known[3]() and a_shape.has_value[
        1
    ]() and c_shape.has_value[1]()
    alias is_sm90_kernel_applicable = ctx.default_device_info is H100 and is_expert_shape_static
    alias is_sm100_kernel_applicable = ctx.default_device_info is B200 and is_expert_shape_static
    alias is_amd_kernel_applicable = has_amd_gpu_accelerator() and is_expert_shape_static

    @parameter
    if is_sm90_kernel_applicable:
        alias static_N = c.shape.get[1]()
        alias BN = _find_largest_bn_for_sm90_matmul[a_type, static_N]()
        alias wgmma_shape = IndexList[3](64, BN, 16)

        grouped_matmul_sm90[
            wgmma_shape=wgmma_shape, elementwise_lambda_fn=elementwise_lambda_fn
        ](
            c,
            a,
            a_offsets,
            max_num_tokens_per_expert,
            b,
            expert_ids,
            num_active_experts,
            ctx,
        )
    elif is_sm100_kernel_applicable:
        grouped_matmul_sm100[elementwise_lambda_fn=elementwise_lambda_fn](
            c,
            a,
            a_offsets,
            max_num_tokens_per_expert,
            b,
            expert_ids,
            num_active_experts,
            ctx,
        )
    elif is_amd_kernel_applicable:
        grouped_matmul_amd[elementwise_lambda_fn=elementwise_lambda_fn](
            c,
            a,
            a_offsets,
            max_num_tokens_per_expert,
            b,
            expert_ids,
            num_active_experts,
            ctx,
        )
    else:
        naive_grouped_matmul[elementwise_lambda_fn=elementwise_lambda_fn](
            c,
            a,
            b,
            a_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )


# ===----------------------------------------------------------------------===#
# Vendor Grouped GEMM for LoRA
# ===----------------------------------------------------------------------===#


fn grouped_matmul_vendor[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool = True,
    use_tf32: Bool = False,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    constrained[transpose_b, "Only support transposed B in grouped matmul."]()
    constrained[
        a_type == b_type, "A and B must have the same dtype for vendor BLAS"
    ]()
    # Push the device context to ensure correct CUDA context
    for i in range(num_active_experts):
        var expert_id = expert_ids[i]

        var token_start = a_offsets[i]
        var token_end = a_offsets[i + 1]
        var num_tokens = token_end - token_start

        # Skip if no tokens for this expert
        if num_tokens <= 0:
            continue

        # Handle experts with expert_id = -1 by writing zeros
        if expert_id < 0:
            # Create output slice and zero it out
            var c_slice = NDBuffer[c_type, 2, MutableAnyOrigin](
                c.data + token_start * c.dim[1](),
                DimList(num_tokens, c.dim[1]()),
            )
            var buff = DeviceBuffer(
                ctx, c_slice.data, c_slice.num_elements(), owning=False
            )
            ctx.enqueue_memset(buff, 0)
            continue

        # Create views into the tensors for this expert
        var a_slice = NDBuffer[a_type, 2, MutableAnyOrigin](
            a.data + token_start * a.dim[1](),
            DimList(num_tokens, a.dim[1]()),
        )
        var b_slice = NDBuffer[b_type, 2, MutableAnyOrigin](
            b.data + expert_id * b.dim[1]() * b.dim[2](),
            DimList(b.dim[1](), b.dim[2]()),
        )
        var c_slice = NDBuffer[c_type, 2, MutableAnyOrigin](
            c.data + token_start * c.dim[1](),
            DimList(num_tokens, c.dim[1]()),
        )

        vendor_matmul[use_tf32](
            ctx,
            c_slice,
            a_slice,
            b_slice,
            c_row_major=True,
            transpose_b=transpose_b,
        )
