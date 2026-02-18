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
from collections import Optional
from math import ceildiv
from sys import align_of, simd_width_of, size_of
from sys.info import has_amd_gpu_accelerator, has_amd_rdna_gpu_accelerator

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, barrier
from gpu.primitives.cluster import (
    cluster_sync,
    cluster_sync_relaxed,
    elect_one_sync,
)
from gpu.globals import WARPGROUP_SIZE
from gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200, H100
from gpu import (
    block_dim,
    block_id_in_cluster,
    block_idx,
    global_idx,
    grid_dim,
    warp_id,
    lane_id,
    thread_idx,
)
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import external_memory, fence_mbarrier_init
from gpu.primitives.grid_controls import PDLLevel

from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.compute.arch.tcgen05 import *
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout
from layout.tensor_core_async import TensorCoreAsync, tile_layout_k_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
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
from .grouped_matmul_sm100 import grouped_matmul_sm100_persistent

from .matmul.gpu import (
    _amdgpu_matmul_build_block_shape_list,
    _amdgpu_matmul_config_from_block_shape,
)
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
    b_shape: DimList,
    //,
    *,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B in grouped matmul."

    comptime kernel = naive_grouped_matmul_kernel[
        c_type,
        c_shape,
        a_type,
        a_shape,
        b_type,
        b_shape,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]
    ctx.enqueue_function[kernel, kernel](
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
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
):
    # There has to be a better way :(
    var M: UInt = UInt(
        a_offsets[Int(block_idx.z) + 1] - a_offsets[Int(block_idx.z)]
    )
    N = b.dim[1]()
    K = b.dim[2]()

    a_start_row = a_offsets[Int(block_idx.z)]
    a_by_expert = a.data + a_start_row * UInt32(K)

    expert = expert_ids[Int(block_idx.z)]
    b_by_expert = b.data + expert * Int32(N) * Int32(K)

    # indices in current matmul
    n = global_idx.x
    m = global_idx.y

    if n >= UInt(N) or m >= M:
        return

    comptime accum_type = get_accum_type[a_type]()

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
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](
            Index(a_start_row + UInt32(m), n), accum.cast[c_type]()
        )
    else:
        c_by_expert = c.data + a_start_row * UInt32(N)
        c_by_expert[m * UInt(N) + n] = accum.cast[c_type]()


fn naive_epilogue[
    c_type: DType,
    c_shape: DimList,
    *,
    elementwise_lambda_fn: elementwise_epilogue_type,
](c: NDBuffer[c_type, 2, MutAnyOrigin, c_shape], ctx: DeviceContext,) raises:
    comptime kernel = naive_epilogue_kernel[
        c_type,
        c_shape,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]
    var M = c.dim[0]()
    var N = c.dim[1]()
    comptime simd_size = simd_width_of[c_type]()
    var block_dim = (128 // simd_size, simd_size, 1)
    ctx.enqueue_function[kernel, kernel](
        c,
        grid_dim=(ceildiv(N, block_dim[0]), ceildiv(M, block_dim[1]), 1),
        block_dim=block_dim,
    )


fn naive_epilogue_kernel[
    c_type: DType,
    c_shape: DimList,
    *,
    elementwise_lambda_fn: elementwise_epilogue_type,
](c: NDBuffer[c_type, 2, MutAnyOrigin, c_shape],):
    comptime simd_size = simd_width_of[c_type]()
    comptime alignment = align_of[SIMD[c_type, simd_size]]()
    var n = global_idx.x * UInt(simd_size)
    var m = global_idx.y
    comptime N = c_shape.get[1]()
    var M = c.dim[0]()

    # note that the most naive implementation of simd_size=1 won't work because
    # different threads will be loading and storing in the same 32-bit region
    # leading to synchronization/data race issues.
    if m < UInt(M) and n < UInt(N):
        var val = c.load[width=simd_size, alignment=alignment](Index(m, n))
        elementwise_lambda_fn[c_type, simd_size, alignment=alignment](
            Index(m, n), val
        )


# ===----------------------------------------------------------------------=== #
# H100 grouped matmul
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads)),
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
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters: Int,
):
    comptime assert transpose_b, "Only support transposed B in layout"
    comptime assert num_threads == 128 or num_threads == 256

    M = a_offsets[Int(block_idx.z + 1)] - a_offsets[Int(block_idx.z)]
    comptime N = c.layout.shape[1].value()
    comptime K = b_layout.shape[1].value()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]  # BM
    comptime MMA_N = mma_shape[1]  # BN
    comptime MMA_K = mma_shape[2]  # 16
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    a_start_row = a_offsets[Int(block_idx.z)]
    expert = expert_ids[Int(block_idx.z)]
    b_start_row = expert * Int32(N)

    m_start = block_idx.y * UInt(BM)
    n_start = block_idx.x * UInt(BN)
    a_m_start = UInt(a_start_row) + m_start
    b_n_start = UInt(b_start_row) + n_start
    if m_start >= UInt(M) or n_start >= UInt(N):
        return

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    comptime sub_a_smem_layout = tile_layout_k_major[
        a_type, BM, 64, swizzle_mode=a_swizzle
    ]()
    comptime sub_b_smem_layout = tile_layout_k_major[
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
    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime sub_a_smem_tile_t = LayoutTensor[
        a_type,
        sub_a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime sub_b_smem_tile_t = LayoutTensor[
        b_type,
        sub_b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 16
    ) == 0, "preserve alignment"
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag = SIMD[
        accum_type, c_frag_size
    ]()  # array of accumulator elements

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    comptime max_tmem_cols = 512

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
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            @parameter
            for j in range(
                BK // 64
            ):  # so we do the copy in 64 chunks or 64 elements at a time (BK // 64). but hmm, we said that the K atom can only be 32 bytes (16 elements)
                comptime k = 64 * j
                comptime a_offset = a_smem_layout(IntTuple(0, k))
                comptime b_offset = b_smem_layout(IntTuple(0, k))
                comptime assert ((a_offset * size_of[a_type]()) % 128) == 0
                comptime assert ((b_offset * size_of[b_type]()) % 128) == 0
                sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                # the answer to the above comment. # The descriptor layout i.e. data per copy can be smaller than the shared memory
                # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
                # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
                # so the async_copy takes care of that. TMA engine will copy the data from global tensor into smem tile A
                k_start = UInt(i) * UInt(BK) + UInt(k)
                a_tma_op.async_copy(
                    sub_a_smem_tile,
                    tma_mbar[0],
                    (Int(k_start), Int(a_m_start)),
                )
                sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)
                b_tma_op.async_copy(
                    sub_b_smem_tile,
                    tma_mbar[0],
                    (Int(k_start), Int(b_n_start)),
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

    comptime num_warps = num_threads // WARP_SIZE
    var warp_id = warp_id()

    comptime c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
    comptime c_gmem_type = LayoutTensor[
        c_type,
        c_gmem_layout,
        MutAnyOrigin,
        layout_int_type = DType.int32,
        address_space = AddressSpace.GENERIC,
    ]

    # FIXME: A list literal initializer should be enough here, but somehow Mojo fails to infer that.
    var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
        Index(M, N), Index(N, 1)
    )

    var c_by_expert = c_gmem_type(
        c.ptr + a_start_row * UInt32(N), c_gmem_runtime_layout
    )

    ctile, ctile_coords, _ = c_by_expert.tile_with_offset[BM, BN](
        Int(block_idx.y), Int(block_idx.x)
    )
    comptime c_coord_type = type_of(ctile_coords)

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // num_warps, MMA_N](
                    4 * m_mma + Int(warp_id), n_mma
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

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            @parameter
            for n_vec in range(num_vecs_n):

                @parameter
                for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    comptime dst_idx = type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    comptime dst_m_offset = dst_idx // N
                    comptime dst_n_offset = dst_idx % N
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < UInt32(N):
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        @parameter
                        if elementwise_lambda_fn:
                            comptime alignment = align_of[SIMD[c_type, 2]]()
                            comptime epilogue = elementwise_lambda_fn.value()
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
    b_shape: DimList,
    //,
    *,
    transpose_b: Bool = True,
    mma_shape: IndexList[3] = Index(64, 128, 16),
    block_tile_shape: IndexList[3] = Index(64, 128, 64),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutAnyOrigin, a_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    max_num_tokens_per_expert: Int,
    b: NDBuffer[b_type, 3, MutAnyOrigin, b_shape],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime num_experts = b.shape.get[0]()
    comptime N = b.shape.get[1]()
    comptime K = b.shape.get[2]()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime assert K % BK == 0
    comptime assert BK == 64

    # hard coded 64 for BK

    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_NONE
    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    a_tensor = from_ndbuffer_row_major(a)
    a_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a_tensor
    )
    b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        MutAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)
    b_tma_op = create_tensor_tile[
        Index(BN, BK) if transpose_b else Index(BK, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b_tensor)
    c_tensor = from_ndbuffer_row_major(c)

    comptime block_dim = 128
    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24

    comptime kernel = grouped_matmul_kernel_sm100[
        a_type,
        b_type,
        c_type,
        type_of(a_tensor).layout,
        type_of(b_tensor).layout,
        type_of(a_tma_op).layout,
        type_of(b_tma_op).layout,
        type_of(c_tensor).layout,
        block_tile_shape,
        mma_shape,
        type_of(a_tma_op).desc_layout,
        type_of(b_tma_op).desc_layout,
        a_swizzle,
        b_swizzle,
        c_swizzle,
        transpose_b=transpose_b,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel, kernel](
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
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
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
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_tensor: LayoutTensor[c_type, layout_c, MutAnyOrigin],
    a_tensor: LayoutTensor[a_type, layout_a, MutAnyOrigin],
    b_tensor: LayoutTensor[b_type, layout_b, MutAnyOrigin],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    num_active_experts: Int,
):
    var M = a_offsets[Int(block_idx.z + 1)] - a_offsets[Int(block_idx.z)]
    comptime N = c_tensor.shape[1]()
    comptime K = b_tensor.shape[1]()

    comptime num_experts = b_tensor.shape[0]()
    var expert_id = expert_ids[Int(block_idx.z)]
    var a_start_row = a_offsets[Int(block_idx.z)]

    var a_ptr = a_tensor.ptr + a_start_row * UInt32(K)
    var b_ptr = b_tensor.ptr + expert_id * Int32(N) * Int32(K)
    var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)

    comptime c_layout = Layout.row_major(UNKNOWN_VALUE, N)
    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, K)
    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)

    var c = LayoutTensor[
        c_type,
        c_layout,
        MutAnyOrigin,
        address_space = c_ptr.address_space,
    ](c_ptr, RuntimeLayout[c_layout](Index(M, N), Index(N, 1)))

    var a = LayoutTensor[
        a_type,
        a_layout,
        MutAnyOrigin,
        address_space = a_ptr.address_space,
    ](a_ptr, RuntimeLayout[a_layout](Index(M, K), Index(K, 1)))

    var b = LayoutTensor[
        b_type,
        b_layout,
        MutAnyOrigin,
        address_space = b_ptr.address_space,
    ](b_ptr, RuntimeLayout[b_layout](Index(N, K), Index(K, 1)))

    @always_inline
    @parameter
    fn elementwise_epilogue_fn_wrapper[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        @parameter
        if elementwise_lambda_fn:
            comptime elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_idx = IndexList[2](
                Int(a_start_row + UInt32(idx[0])), idx[1]
            )
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
            Optional[elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ](c, a, b)

    # Perform the epilogue function separately if expert_id is -1
    else:
        _ = c.fill(0.0)

        @parameter
        if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()

            comptime BM = config.block_tile_shape[0]
            comptime BN = config.block_tile_shape[1]
            comptime vec_width = simd_width_of[c_type]()
            comptime alignment = align_of[SIMD[c_type, vec_width]]()

            var block_m = Int(block_idx.y)
            var block_n = Int(block_idx.x)

            # Early exit if this block is completely outside the matrix bounds
            if UInt32(block_m * BM) >= M:
                return

            comptime threads_per_block = 256
            comptime elements_per_thread = ceildiv(BM * BN, threads_per_block)

            var tid = Int(thread_idx.x)
            var thread_start = tid * elements_per_thread
            var thread_end = min(thread_start + elements_per_thread, BM * BN)

            var elements_to_process = thread_end - thread_start

            @always_inline
            fn process_elements[width: Int](idx: Int) unified {mut}:
                var elem_idx = thread_start + idx
                var tile_row, tile_col = divmod(elem_idx, BN)
                var local_row: UInt32 = UInt32(block_m * BM + tile_row)
                var local_col: UInt32 = UInt32(block_n * BN + tile_col)

                if local_row < M:
                    var remaining_in_row = UInt32(N) - local_col
                    var remaining_in_tile_row = BN - tile_col
                    var actual_width = min(
                        width,
                        min(Int(remaining_in_row), remaining_in_tile_row),
                    )

                    if actual_width == width and local_col + UInt32(
                        width
                    ) <= UInt32(N):
                        var zero_vec = SIMD[c_type, width](0.0)
                        epilogue[
                            dtype=c_type,
                            width=width,
                            alignment = align_of[SIMD[c_type, width]](),
                        ]((Int(local_row), Int(local_col)), zero_vec)
                    else:
                        for i in range(actual_width):
                            if local_col + UInt32(i) < UInt32(N):
                                var zero_scalar = SIMD[c_type, 1](0.0)
                                epilogue[dtype=c_type, width=1, alignment=1](
                                    (
                                        Int(local_row),
                                        Int(local_col + UInt32(i)),
                                    ),
                                    zero_scalar,
                                )

            vectorize[vec_width](elements_to_process, process_elements)


@always_inline
fn dispatch_amd_matmul_by_block_shape[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    N: Int,
    K: Int,
    launcher_fn: fn[
        config: MatmulConfig[a_type, b_type, c_type, transpose_b]
    ]() raises capturing -> None,
    default_block_tile_shape: IndexList[3],
    use_heuristic: Bool = False,
](M: Int, ctx: DeviceContext) raises:
    """Dispatches to the best kernel configuration based on runtime M dimension.
    """

    @parameter
    if use_heuristic:
        comptime block_shape_list = _amdgpu_matmul_build_block_shape_list[N]()

        # Auto-tune block shape selection: Find the configuration that minimizes
        # SM idle time by scoring how evenly work distributes across all SMs.
        # Lower score = better load balance (fewer idle SMs in the last wave).
        var best_idx = -1
        var best_score = Int.MAX
        var sm_count = ctx.default_device_info.sm_count

        @parameter
        for i in range(len(block_shape_list)):
            comptime block_shape = block_shape_list[i]
            comptime block_m = block_shape[0]
            comptime block_n = block_shape[1]
            comptime n_blocks = ceildiv(N, block_n)

            var m_blocks = ceildiv(M, block_m)
            var total_blocks = m_blocks * n_blocks
            var batch, extra = divmod(total_blocks - 1, sm_count)
            var score = batch * sm_count + (sm_count - extra - 1)

            if score < best_score:
                best_idx = i
                best_score = score

        # Dispatch to the best configuration if found
        @parameter
        for i in range(len(block_shape_list)):
            if best_idx == i:
                comptime config = _amdgpu_matmul_config_from_block_shape[
                    c_type,
                    a_type,
                    b_type,
                    transpose_b,
                    K,
                    pdl_level = PDLLevel(),
                ](block_shape_list[i])
                launcher_fn[config]()
                return

    # Fallback to default config
    @always_inline
    @parameter
    fn default_config_launcher[
        block_m: Int,
        block_n: Int,
        block_k: Int,
    ]() raises:
        comptime default_config = MatmulConfig[
            a_type, b_type, c_type, transpose_b
        ](
            block_tile_shape=Index(block_m, block_n, block_k),
            warp_tile_shape=Index(
                block_m // 2,
                block_n // 2,
                block_k,
            ),
            num_pipeline_stages=1,
            num_k_partitions=1,
        )
        launcher_fn[default_config]()

    # auto-tuned sizes
    if M == 128 and N == 256 and K == 256:
        default_config_launcher[32, 32, 128]()
    elif M == 256 and N == 512 and K == 1024:
        default_config_launcher[32, 32, 128]()
    elif M == 384 and N == 768 and K == 1024:
        default_config_launcher[32, 64, 128]()
    elif M == 1977 and N == 192 and K == 1024:
        default_config_launcher[64, 96, 128]()
    elif M == 1977 and N == 1280 and K == 1024:
        default_config_launcher[96, 96, 64]()
    else:
        default_config_launcher[64, 64, 64]()


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
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutAnyOrigin, a_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    max_num_tokens_per_expert: Int,
    b: NDBuffer[b_type, 3, MutAnyOrigin, b_shape],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime num_experts = b.shape.get[0]()
    comptime N = b.shape.get[1]()
    comptime K = b.shape.get[2]()

    var total_M = 0
    for i in range(num_active_experts):
        total_M += Int(a_offsets[i + 1] - a_offsets[i])

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime assert K % BK == 0

    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        MutAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)
    var c_tensor = from_ndbuffer_row_major(c)

    comptime block_dim = 256

    @always_inline
    @parameter
    @__copy_capture(
        c_tensor,
        a_tensor,
        b_tensor,
        a_offsets,
        expert_ids,
        num_active_experts,
        max_num_tokens_per_expert,
    )
    fn launch_kernel[
        config: MatmulConfig[a_type, b_type, c_type, transpose_b]
    ]() raises:
        comptime kernel = grouped_matmul_amd_kernel_launcher[
            c_type,
            a_type,
            b_type,
            type_of(c_tensor).layout,
            type_of(a_tensor).layout,
            type_of(b_tensor).layout,
            transpose_b,
            config,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function[kernel, kernel](
            c_tensor,
            a_tensor,
            b_tensor,
            a_offsets,
            expert_ids,
            num_active_experts,
            grid_dim=(
                ceildiv(N, config.block_tile_shape[1]),
                ceildiv(max_num_tokens_per_expert, config.block_tile_shape[0]),
                num_active_experts,
            ),
            block_dim=(block_dim),
        )

    # Dispatch to the best configuration based on runtime dimensions
    dispatch_amd_matmul_by_block_shape[
        c_type,
        a_type,
        b_type,
        transpose_b,
        N,
        K,
        launch_kernel,
        block_tile_shape,
    ](total_M, ctx)


# ===----------------------------------------------------------------------=== #
# Entry Point and Dispatch
# ===----------------------------------------------------------------------=== #


fn grouped_matmul[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    //,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime is_expert_shape_static = b_shape.all_known[
        3
    ]() and a_shape.has_value[1]() and c_shape.has_value[1]()
    comptime is_sm90_kernel_applicable = ctx.default_device_info == H100 and is_expert_shape_static
    comptime is_sm100_kernel_applicable = ctx.default_device_info == B200 and is_expert_shape_static
    comptime is_amd_kernel_applicable = has_amd_gpu_accelerator() and not has_amd_rdna_gpu_accelerator() and is_expert_shape_static

    @parameter
    if is_sm90_kernel_applicable:
        comptime static_N = c.shape.get[1]()
        comptime BN = _find_largest_bn_for_sm90_matmul[a_type, static_N]()
        comptime wgmma_shape = IndexList[3](64, BN, 16)

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
        comptime N = b.shape.get[1]()
        comptime K = b.shape.get[2]()
        comptime contiguous_bytes = K * size_of[a_type]()

        fn get_swizzle_mode(contiguous_bytes: Int) -> TensorMapSwizzle:
            if contiguous_bytes >= TensorMapSwizzle.SWIZZLE_128B.bytes():
                return TensorMapSwizzle.SWIZZLE_128B
            elif contiguous_bytes >= TensorMapSwizzle.SWIZZLE_64B.bytes():
                return TensorMapSwizzle.SWIZZLE_64B
            elif contiguous_bytes >= TensorMapSwizzle.SWIZZLE_32B.bytes():
                return TensorMapSwizzle.SWIZZLE_32B
            else:
                return TensorMapSwizzle.SWIZZLE_NONE

        comptime a_swizzle = get_swizzle_mode(contiguous_bytes)
        comptime b_swizzle = a_swizzle
        comptime BK = (a_swizzle.bytes() // size_of[a_type]())
        comptime _MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
        comptime MMA_K = min(_MMA_K, K)
        # For cta_group = 2, N must be divisible by 256 to ensure correct tiling and memory alignment for the kernel.
        comptime cta_group = 2 if N % 256 == 0 else 1
        comptime block_tile_shape = Index(128, 32 // cta_group, BK)
        comptime umma_shape = Index(
            block_tile_shape[0] * cta_group,
            block_tile_shape[1] * cta_group,
            MMA_K,
        )
        comptime cluster_shape = Index(cta_group, 1, 1)
        comptime transpose_b = True
        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=block_tile_shape,
            mma_shape=umma_shape,
            cluster_shape=cluster_shape,
        )
        comptime assert (
            K % BK == 0
        ), "b_shape[2] must be a multiple of BK. Got " + String(K)

        grouped_matmul_sm100_persistent[
            transpose_b=transpose_b,
            config=config,
            cta_group=cta_group,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            elementwise_lambda_fn=elementwise_lambda_fn,
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
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B in grouped matmul."
    comptime assert (
        a_type == b_type
    ), "A and B must have the same dtype for vendor BLAS"
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
            var c_slice = NDBuffer[c_type, 2, MutAnyOrigin](
                c.data + token_start * UInt32(c.dim[1]()),
                DimList(num_tokens, c.dim[1]()),
            )
            var buff = DeviceBuffer(
                ctx, c_slice.data, c_slice.num_elements(), owning=False
            )
            ctx.enqueue_memset(buff, 0)
            continue

        # Create views into the tensors for this expert
        var a_slice = NDBuffer[a_type, 2, MutAnyOrigin](
            a.data + token_start * UInt32(a.dim[1]()),
            DimList(num_tokens, a.dim[1]()),
        )
        var b_slice = NDBuffer[b_type, 2, MutAnyOrigin](
            b.data + expert_id * Int32(b.dim[1]()) * Int32(b.dim[2]()),
            DimList(b.dim[1](), b.dim[2]()),
        )
        var c_slice = NDBuffer[c_type, 2, MutAnyOrigin](
            c.data + token_start * UInt32(c.dim[1]()),
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
