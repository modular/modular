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
"""Test kernel for FP8 B matrix with gmem->registers->cast->smem pattern.

Matrix A: Loaded via TMA to shared memory (bfloat16)
Matrix B: FP8 in global memory, loaded to registers, cast to BF16, stored to smem
MMA: Uses BF16 operands (KIND_F16)
"""

from math import sqrt
from memory import LegacyUnsafePointer, bitcast

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from gpu import WARP_SIZE, barrier
from gpu import lane_id as get_lane_id
from gpu.primitives.cluster import block_rank_in_cluster
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_idx, lane_id, thread_idx, warp_id as get_warp_id
from gpu.memory import external_memory, fence_async_view_proxy
from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.compute.arch.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList
from utils.numerics import get_accum_type, max_finite, min_finite
from utils.static_tuple import StaticTuple


fn cpu_matmul_naive[
    *, transpose_a: Bool, transpose_b: Bool
](C: LayoutTensor, A: LayoutTensor, B: LayoutTensor):
    comptime M = C.layout[0].size()
    comptime N = C.layout[1].size()
    # layout_a is M x K
    comptime layout_a = A.layout.transpose() if transpose_a else A.layout
    # layout_b is K x N
    comptime layout_b = B.layout.transpose() if transpose_b else B.layout
    comptime K = layout_a[1].size()
    comptime assert M == layout_a[0].size(), String(
        "C.M = ", M, "; A.M = ", layout_a[0].size()
    )
    comptime assert N == layout_b[1].size(), String(
        "C.N = ", M, "; B.N = ", layout_b[1].size()
    )
    comptime assert K == layout_b[0].size(), String(
        "A.K = ", K, "; B.K = ", layout_b[0].size()
    )
    for n in range(N):
        for m in range(M):
            var acc: Float32 = 0.0
            for k in range(K):
                var a_idx: Int

                comptime if transpose_a:
                    a_idx = k * M + m
                else:
                    a_idx = m * K + k
                var b_idx: Int

                comptime if transpose_b:
                    b_idx = n * K + k
                else:
                    b_idx = k * N + n
                acc += (
                    A.ptr.load(a_idx).cast[DType.float32]()
                    * B.ptr.load(b_idx).cast[DType.float32]()
                )
            c_idx = m * N + n
            C.ptr.store(c_idx, acc.cast[C.dtype]())


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
fn tma_umma_kernel_sgs[
    a_type: DType,  # A type in gmem and smem (bfloat16)
    b_gmem_type: DType,  # B type in gmem (float8_e4m3fn)
    c_type: DType,  # Output type (bfloat16)
    a_layout: Layout,
    b_layout: Layout,  # B's gmem layout (FP8)
    c_layout: Layout,
    a_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b: LayoutTensor[b_gmem_type, b_layout, MutAnyOrigin],  # FP8 in gmem
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters: Int,
):
    """Kernel with A via TMA to smem, B from gmem->registers->cast->smem.

    Matrix A: Loaded via TMA to shared memory (existing pattern)
    Matrix B: FP8 in global memory, loaded to registers, cast to BF16, stored to smem
    MMA: Uses BF16 operands (KIND_F16)
    """
    comptime assert num_threads == 128 or num_threads == 256
    comptime assert (
        a_type == DType.bfloat16
    ), "a_type must be bfloat16 for this kernel"
    comptime assert (
        b_gmem_type == DType.float8_e4m3fn
    ), "b_gmem_type must be float8_e4m3fn for this kernel"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    # A smem layout unchanged (uses a_type = bfloat16)
    comptime a_k_major = True  # A is not transposed, so k-major
    comptime b_k_major = transpose_b
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()

    # B smem layout uses bfloat16, NOT b_gmem_type (fp8)
    comptime b_smem_type = DType.bfloat16
    comptime b_smem_layout = tile_layout_k_major[
        b_smem_type, BN, BK, swizzle_mode=b_swizzle
    ]() if b_k_major else tile_layout_mn_major[
        b_smem_type, BN, BK, swizzle_mode=b_swizzle
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
    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_smem_type,  # BF16 in smem
        b_smem_layout,
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
        (b_size * size_of[b_smem_type]()) % 16
    ) == 0, "preserve alignment"
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_smem_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // Int(num_threads)
    var c_frag = SIMD[accum_type, c_frag_size]()

    comptime a_expected_bytes = a_size * size_of[a_type]()
    # B is loaded manually, not via TMA

    tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = ptr_tmem_addr[0]

    comptime if num_threads > 128:
        if thread_idx.x >= 128:
            tmem_addr += 1 << 20  # offset for lane 16

    comptime a_canonical_layout = tile_to_descriptor[
        a_type, a_smem_layout, is_k_major=a_k_major
    ]()
    comptime b_canonical_layout = tile_to_descriptor[
        b_smem_type, b_smem_layout, is_k_major=b_k_major
    ]()
    comptime a_stride01 = a_canonical_layout[0].stride[1].value()
    comptime a_stride11 = a_canonical_layout[1].stride[1].value()
    comptime aSBO = (
        a_stride01 if a_k_major
        or a_swizzle == TensorMapSwizzle.SWIZZLE_NONE else a_stride11
    ) * size_of[a_type]()
    comptime aLBO = (
        a_stride11 if a_k_major
        or a_swizzle == TensorMapSwizzle.SWIZZLE_NONE else a_stride01
    ) * size_of[a_type]()
    comptime b_shape00 = b_canonical_layout[0].shape[0].value()
    comptime b_shape10 = b_canonical_layout[1].shape[0].value()
    comptime b_stride00 = b_canonical_layout[0].stride[0].value()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride10 = b_canonical_layout[1].stride[0].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (
        b_stride01 if b_k_major
        or b_swizzle == TensorMapSwizzle.SWIZZLE_NONE else b_stride11
    ) * size_of[b_smem_type]()
    comptime bLBO = (
        b_stride11 if b_k_major
        or b_swizzle == TensorMapSwizzle.SWIZZLE_NONE else b_stride01
    ) * size_of[b_smem_type]()

    adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    # Use KIND_F16 since both A and B are BF16 in smem
    comptime mma_kind = UMMAKind.KIND_F16
    idesc = UMMAInsDescriptor[mma_kind].create[
        accum_type,
        a_type,
        b_smem_type,  # bfloat16
        Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
        transpose_a=False,  # A is not transposed
        transpose_b=transpose_b,
    ]()

    comptime num_warps = num_threads // Int(WARP_SIZE)
    var warp_id = get_warp_id()

    comptime if num_threads > 128:
        warp_id = UInt(Int(2 * Int(warp_id % 4) + Int(warp_id // 4)))

    for i in range(num_iters):
        # Load A via TMA
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(a_expected_bytes))

            var m = Int(block_idx.y) * BM
            var k = Int(i) * BK
            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (k, m),
            )

        # Load B from global memory, cast to BF16, store to smem
        # B is NxK in gmem when transpose_b=True, KxN when transpose_b=False
        # Use explicit element-by-element copy for simplicity
        # Each thread handles a portion of the BN*BK elements
        comptime elems_per_thread = (BN * BK) // Int(num_threads)
        comptime simd_size = 8
        comptime assert elems_per_thread % simd_size == 0
        comptime K_total = b_layout.shape[
            1
        ].value() if transpose_b else b_layout.shape[0].value()
        comptime N_total = b_layout.shape[
            0
        ].value() if transpose_b else b_layout.shape[1].value()

        var tid = Int(thread_idx.x)
        comptime swizzle = make_swizzle[b_smem_type, b_swizzle]()

        comptime for elem in range(elems_per_thread // simd_size):
            local_idx = simd_size * (elem * num_threads + tid)

            # Compute local tile coordinates based on memory layout
            # transpose_b=True: gmem NxK (K fast), smem K-major (K fast)
            # transpose_b=False: gmem KxN (N fast), smem N-major (N fast)
            comptime if transpose_b:
                n_local = local_idx // BK
                k_local = local_idx % BK
            else:
                k_local = local_idx // BN
                n_local = local_idx % BN

            # Global coordinates
            gmem_n = Int(block_idx.x) * BN + n_local
            gmem_k = Int(i) * BK + k_local

            # Load from gmem - layout is NxK when transpose_b, KxN otherwise
            comptime if transpose_b:
                fp8_val = b.ptr.load[width=simd_size, alignment=simd_size](
                    gmem_n * K_total + gmem_k
                )
            else:
                fp8_val = b.ptr.load[width=simd_size, alignment=simd_size](
                    gmem_k * N_total + gmem_n
                )

            # Cast and store to smem using local coordinates
            bf16_val = fp8_val.cast[b_smem_type]()
            n_offset = (n_local // b_shape00) * b_stride01 + (
                n_local % b_shape00
            ) * b_stride00
            k_offset = (k_local // b_shape10) * b_stride11 + (
                k_local % b_shape10
            ) * b_stride10
            offset = swizzle(n_offset + k_offset)
            b_smem_tile.ptr.store[alignment = 2 * simd_size](offset, bf16_val)

        # Sync: wait for TMA to complete and all threads to finish storing to smem
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1
        # This fence is needed for correctness!
        fence_async_view_proxy()
        barrier()

        if elect_one_thread:
            if i == 0:
                mma[c_scale=0](adesc, bdesc, tmem_addr, idesc)

                comptime for j in range(1, num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[
                        b_smem_type
                    ]()
                    mma[c_scale=1](
                        adesc + a_offset, bdesc + b_offset, tmem_addr, idesc
                    )
            else:
                comptime for j in range(num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[
                        b_smem_type
                    ]()
                    mma[c_scale=1](
                        adesc + a_offset, bdesc + b_offset, tmem_addr, idesc
                    )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    ctile = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile = ctile.tile[MMA_M // Int(num_warps), MMA_N](
                4 * m_mma + Int(warp_id), n_mma
            )

            c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


def test_tma_umma_fp8_b[
    a_type: DType,  # bfloat16
    b_gmem_type: DType,  # float8_e4m3fn
    c_type: DType,  # bfloat16
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext):
    """Test for FP8 B with gmem->registers->cast->smem pattern.

    Matrix A: Loaded via TMA to shared memory (bfloat16)
    Matrix B: FP8 in global memory, loaded to registers, cast to BF16, stored to smem
    MMA: Uses BF16 operands (KIND_F16)
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    print(
        "mma_sgs_"
        + String(a_type)
        + "_"
        + String(b_gmem_type)
        + "_"
        + String(c_type)
        + " problem shape "
        + String(prob_shape)
        + " block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(mma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
    )

    comptime M = prob_shape[0]
    comptime N = prob_shape[1]
    comptime K = prob_shape[2]

    # A is bfloat16, row-major (M x K)
    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(M, K),
    ](ctx)

    var a_extreme: Float32 = 10
    random(
        a.tensor[update=False](),
        min=(-a_extreme).cast[a_type](),
        max=a_extreme.cast[a_type](),
    )

    # B is FP8 in global memory
    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_gmem_type, b_layout](ctx)
    # Create a BF16 copy of B for reference computation (cuBLAS needs matching types)
    var b_bf16 = ManagedLayoutTensor[DType.bfloat16, b_layout](ctx)

    var b_extreme: Float32 = 10
    random(
        b.tensor[update=False](),
        min=(-b_extreme).cast[b_gmem_type](),
        max=b_extreme.cast[b_gmem_type](),
    )

    # Cast B from FP8 to BF16 for reference computation
    # Use update=False to access host buffers directly (random data is on host)
    var b_host_for_copy = b.tensor[update=False]()
    var b_bf16_host = b_bf16.tensor[update=False]()
    for row in range(b_layout.shape[0].value()):
        for col in range(b_layout.shape[1].value()):
            b_bf16_host[row, col] = b_host_for_copy[row, col].cast[
                DType.bfloat16
            ]()

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    # Only A uses TMA
    a_tma_op = create_tensor_tile[
        Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a.device_tensor())

    comptime block_dim = Int(2 * MMA_M)

    # smem_use accounts for BF16 B size (not FP8)
    # A: BM * BK * sizeof(bfloat16)
    # B: BN * BK * sizeof(bfloat16) -- stored as BF16 after cast
    comptime smem_use = BM * size_of[a_type]() * BK + BN * size_of[
        DType.bfloat16
    ]() * BK + 24

    comptime kernel = tma_umma_kernel_sgs[
        a_type,
        b_gmem_type,
        c_type,
        type_of(a_tma_op).layout,
        b_layout,
        Layout.row_major(M, N),
        type_of(a_tma_op).desc_layout,
        block_tile_shape,
        mma_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
    ]
    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b.device_tensor(),
        c.device_tensor(),
        Int(K // BK),
        grid_dim=(N // BN, M // BM),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )

    # Reference computation using CPU to avoid any device sync issues
    # This uses the host tensors directly - capture tensors explicitly
    var a_host = a.tensor[update=False]()
    var c_ref_host = c_ref.tensor[update=False]()
    cpu_matmul_naive[transpose_a=False, transpose_b=transpose_b](
        c_ref_host,
        a_host,
        b_bf16_host,  # Already captured above for B copy
    )

    ctx.synchronize()

    # Get kernel results back from device
    c_host = c.tensor()
    # Use c_ref_host directly since CPU reference wrote to it

    for m in range(M):
        for n in range(N):
            # Increased tolerance for FP8/bfloat16 accumulation errors
            # FP8/bf16 matrix multiplication can have larger numerical errors
            # due to reduced precision in intermediate accumulations
            assert_almost_equal(
                c_host[m, n],
                c_ref_host[m, n],  # Use captured host tensor
                atol=0.01,
                rtol=0.01,
                msg=String(m) + ", " + String(n),
            )

    _ = a^
    _ = b^
    _ = b_bf16^
    _ = c^
    _ = c_ref^


def main():
    with DeviceContext() as ctx:
        # Test FP8 B with gmem->cast->smem pattern (sgs kernel)
        # A: bfloat16 via TMA, B: FP8 in gmem -> cast to BF16 -> smem, MMA: BF16
        comptime for transpose_b in [True, False]:
            comptime for a_swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
                comptime for b_swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
                    # BK for BF16 MMA (not FP8)
                    comptime BK = a_swizzle.bytes() // size_of[
                        DType.bfloat16
                    ]()  # 64 for BF16

                    # Only use MMA_M=64 for now; MMA_M=128 with 256 threads has tmem issues
                    comptime MMA_M = 64
                    comptime MMA_K = 16  # BF16 MMA_K

                    # Test single block case with SWIZZLE_NONE for B
                    # to avoid swizzle complexity in manual B loading
                    test_tma_umma_fp8_b[
                        DType.bfloat16,  # A type
                        DType.float8_e4m3fn,  # B gmem type
                        DType.bfloat16,  # C type
                        Index(MMA_M, 128, BK),  # prob_shape matching block_tile
                        Index(MMA_M, 128, BK),  # block_tile
                        Index(MMA_M, 128, MMA_K),  # mma_shape
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,  # No swizzle for B
                        transpose_b=transpose_b,
                    ](ctx)

                    # Test with multiple K iterations
                    test_tma_umma_fp8_b[
                        DType.bfloat16,
                        DType.float8_e4m3fn,
                        DType.bfloat16,
                        Index(MMA_M, 128, BK * 2),  # 2 K iterations
                        Index(MMA_M, 128, BK),
                        Index(MMA_M, 128, MMA_K),
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,
                        transpose_b=transpose_b,
                    ](ctx)

                    # Test multi-block in M dimension
                    test_tma_umma_fp8_b[
                        DType.bfloat16,
                        DType.float8_e4m3fn,
                        DType.bfloat16,
                        Index(MMA_M * 2, 128, BK),  # 2 M blocks
                        Index(MMA_M, 128, BK),
                        Index(MMA_M, 128, MMA_K),
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,
                        transpose_b=transpose_b,
                    ](ctx)

                    # Test multi-block in N dimension
                    test_tma_umma_fp8_b[
                        DType.bfloat16,
                        DType.float8_e4m3fn,
                        DType.bfloat16,
                        Index(MMA_M, 128 * 2, BK),  # 2 N blocks
                        Index(MMA_M, 128, BK),
                        Index(MMA_M, 128, MMA_K),
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,
                        transpose_b=transpose_b,
                    ](ctx)
