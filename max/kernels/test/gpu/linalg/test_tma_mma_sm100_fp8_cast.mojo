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


from std.sys import size_of

from std.math.uutils import udivmod
from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import block_rank_in_cluster
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, lane_id, thread_idx, warp_id as get_warp_id
from max.gpu.memory import external_memory, fence_async_view_proxy
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    Coord,
    Idx,
    MixedLayout,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


def cpu_matmul_naive[
    *, transpose_a: Bool, transpose_b: Bool
](C: TileTensor[mut=True, ...], A: TileTensor, B: TileTensor):
    var M = Int(C.dim[0]())
    var N = Int(C.dim[1]())
    var K = Int(A.dim[0]()) if transpose_a else Int(A.dim[1]())
    var a_ptr = A.unsafe_ptr()
    var b_ptr = B.unsafe_ptr()
    var c_ptr = C.unsafe_ptr()
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
                    a_ptr[a_idx].cast[.float32]()
                    * b_ptr[b_idx].cast[.float32]()
                )
            c_ptr[m * N + n] = acc.cast[C.dtype]()


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
def tma_umma_kernel_sgs[
    a_type: DType,  # A type in gmem and smem (bfloat16)
    b_gmem_type: DType,  # B type in gmem (float8_e4m3fn)
    c_type: DType,  # Output type (bfloat16)
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_layout: TensorLayout,  # B's gmem layout (FP8)
    c_layout: TensorLayout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b: TileTensor[b_gmem_type, b_layout, ImmutAnyOrigin],  # FP8 in gmem
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    """Kernel with A via TMA to smem, B from gmem->registers->cast->smem.

    Matrix A: Loaded via TMA to shared memory (existing pattern)
    Matrix B: FP8 in global memory, loaded to registers, cast to BF16, stored to smem
    MMA: Uses BF16 operands (KIND_F16)
    """
    var num_iters = Int(num_iters_dev)
    comptime assert num_threads == 128 or num_threads == 256
    comptime assert (
        a_type == .bfloat16
    ), "a_type must be bfloat16 for this kernel"
    comptime assert (
        b_gmem_type == .float8_e4m3fn
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
    comptime b_k_major = transpose_b
    comptime a_smem_layout = tile_layout_k_major_typed[
        a_type, BM, BK, a_swizzle
    ]

    # B smem layout uses bfloat16, NOT b_gmem_type (fp8)
    comptime b_smem_type = DType.bfloat16
    comptime BSmemLayout = type_of(
        tile_layout_k_major_typed[b_smem_type, BN, BK, b_swizzle]
    ) if b_k_major else type_of(
        tile_layout_mn_major_typed[b_smem_type, BN, BK, b_swizzle]
    )
    comptime b_smem_layout = MixedLayout[
        shape_types=BSmemLayout._shape_types,
        stride_types=BSmemLayout._stride_types,
    ]()

    var a_smem = rebind[
        MutPointer[Scalar[a_type], address_space=.SHARED, MutUntrackedOrigin]
    ](
        external_memory[
            Scalar[a_type],
            address_space=.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )

    comptime a_size = BM * BK
    comptime b_size = BN * BK

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_smem_type]()) % 16
    ) == 0, "preserve alignment"
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_smem_type]]()

    var a_smem_tile = TileTensor(a_smem, a_smem_layout)
    var b_smem_tile = TileTensor(b_smem, b_smem_layout)

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag: Array[Scalar[accum_type], c_frag_size]

    comptime a_expected_bytes = a_size * size_of[a_type]()
    # B is loaded manually, not via TMA

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

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

    var tmem_addr = ptr_tmem_addr[0]

    comptime if num_threads > 128:
        if thread_idx.x >= 128:
            tmem_addr += 1 << 20  # offset for lane 16

    # The typed tile layout is ((8, MN/8), (sw, K/sw)) for K-major and its
    # transpose for MN-major, so the two descriptor strides are the outer
    # strides at flattened positions 1 and 3; swizzled MN-major swaps them.
    comptime aSBO = type_of(a_smem_layout).static_stride[1] * size_of[a_type]()
    comptime aLBO = type_of(a_smem_layout).static_stride[3] * size_of[a_type]()
    comptime b_shape00 = type_of(b_smem_layout).static_shape[0]
    comptime b_shape10 = type_of(b_smem_layout).static_shape[2]
    comptime b_stride00 = type_of(b_smem_layout).static_stride[0]
    comptime b_stride01 = type_of(b_smem_layout).static_stride[1]
    comptime b_stride10 = type_of(b_smem_layout).static_stride[2]
    comptime b_stride11 = type_of(b_smem_layout).static_stride[3]
    comptime bSBO = (
        b_stride01 if b_k_major
        or b_swizzle == TensorMapSwizzle.SWIZZLE_NONE else b_stride11
    ) * size_of[b_smem_type]()
    comptime bLBO = (
        b_stride11 if b_k_major
        or b_swizzle == TensorMapSwizzle.SWIZZLE_NONE else b_stride01
    ) * size_of[b_smem_type]()

    var adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](
        a_smem_tile.unsafe_ptr()
    )
    var bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](
        b_smem_tile.unsafe_ptr()
    )

    # Use KIND_F16 since both A and B are BF16 in smem
    comptime mma_kind = UMMAKind.KIND_F16
    var idesc = UMMAInsDescriptor[mma_kind].create[
        accum_type,
        a_type,
        b_smem_type,  # bfloat16
        Index[dtype=.uint32](mma_shape[0], mma_shape[1]),
        transpose_a=False,  # A is not transposed
        transpose_b=transpose_b,
    ]()

    comptime num_warps = num_threads // WARP_SIZE
    var warp_id = get_warp_id()

    comptime if num_threads > 128:
        var warp_id_q, warp_id_r = udivmod(warp_id, 4)
        warp_id = 2 * warp_id_r + warp_id_q

    for i in range(num_iters):
        # Load A via TMA
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(a_expected_bytes))

            var m = block_idx.y * BM
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
        comptime elems_per_thread = (BN * BK) // num_threads
        comptime simd_size = 8
        comptime assert elems_per_thread % simd_size == 0

        var tid = Int(thread_idx.x)
        comptime swizzle = make_swizzle[b_smem_type, b_swizzle]()

        comptime for elem in range(elems_per_thread // simd_size):
            var local_idx = simd_size * (elem * num_threads + tid)

            var n_local: Int
            var k_local: Int
            # Compute local tile coordinates based on memory layout
            # transpose_b=True: gmem NxK (K fast), smem K-major (K fast)
            # transpose_b=False: gmem KxN (N fast), smem N-major (N fast)
            comptime if transpose_b:
                n_local, k_local = divmod(local_idx, BK)
            else:
                k_local, n_local = divmod(local_idx, BN)

            # Global coordinates
            var gmem_n = Int(block_idx.x) * BN + n_local
            var gmem_k = i * BK + k_local

            var fp8_val: SIMD[b_gmem_type, simd_size]
            # Load from gmem - layout is NxK when transpose_b, KxN otherwise
            comptime if transpose_b:
                fp8_val = b.load[width=simd_size](Coord(gmem_n, gmem_k))
            else:
                fp8_val = b.load[width=simd_size](Coord(gmem_k, gmem_n))

            # Cast and store to smem using local coordinates
            var bf16_val = fp8_val.cast[b_smem_type]()
            var n_q, n_r = divmod(n_local, b_shape00)
            var n_offset = n_q * b_stride01 + n_r * b_stride00
            var k_q, k_r = divmod(k_local, b_shape10)
            var k_offset = k_q * b_stride11 + k_r * b_stride10
            var offset = swizzle(n_offset + k_offset)
            b_smem_tile.raw_store[width=simd_size, alignment=2 * simd_size](
                offset, bf16_val
            )

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
                    comptime idx = Coord(Idx[0], Idx[MMA_K * j])
                    comptime a_offset = Int(a_smem_layout(idx)) * size_of[
                        a_type
                    ]()
                    comptime b_offset = Int(b_smem_layout(idx)) * size_of[
                        b_smem_type
                    ]()
                    mma[c_scale=1](
                        adesc + a_offset, bdesc + b_offset, tmem_addr, idesc
                    )
            else:
                comptime for j in range(num_k_mmas):
                    comptime idx = Coord(Idx[0], Idx[MMA_K * j])
                    comptime a_offset = Int(a_smem_layout(idx)) * size_of[
                        a_type
                    ]()
                    comptime b_offset = Int(b_smem_layout(idx)) * size_of[
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
        repeat=BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    var ctile = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            var c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + Int(warp_id), n_mma
            )

            var c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                row_major[8, 4]()
            ](Int(lane_id()))

            comptime num_vecs_m = type_of(c_gmem_frag).static_shape[0]
            comptime num_vecs_n = type_of(c_gmem_frag).static_shape[1]

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        type_of(c_gmem_frag).ElementType
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
](ctx: DeviceContext) raises:
    """Test for FP8 B with gmem->registers->cast->smem pattern.

    Matrix A: Loaded via TMA to shared memory (bfloat16)
    Matrix B: FP8 in global memory, loaded to registers, cast to BF16, stored to smem
    MMA: Uses BF16 operands (KIND_F16)
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime MMA_M = mma_shape[0]

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
    var a = HostDeviceTileTensor[a_type](row_major[M, K](), ctx)
    var a_host = a.host_tensor()

    var a_extreme: Float32 = 10
    random(
        a_host,
        min=(-a_extreme).cast[a_type](),
        max=a_extreme.cast[a_type](),
    )

    # B is FP8 in global memory
    comptime b_rows = N if transpose_b else K
    comptime b_cols = K if transpose_b else N
    var b = HostDeviceTileTensor[b_gmem_type](row_major[b_rows, b_cols](), ctx)
    var b_host = b.host_tensor()
    # Create a BF16 copy of B for the CPU reference computation
    var b_bf16 = HostDeviceTileTensor[.bfloat16](row_major[b_rows, b_cols]())
    var b_bf16_host = b_bf16.host_tensor()

    var b_extreme: Float32 = 10
    random(
        b_host,
        min=(-b_extreme).cast[b_gmem_type](),
        max=b_extreme.cast[b_gmem_type](),
    )

    # Cast B from FP8 to BF16 for reference computation
    for row in range(b_rows):
        for col in range(b_cols):
            b_bf16_host[row, col] = b_host[row, col].cast[.bfloat16]()

    var c = HostDeviceTileTensor[c_type](row_major[M, N](), ctx)
    var c_ref = HostDeviceTileTensor[c_type](row_major[M, N]())

    a.to_device()
    b.to_device()

    var b_dev = b.device_tensor()
    var c_dev = c.device_tensor()

    # Only A uses TMA
    var a_tma_op = create_tensor_tile[
        Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a.device_tensor())

    comptime block_dim = 2 * MMA_M

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
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_dev).LayoutType,
        type_of(c_dev).LayoutType,
        block_tile_shape,
        mma_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
    ]
    ctx.enqueue_function[kernel](
        a_tma_op,
        b_dev.as_imm(),
        c_dev,
        Int32(K // BK),
        grid_dim=(N // BN, M // BM),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )

    # Reference computation using CPU to avoid any device sync issues
    var c_ref_host = c_ref.host_tensor()
    cpu_matmul_naive[transpose_a=False, transpose_b=transpose_b](
        c_ref_host, a_host, b_bf16_host
    )

    ctx.synchronize()
    c.to_host()

    var c_host = c.host_tensor()

    for m in range(M):
        for n in range(N):
            # Increased tolerance for FP8/bfloat16 accumulation errors
            # FP8/bf16 matrix multiplication can have larger numerical errors
            # due to reduced precision in intermediate accumulations
            assert_almost_equal(
                c_host[m, n],
                c_ref_host[m, n],
                atol=0.01,
                rtol=0.01,
                msg=String(m) + ", " + String(n),
            )


def main() raises:
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
                        .bfloat16,  # A type
                        .float8_e4m3fn,  # B gmem type
                        .bfloat16,  # C type
                        Index(MMA_M, 128, BK),  # prob_shape matching block_tile
                        Index(MMA_M, 128, BK),  # block_tile
                        Index(MMA_M, 128, MMA_K),  # mma_shape
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,  # No swizzle for B
                        transpose_b=transpose_b,
                    ](ctx)

                    # Test with multiple K iterations
                    test_tma_umma_fp8_b[
                        .bfloat16,
                        .float8_e4m3fn,
                        .bfloat16,
                        Index(MMA_M, 128, BK * 2),  # 2 K iterations
                        Index(MMA_M, 128, BK),
                        Index(MMA_M, 128, MMA_K),
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,
                        transpose_b=transpose_b,
                    ](ctx)

                    # Test multi-block in M dimension
                    test_tma_umma_fp8_b[
                        .bfloat16,
                        .float8_e4m3fn,
                        .bfloat16,
                        Index(MMA_M * 2, 128, BK),  # 2 M blocks
                        Index(MMA_M, 128, BK),
                        Index(MMA_M, 128, MMA_K),
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,
                        transpose_b=transpose_b,
                    ](ctx)

                    # Test multi-block in N dimension
                    test_tma_umma_fp8_b[
                        .bfloat16,
                        .float8_e4m3fn,
                        .bfloat16,
                        Index(MMA_M, 128 * 2, BK),  # 2 N blocks
                        Index(MMA_M, 128, BK),
                        Index(MMA_M, 128, MMA_K),
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,
                        transpose_b=transpose_b,
                    ](ctx)
