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
"""Tests SM100TensorAccumulatorSS with cta_group=2 via bulk_mma pair-CTA path.

This exercises the parameterized cta_group in build_mma_ss -> bulk_mma ->
SM100TensorAccumulatorSS.mma, validating that the generated PTX with
cta_group::2 and 8-element mask produces correct results.
"""

from std.math import align_up
from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from std.gpu import barrier
from std.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync_with_mask,
)
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu import (
    block_id_in_cluster,
    block_idx_uint as block_idx,
    lane_id_uint as lane_id,
    warp_id_uint as warp_id,
)
from std.gpu.memory import external_memory
from std.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
    mma_arrive_multicast,
)
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from layout import Layout, LayoutTensor
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulatorSS,
    elect,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type, max_finite, min_finite
from std.utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def bulk_mma_pair_cta_kernel[
    ab_type: DType,
    c_type: DType,
    a_tma_rank: Int,
    b_tma_rank: Int,
    a_tile_shape: IndexList[a_tma_rank],
    b_tile_shape: IndexList[b_tma_rank],
    c_layout: Layout,
    a_desc_shape: IndexList[a_tma_rank],
    b_desc_shape: IndexList[b_tma_rank],
    block_tile_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](2, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    a_tma_op: TMATensorTile[ab_type, a_tma_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[ab_type, b_tma_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters: UInt,
):
    comptime cta_group = 2
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    # MMA shape: pair-CTA doubles both M and N dimensions.
    comptime MMA_M = 2 * BM
    comptime MMA_N = 2 * BN
    comptime MMA_K = 16  # bf16

    comptime CLUSTER_M = Int(cluster_shape[0])
    comptime CLUSTER_N = Int(cluster_shape[1])

    comptime a_smem_layout = tile_layout_k_major[
        ab_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        ab_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        ab_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var smem = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=8
    ]()

    comptime a_smem_bytes = a_smem_layout.size() * size_of[ab_type]()
    comptime b_smem_bytes = b_smem_layout.size() * size_of[ab_type]()

    var a_smem = smem.bitcast[Scalar[ab_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[ab_type]]()

    var smem_pool = (smem + a_smem_bytes + b_smem_bytes).bitcast[Int64]()

    var a_smem_tile = LayoutTensor[
        ab_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](a_smem)

    var b_smem_tile = LayoutTensor[
        ab_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](b_smem)

    comptime accum_type = get_accum_type[ab_type]()

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (smem_pool.bitcast[Int64]() + 4).bitcast[UInt32]()

    comptime c_frag_size = MMA_M * MMA_N // 128 // cta_group

    comptime a_expected_bytes = a_smem_layout.size() * size_of[ab_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[ab_type]()
    comptime expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    var tma_mbar_ptr = smem_pool.bitcast[Int64]()
    var mma_mbar_ptr = smem_pool.bitcast[Int64]() + 2

    tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = warp_id() == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[Int32(cta_group)](ptr_tmem_addr, max_tmem_cols)

    barrier()

    if elect_one_warp and elect_one_thread:
        tma_mbar[0].init()
        mma_mbar[0].init(
            cluster_shape[0] // Int32(cta_group) + cluster_shape[1] - 1
        )

    cluster_sync()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    tmem_addr = ptr_tmem_addr[0]

    # Build descriptors for the accumulator (MMASmemDescriptorPair)
    comptime a_canonical_layout = tile_to_descriptor[ab_type, a_smem_layout]()
    comptime b_canonical_layout = tile_to_descriptor[
        ab_type, b_smem_layout, is_k_major=transpose_b
    ]()
    comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[ab_type]()
    comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[ab_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[
        ab_type
    ]()
    comptime bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[
        ab_type
    ]()

    adesc_base = MMASmemDescriptorPair.create[aSBO, aLBO, a_swizzle](
        a_smem_tile.ptr
    )
    bdesc_base = MMASmemDescriptorPair.create[bSBO, bLBO, b_swizzle](
        b_smem_tile.ptr
    )

    # The SM100TensorAccumulatorSS handles the k-loop (num_k_mmas) internally
    # via bulk_mma, generating all k-step MMA instructions in one assembly block.
    comptime Acc = SM100TensorAccumulatorSS[
        ab_type,
        accum_type,
        MMA_M,
        MMA_N,
        BK,
        mma_kind=UMMAKind.KIND_F16,
        swizzle_a=a_swizzle,
        swizzle_b=b_swizzle,
        transpose_b=transpose_b,
        cta_group=cta_group,
    ]

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    var peer_cta_coord = (
        rank_m % UInt(cta_group),
        rank_m // UInt(cta_group),
        rank_n,
    )

    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0] if transpose_b else b_desc_shape[1]

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))

    comptime for i in range(CLUSTER_M // cta_group):
        b_multicast_mask |= UInt16(1 << (i * cta_group))

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * UInt(CLUSTER_M))

    var a_mma_mask = a_multicast_mask >> UInt16(peer_cta_coord[0])
    var b_mma_mask = b_multicast_mask >> UInt16(peer_cta_coord[0])
    var c_mma_mask: UInt16 = (a_mma_mask | a_mma_mask << 1) | (
        b_mma_mask | b_mma_mask << 1
    )

    for i in range(Int(num_iters)):
        if elect_one_warp and elect_one_thread:
            if elect_one_cta:
                tma_mbar[0].expect_bytes(Int32(expected_bytes))

            var a_gmem_slice_coord = (
                Int(peer_cta_coord[2]) * a_tma_rows + Int(block_idx.x) * BM
            )
            var b_gmem_slice_coord = (
                Int(peer_cta_coord[1]) * b_tma_rows
                + Int(peer_cta_coord[0]) * BN
                + Int(block_idx.y) * MMA_N
            )

            var a_smem_reshape = a_smem_tile.reshape[Layout.row_major(BM, BK)]()
            var b_smem_reshape = b_smem_tile.reshape[Layout.row_major(BN, BK)]()

            a_tma_op.async_multicast_load[cta_group](
                a_smem_reshape.split[CLUSTER_N]()[peer_cta_coord[2]],
                tma_mbar[0],
                (i * BK, a_gmem_slice_coord),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load[cta_group](
                b_smem_reshape.split[CLUSTER_M // cta_group]()[
                    peer_cta_coord[1]
                ],
                tma_mbar[0],
                (i * BK, b_gmem_slice_coord) if transpose_b else (
                    b_gmem_slice_coord,
                    i * BK,
                ),
                b_multicast_mask,
            )

        if elect_one_cta:
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            if elect_one_warp:
                var elect_val = elect()
                Acc.mma(
                    adesc_base,
                    bdesc_base,
                    tmem_addr,
                    c_scale=UInt32(0) if i == 0 else UInt32(1),
                    elect=elect_val,
                )

                if elect_one_thread:
                    mma_arrive_multicast[cta_group](mma_mbar, c_mma_mask)
        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # Read results from TMEM to global memory
    comptime total_repeat = BN if MMA_M == 128 else MMA_N
    comptime ld_repeat = min(total_repeat, 128)
    comptime num_ld_iters = total_repeat // ld_repeat
    comptime ld_width = c_frag_size // num_ld_iters

    var cluster_idx_m = block_idx.x // UInt(CLUSTER_M)
    var cluster_idx_n = block_idx.y // UInt(CLUSTER_N)
    var global_mma_m = (
        cluster_idx_m * UInt(CLUSTER_M // cta_group) + peer_cta_coord[1]
    )
    var global_mma_n = cluster_idx_n * UInt(CLUSTER_N) + peer_cta_coord[2]

    var c_gmem_block = c.tile[MMA_M, MMA_N](
        Int(global_mma_m), Int(global_mma_n)
    )
    var c_gmem_slice = c_gmem_block.tile[BM, MMA_N](Int(peer_cta_coord[0]), 0)

    comptime for ld_i in range(num_ld_iters):
        var c_frag = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=ld_repeat,
            dtype=accum_type,
            pack=False,
            width=ld_width,
        ](tmem_addr + UInt32(ld_i * ld_repeat))
        tcgen05_load_wait()

        comptime if ld_i == num_ld_iters - 1:
            if elect_one_warp:
                tcgen05_release_allocation_lock[Int32(cta_group)]()
                tcgen05_dealloc[Int32(cta_group)](tmem_addr, max_tmem_cols)

        comptime if MMA_M == 128:
            var c_gmem_frag = c_gmem_slice.tile[BM // 2, BN](
                Int(warp_id() % 2), Int(warp_id() // 2)
            ).vectorize[1, 2]()

            comptime for i in range(ld_width // 2):
                c_gmem_frag[lane_id(), i] = rebind[c_gmem_frag.element_type](
                    SIMD[accum_type, 2](
                        rebind[Scalar[accum_type]](c_frag[2 * i]),
                        rebind[Scalar[accum_type]](c_frag[2 * i + 1]),
                    ).cast[c_type]()
                )
        else:
            var c_gmem_frag = c_gmem_slice.tile[BM // 4, ld_repeat](
                Int(warp_id()), ld_i
            ).vectorize[1, 2]()

            comptime for i in range(ld_width // 2):
                c_gmem_frag[lane_id(), i] = rebind[c_gmem_frag.element_type](
                    SIMD[accum_type, 2](
                        rebind[Scalar[accum_type]](c_frag[2 * i]),
                        rebind[Scalar[accum_type]](c_frag[2 * i + 1]),
                    ).cast[c_type]()
                )


def test_bulk_mma_pair_cta[
    *,
    ab_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](2, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](ctx: DeviceContext) raises:
    comptime cta_group = 2
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = 2 * BM
    comptime MMA_N = 2 * BN

    comptime assert (BN if MMA_M == 128 else MMA_N) <= 256, String(
        "MMA_M = ",
        MMA_M,
        " MMA_N = ",
        MMA_N,
        " BN = ",
        BN,
    )

    print(
        "bulk_mma_pair_cta SS "
        + String(ab_type)
        + " block_tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + " cluster "
        + String(cluster_shape[0])
        + "x"
        + String(cluster_shape[1])
        + " swizzle "
        + String(a_swizzle)
    )

    comptime M = prob_shape[0]
    comptime N = prob_shape[1]
    comptime K = prob_shape[2]

    var a = ManagedLayoutTensor[ab_type, Layout.row_major(M, K)](ctx)
    random(
        a.tensor[update=False](),
        min=min_finite[ab_type](),
        max=max_finite[ab_type](),
    )

    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[ab_type, b_layout](ctx)
    var b_col_major = ManagedLayoutTensor[ab_type, Layout.row_major(N, K)](ctx)
    random(
        b.tensor[update=False](),
        min=min_finite[ab_type](),
        max=max_finite[ab_type](),
    )

    var c = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)
    var c_ref = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)

    a_tma_op = create_tensor_tile[
        Index(Int32(BM) // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a.device_tensor())
    b_tma_op = create_tensor_tile[
        Index(
            Int32(BN) // (cluster_shape[0] // Int32(cta_group)), BK
        ) if transpose_b else Index(
            BK, Int32(BN) // (cluster_shape[0] // Int32(cta_group))
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    comptime smem_size = BM * BK * size_of[ab_type]() + BN * BK * size_of[
        ab_type
    ]() + 16 + 16 + 16

    comptime kernel = bulk_mma_pair_cta_kernel[
        ab_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        Layout.row_major(M, N),
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
        block_tile_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
    ]

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c.device_tensor(),
        UInt(K // BK),
        grid_dim=(
            align_up(M // BM, Int(cluster_shape[0])),
            align_up(N // BN // cta_group, Int(cluster_shape[1])),
            1,
        ),
        block_dim=(128),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )

    comptime if not transpose_b:
        # NOTE: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = b_col_major.tensor()
        var b_tensor = b.tensor()
        for i in range(N):
            for j in range(K):
                b_host_col_major[i, j] = b_tensor[j, i]

        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor[update=False](),
            a.device_tensor[update=False](),
            b_col_major.device_tensor[update=True](),
            c_row_major=True,
            transpose_b=True,
        )
    else:
        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor[update=False](),
            a.device_tensor[update=False](),
            b.device_tensor[update=False](),
            c_row_major=True,
            transpose_b=True,
        )

    ctx.synchronize()

    c_host = c.tensor()
    c_host_ref = c_ref.tensor()
    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                c_host[m, n],
                c_host_ref[m, n],
                atol=0.01,
                rtol=0.01,
                msg=String(m) + ", " + String(n),
            )

    _ = a^
    _ = b^
    _ = b_col_major^
    _ = c^
    _ = c_ref^


def main() raises:
    with DeviceContext() as ctx:
        comptime for swizzle in [
            TensorMapSwizzle.SWIZZLE_32B,
            TensorMapSwizzle.SWIZZLE_64B,
            TensorMapSwizzle.SWIZZLE_128B,
        ]:
            comptime BK = swizzle.bytes() // size_of[DType.bfloat16]()

            comptime for transpose_b in [True, False]:
                # BM=64, BN=128 -> larger N tile, tests wider bulk_mma
                test_bulk_mma_pair_cta[
                    ab_type=DType.bfloat16,
                    c_type=DType.bfloat16,
                    prob_shape=Index(512, 1024, 8 * BK),
                    block_tile_shape=Index(64, 128, BK),
                    transpose_b=transpose_b,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                ](ctx)

                comptime BN_BM64 = 64 if transpose_b else BK

                # BM=64 -> MMA_M=128, cluster_shape=(2,1,1)
                test_bulk_mma_pair_cta[
                    ab_type=DType.bfloat16,
                    c_type=DType.bfloat16,
                    prob_shape=Index(128, 2 * BN_BM64, 2 * BK),
                    block_tile_shape=Index(64, BN_BM64, BK),
                    transpose_b=transpose_b,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                ](ctx)

                # Larger cluster: (2,2,1)
                test_bulk_mma_pair_cta[
                    ab_type=DType.bfloat16,
                    c_type=DType.bfloat16,
                    prob_shape=Index(128, 4 * BN_BM64, 2 * BK),
                    block_tile_shape=Index(64, BN_BM64, BK),
                    transpose_b=transpose_b,
                    cluster_shape=StaticTuple[Int32, 3](2, 2, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                ](ctx)

                comptime BN_BM128 = 64 if transpose_b else BK

                # BM=128 -> MMA_M=256, tests different TMEM read-back path
                test_bulk_mma_pair_cta[
                    ab_type=DType.bfloat16,
                    c_type=DType.bfloat16,
                    prob_shape=Index(256, 2 * BN_BM128, 2 * BK),
                    block_tile_shape=Index(128, BN_BM128, BK),
                    transpose_b=transpose_b,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                ](ctx)

        # ---- Multi-block K tests (BK = 2 * swizzle_width) -------------------
        # These exercise _outer_k_stride != 0 in tile_layout_k_major, matching
        # the depth=512 MHA kernel's Q@K' with BK0=128 and SWIZZLE_128B.
        # Previous tests only had BK == swizzle_width (one block per row).
        comptime for swizzle in [
            TensorMapSwizzle.SWIZZLE_64B,
            TensorMapSwizzle.SWIZZLE_128B,
        ]:
            comptime sw_elems = swizzle.bytes() // size_of[DType.bfloat16]()
            comptime BK2 = 2 * sw_elems  # 2 swizzle blocks per K row

            # Matches depth512 Q@K': BM=64, BN=128, BK=128, transpose_b=True
            test_bulk_mma_pair_cta[
                ab_type=DType.bfloat16,
                c_type=DType.bfloat16,
                prob_shape=Index(128, 256, 2 * BK2),
                block_tile_shape=Index(64, 128, BK2),
                transpose_b=True,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
            ](ctx)

            # Same with transpose_b=False (matches P@V geometry)
            test_bulk_mma_pair_cta[
                ab_type=DType.bfloat16,
                c_type=DType.bfloat16,
                prob_shape=Index(128, 256, 2 * BK2),
                block_tile_shape=Index(64, 128, BK2),
                transpose_b=False,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
            ](ctx)
