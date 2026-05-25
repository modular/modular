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
"""MXFP4 block-scaled matmul on AMD CDNA4 with preshuffled-B + direct VGPR loads.

Variant of `MXFP4MatmulAMD` that skips LDS staging for B. B is preshuffled
host-side via `Shuffler.preshuffle_b_5d` so each lane's 16-byte fragment
lives at a known DRAM offset and is read with a single `buffer_load_dwordx4`.

Only suitable when `num_warps_m == 1` (BM == WM) — otherwise B would be
read multiply across the warps in the M direction without LDS reuse.
"""

from std.math import ceildiv
from std.sys import simd_width_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext

from layout import TensorLayout, TileTensor
from layout.tile_layout import row_major, col_major
from layout.tile_tensor import stack_allocation

from std.utils import IndexList, StaticTuple
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)
from structured_kernels.amd_tile_io import RegTileLoader, RegTileWriter

from .mxfp4_matmul_amd import BlockScaledMmaOp, MX_BLOCK_SIZE
from .mxfp4_preshuffle_loaders import PreshuffledBLoader


struct MXFP4MatmulAMD_PreB[
    BM: Int = 64,
    BN: Int = 128,
    BK_ELEMS: Int = 512,
    WN: Int = 64,
    B_PREFETCH: Bool = False,
]:
    """Preshuffled-B variant of `MXFP4MatmulAMD`.

    The preb path requires `num_warps_m == 1` (no LDS staging for B = no
    cross-warp M-direction B reuse), so `WM` is structurally fixed to `BM`.

    When `B_PREFETCH=True`, runs a depth-2 outer-K software pipeline: while
    the current iter's MFMAs execute, the next iter's B fragments stream
    from DRAM into the alternate b_reg slot. Doubles `_b_reg` size (extra
    VGPRs) but hides DRAM B latency across the inner MFMA chain. Targets
    K-heavy shapes (e.g. gate/up, K=7168) where outer-iter serialization
    dominates.
    """

    # WM is locked to BM — single warp along M for the preb (no-LDS-B) path.
    comptime WM = Self.BM

    comptime BK_BYTES = Self.BK_ELEMS // 2

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 128

    comptime num_warps_m = 1
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps = Self.num_warps_n
    comptime num_threads = Self.num_warps * WARP_SIZE

    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N

    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE
    comptime packed_k_per_mma = Self.MMA_K // 2
    comptime num_k_tiles = Self.BK_BYTES // Self.packed_k_per_mma

    comptime simd_width = simd_width_of[DType.uint8]()
    comptime scales_per_mma = Self.MMA_K // MX_BLOCK_SIZE

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    @staticmethod
    def run[
        out_dtype: DType,
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_pre_layout: TensorLayout,
        sfa_layout: TensorLayout,
        sfb_layout: TensorLayout,
        N: Int,
        K_BYTES: Int,
    ](
        c: TileTensor[out_dtype, c_layout, MutAnyOrigin],
        a: TileTensor[DType.uint8, a_layout, ImmutAnyOrigin],
        b_pre: TileTensor[DType.uint8, b_pre_layout, ImmutAnyOrigin],
        sfa: TileTensor[DType.float8_e8m0fnu, sfa_layout, ImmutAnyOrigin],
        sfb: TileTensor[DType.float8_e8m0fnu, sfb_layout, ImmutAnyOrigin],
        n_tile_idx: Int,
        m_tile_idx: Int,
    ):
        comptime assert (
            K_BYTES % Self.BK_BYTES == 0
        ), "K_BYTES must be a multiple of BK_BYTES"

        comptime BK_BYTES = Self.BK_BYTES
        comptime num_k_tiles = Self.num_k_tiles
        comptime scales_per_mma = Self.scales_per_mma
        comptime K_SCALES = type_of(sfa).static_shape[1]

        var M = Int(a.dim[0]())

        var warp_id = warp_id()
        var warp_m, warp_n = divmod(warp_id, Self.num_warps_n)

        # SMEM for A and scales only — B comes direct from preshuffled DRAM.
        var a_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BM, BK_BYTES]()
        )
        var sfa_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BM, scales_per_mma * num_k_tiles]()
        )
        var sfb_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BN, scales_per_mma * num_k_tiles]()
        )

        var b_loader = PreshuffledBLoader[N=N, K_BYTES=K_BYTES](b_pre)

        comptime load_thread_cols = BK_BYTES // Self.simd_width
        comptime load_thread_rows = Self.num_threads // load_thread_cols
        comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
        comptime a_loads_per_tile = Self.BM // load_thread_rows
        comptime a_reg_elems = Self.BM * BK_BYTES // Self.num_threads

        var a_blockrow = a.tile[Self.BM, K_BYTES](m_tile_idx, 0)

        var a_load_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            row_major[1, a_reg_elems]()
        )

        var a_loader = RegTileLoader[DType.uint8, load_layout](
            a_blockrow,
            bounds_from=a,
        )

        comptime num_b_slots = 2 if Self.B_PREFETCH else 1
        var mma_op = BlockScaledMmaOp[
            mma_shape=IndexList[3](Self.MMA_M, Self.MMA_N, Self.MMA_K),
            num_m_mmas=Self.num_m_mmas,
            num_n_mmas=Self.num_n_mmas,
            num_k_tiles=num_k_tiles,
            num_b_slots=num_b_slots,
        ]()

        var c_writer = RegTileWriter[
            out_dtype, Self.MMA_M, WARP_SIZE // Self.MMA_M
        ](c)

        var k_counter = 0
        var k_scale_counter = 0
        var warp_n_off_global = n_tile_idx * Self.BN + warp_n * Self.WN

        @always_inline
        @parameter
        def load_a_tile_from_dram():
            var a_block = a_blockrow.tile[Self.BM, BK_BYTES](0, k_counter)
            a_loader.load(a_load_reg, a_block.vectorize[1, Self.simd_width]())
            k_counter += 1

        @always_inline
        @parameter
        def copy_a_tile_to_smem():
            var a_smem_dist = a_smem.vectorize[1, Self.simd_width]().distribute[
                load_layout
            ](thread_idx.x)
            comptime for v in range(a_loads_per_tile):
                a_smem_dist[v, 0] = a_load_reg.raw_load[width=Self.simd_width](
                    v * Self.simd_width
                )

        @always_inline
        @parameter
        def load_scales_to_smem():
            var tid = Int(thread_idx.x)
            var base_scale_k = k_scale_counter * scales_per_mma * num_k_tiles
            var a_base_row = m_tile_idx * Self.BM
            var b_base_row = n_tile_idx * Self.BN

            if tid < Self.BM:
                var row = a_base_row + tid
                if row < M:
                    comptime for k in range(num_k_tiles):
                        var src_off = (
                            row * K_SCALES + base_scale_k + k * scales_per_mma
                        )
                        var src_word = sfa.ptr.bitcast[Scalar[DType.int32]]()[
                            src_off // scales_per_mma
                        ]
                        sfa_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = src_word
                else:
                    comptime for k in range(num_k_tiles):
                        sfa_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = Int32(0)
            if tid < Self.BN:
                var row = b_base_row + tid
                if row < N:
                    comptime for k in range(num_k_tiles):
                        var src_off = (
                            row * K_SCALES + base_scale_k + k * scales_per_mma
                        )
                        var src_word = sfb.ptr.bitcast[Scalar[DType.int32]]()[
                            src_off // scales_per_mma
                        ]
                        sfb_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = src_word
                else:
                    comptime for k in range(num_k_tiles):
                        sfb_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = Int32(0)

            k_scale_counter += 1

        comptime num_outer = K_BYTES // BK_BYTES

        # TODO use comptime pipeline scheduler

        comptime if Self.B_PREFETCH:
            # Depth-2 outer-K software pipeline.
            #
            # Prologue: load A + scales (smem) and all B fragments (slot 0)
            # for iter 0.
            load_a_tile_from_dram()
            load_scales_to_smem()
            copy_a_tile_to_smem()
            comptime for k in range(num_k_tiles):
                mma_op.load_b_frag_preshuffled[k, N, K_BYTES, slot=0](
                    b_loader, warp_n_off_global, 0
                )
            barrier()

            # Steady state: for each i in [0, num_outer-1), MFMA iter i from
            # `cur_slot` while prefetching iter i+1's B into `nxt_slot`. A
            # and scales smem is refilled before the barrier so iter i+1's
            # MFMAs can read it next pass.
            comptime for i in range(num_outer - 1):
                comptime cur_slot = i % 2
                comptime nxt_slot = (i + 1) % 2
                var nxt_k_byte_base = (i + 1) * BK_BYTES

                comptime for k in range(num_k_tiles):
                    mma_op.load_b_frag_preshuffled[
                        k, N, K_BYTES, slot=nxt_slot
                    ](b_loader, warp_n_off_global, nxt_k_byte_base)

                var a_warp = a_smem.tile[Self.WM, BK_BYTES](warp_m, 0)
                comptime for k in range(num_k_tiles):
                    mma_op.load_a_frag_from_smem[k](a_warp)
                    var sfa_k = sfa_smem.tile[Self.WM, scales_per_mma](
                        warp_m, k
                    )
                    var sfb_k = sfb_smem.tile[Self.WN, scales_per_mma](
                        warp_n, k
                    )
                    mma_op.load_scales_from_smem[k](sfa_k, sfb_k)
                    mma_op.mma[k, slot=cur_slot]()

                load_a_tile_from_dram()
                barrier()
                load_scales_to_smem()
                copy_a_tile_to_smem()
                barrier()

            # Epilogue: MFMA the last iter from its slot.
            comptime last_slot = (num_outer - 1) % 2
            var a_warp = a_smem.tile[Self.WM, BK_BYTES](warp_m, 0)
            comptime for k in range(num_k_tiles):
                mma_op.load_a_frag_from_smem[k](a_warp)
                var sfa_k = sfa_smem.tile[Self.WM, scales_per_mma](warp_m, k)
                var sfb_k = sfb_smem.tile[Self.WN, scales_per_mma](warp_n, k)
                mma_op.load_scales_from_smem[k](sfa_k, sfb_k)
                mma_op.mma[k, slot=last_slot]()
            barrier()
        else:
            for k_iter in range(num_outer):
                load_a_tile_from_dram()
                load_scales_to_smem()
                copy_a_tile_to_smem()
                barrier()

                var a_warp = a_smem.tile[Self.WM, BK_BYTES](warp_m, 0)
                var k_byte_base = k_iter * BK_BYTES

                comptime for k in range(num_k_tiles):
                    mma_op.load_a_frag_from_smem[k](a_warp)
                    mma_op.load_b_frag_preshuffled[k, N, K_BYTES](
                        b_loader, warp_n_off_global, k_byte_base
                    )

                    var sfa_k = sfa_smem.tile[Self.WM, scales_per_mma](
                        warp_m, k
                    )
                    var sfb_k = sfb_smem.tile[Self.WN, scales_per_mma](
                        warp_n, k
                    )
                    mma_op.load_scales_from_smem[k](sfa_k, sfb_k)

                    mma_op.mma[k]()
                barrier()

        var c_reg = mma_op.accum_tile()
        var c_block = c.tile[Self.BM, Self.BN](m_tile_idx, n_tile_idx)
        var c_warp = c_block.tile[Self.WM, Self.WN](warp_m, warp_n)

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                c_writer.store(
                    c_warp.tile[Self.MMA_M, Self.MMA_N](m_mma, n_mma).vectorize[
                        1, Self.c_frag_size
                    ](),
                    c_reg.tile[1, Self.c_frag_size](m_mma, n_mma),
                )
