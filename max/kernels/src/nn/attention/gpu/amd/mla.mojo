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

from std.sys import align_of, simd_width_of
from std.sys.intrinsics import readfirstlane, _type_is_eq
from std.math.uutils import ufloordiv

from std.gpu import block_idx
from std.gpu import warp_id as get_warp_id
from std.memory import bitcast, stack_allocation
from layout.swizzle import Swizzle
from nn.attention.mha_mask import CausalMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import MHAConfig

from std.utils import IndexList
from std.utils.numerics import get_accum_type

from .attention import AttentionConfig
from .mha import (
    Attention,
    KVBuffer as GFX950KVBuffer,
    MHAAttentionConfig,
    barrier as gfx950_barrier,
    block_sync_lds_direct_load,
)
from .mma import TiledMmaOp


@fieldwise_init
struct MLAAttentionConfig[token_gen: Bool, config: MHAConfig](AttentionConfig):
    # share shared memory for k and v
    # depth=512 with BN=128: separate K+V = 256KB > 160KB LDS limit.
    # Share K/V SMEM (max(K,V) = 128KB + 4KB P = 132KB, fits 1 WG).
    comptime shared_kv = Self.token_gen and Self.config.depth > 256
    # shared memory for the full tile vs BK blocks
    comptime full_kv = True
    # pad the depth for v smem
    comptime depth_padded = False
    # double shared memory for k and v (prefill only; decode uses single buffer)
    comptime double_buffer = not Self.token_gen
    # double buffer K only for decode with small BN (V stays single-buffered).
    # BN=128 uses single-buffer K to fit 2 WGs in MI355's 160KB LDS.
    comptime double_buffer_k_only = (
        Self.token_gen and Self.config.block_n() <= 64
    )

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        return block_idx.y if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].q_head_idx()

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        return Self.q_head_idx() if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].q_tile_idx()

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        return 0 if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].kv_head_idx()

    @staticmethod
    @always_inline
    def get_mma_shape() -> IndexList[3]:
        return MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].get_mma_shape()

    @staticmethod
    @always_inline
    def get_q_offset[q_depth: Int]() -> UInt32:
        return UInt32(
            q_depth
            * (
                block_idx.x
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            ) if not Self.token_gen else q_depth
            * Self.q_tile_idx()
            * Self.config.block_m()
        )

    @staticmethod
    @always_inline
    def get_output_offset[output_depth: Int]() -> UInt32:
        return Self.get_q_offset[output_depth]()


__extension Attention:
    @always_inline
    def mla_prefill_gfx950[
        k_rope_t: MHAOperand,
        //,
    ](mut self, k_rope: k_rope_t):
        """Double-buffered gfx950 MLA prefill with K_rope support.

        Uses gfx950 double-buffered KVBuffer for K, V, and K_rope
        (DRAM→SMEM direct). K_rope has its own SMEM double-buffer.
        Two-phase QK matmul:
        Phase 1 (nope): Q[:,:depth] @ K^T  (gfx950 KVBuffer path)
        Phase 2 (rope): Q[:,depth:q_depth] @ K_rope^T  (gfx950 KVBuffer path)
        """
        comptime cache_num_heads = 1
        comptime cache_depth = 576
        comptime rope_depth = q_depth - Self.depth
        comptime cache_group = Self.num_heads // cache_num_heads

        comptime assert Self.BK == 32 or Self.BK == 64, "BK must be 32 or 64"
        comptime assert Self.depth == 128, "MLA depth must be 128"
        comptime assert (
            Self.depth % Self.BK == 0
        ), "depth must be a multiple of BK"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"
        comptime assert (
            rope_depth % Self.BK == 0
        ), "rope_depth must be a multiple of BK"

        comptime k_swizzle = (
            Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None)
        )

        var warp_id = UInt32(
            readfirstlane(bitcast[DType.int32](UInt32(get_warp_id())))
        )

        # K buffer (nope): depth=128, double-buffered gfx950 style
        var k_buffer = GFX950KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=True,
        ](
            self.k,
            self.batch_idx,
            self.kv_head_idx(),
            self.smem_manager.get_k_ptr[type_of(self.k).dtype](),
            self.num_keys,
            warp_id,
        )

        # V buffer: depth=128, double-buffered gfx950 style
        var v_buffer = GFX950KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=None,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
        ](
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            self.smem_manager.get_v_ptr[type_of(self.v).dtype](),
            self.num_keys,
            warp_id,
        )

        # K_rope buffer: depth=rope_depth, double-buffered gfx950 style.
        # Separate SMEM allocation since K_rope DMA overlaps with K_nope.
        comptime alignment = align_of[
            SIMD[k_rope_t.dtype, simd_width_of[k_rope_t.dtype]()]
        ]()
        comptime k_rope_smem_elems = 2 * Self.BN * rope_depth
        var k_rope_smem_ptr = stack_allocation[
            k_rope_smem_elems,
            k_rope_t.dtype,
            address_space=AddressSpace.SHARED,
            alignment=alignment,
        ]()
        var k_rope_buffer = GFX950KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=rope_depth,
            kv_num_heads=cache_num_heads,
            transpose=True,
            cache_depth=cache_depth,
            head_dim_offset=cache_depth - rope_depth,
        ](
            k_rope,
            self.batch_idx,
            ufloordiv(Int(self.kv_head_idx()), cache_group),
            k_rope_smem_ptr,
            self.num_keys,
            warp_id,
        )

        comptime accum_type = get_accum_type[type_of(self.k).dtype]()

        # Phase 1: Q_nope @ K^T (depth // BK iterations)
        @always_inline
        @parameter
        def mma_qk_nope():
            comptime MmaOp = TiledMmaOp[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]
            self.zero_p_buffer[0]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[i, k_mma](),
                        k_buffer.mma_subtile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        # Phase 2: Q_rope @ K_rope^T (rope_depth // BK iterations)
        @always_inline
        @parameter
        def mma_qk_rope():
            comptime MmaOp = TiledMmaOp[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]
            comptime nope_tiles = Self.depth // Self.BK

            comptime for i in range(rope_depth // Self.BK):
                comptime for k_mma in range(k_rope_buffer.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[nope_tiles + i, k_mma](),
                        k_rope_buffer.mma_subtile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        @always_inline
        @parameter
        def mma_pv():
            comptime PVMmaOp = TiledMmaOp[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]

            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[i, k_mma, 0](),
                        v_buffer.mma_subtile[k_mma, i](),
                        self.out_reg_buffer.reg_tile,
                    )

        # Calculate iteration bounds using mask helpers.
        var score_row = UInt32(self.mask_block_row + UInt32(self.start_pos))
        var start_col = self.mask.start_column[Self.BM, Self.BN, 1](score_row)
        var num_tiles = Int(
            self.mask.last_masked_set_end[Self.BM, Self.BN, 1](
                score_row, UInt32(self.num_keys)
            )
        )

        # Advance KV iterators and mask tracking to start_col.
        k_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        v_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        k_rope_buffer.kv_cache_iter.tile_start_row = (
            Int(start_col) + self.cache_start_pos
        )
        self.kv_start_row = start_col
        self.mask_warp_col += start_col

        var num_pairs = num_tiles // 2

        # Pre-load first tile K+K_rope+V into LDS slot 0.
        # Issue order: K, K_rope, V — so waiting for vmcnt=V.vm
        # ensures K and K_rope are done while V may still be in-flight.
        _ = k_buffer.load_from_dram[0]()
        _ = k_rope_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[0]()

        comptime has_interior_full_mask = not _type_is_eq[
            Self.mask_t, CausalMask
        ]()

        @always_inline
        @parameter
        def process_tile[slot: Int, has_next: Bool]():
            comptime next_slot = 1 - slot

            # Wait for K + K_rope loads to complete for this slot.
            # V was issued last, so vmcnt=V.vm means K and K_rope are done.
            comptime if has_next:
                block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
            else:
                block_sync_lds_direct_load[vmcnt=0]()
            gfx950_barrier[
                schedule_barrier_before=False, schedule_barrier_after=False
            ]()

            # Skip fully masked tiles for non-causal masks.
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        _ = k_buffer.load_from_dram[next_slot]()
                        _ = k_rope_buffer.load_from_dram[next_slot]()
                        _ = v_buffer.load_from_dram[next_slot]()
                        gfx950_barrier[
                            schedule_barrier_before=False,
                            schedule_barrier_after=False,
                        ]()
                    return

            # Load K (nope) from shared → registers and compute Q_nope @ K^T
            k_buffer.load_from_shared(slot)
            mma_qk_nope()

            # Load K_rope from shared → registers and compute Q_rope @ K_rope^T
            k_rope_buffer.load_from_shared(slot)
            mma_qk_rope()

            # Prefetch next K+K_rope+V tile before softmax to hide latency.
            comptime if has_next:
                _ = k_buffer.load_from_dram[next_slot]()
                _ = k_rope_buffer.load_from_dram[next_slot]()
                _ = v_buffer.load_from_dram[next_slot]()

            # Online softmax step 0: scale + mask + max + exp(even)
            self.online_softmax_step_0[0]()
            # Online softmax step 1: exp(odd) + sum + correction + updates
            self.online_softmax_step_1[0]()

            # Wait for V loads from current slot before reading.
            comptime if has_next:
                block_sync_lds_direct_load[
                    vmcnt=k_buffer.vm_instrs_per_load
                    + k_rope_buffer.vm_instrs_per_load
                    + v_buffer.vm_instrs_per_load
                ]()
            gfx950_barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()

            v_buffer.load_from_shared(slot)

            self.online_softmax_update_output()

            mma_pv()

        # Main loop: process tiles in pairs (double-buffered).
        for _ in range(num_pairs):
            process_tile[0, True]()
            process_tile[1, True]()

        # Remainder: last odd tile from slot 0.
        if num_tiles % 2 != 0:
            process_tile[0, False]()

        # Apply final softmax denominator and store
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_output()

    @always_inline
    def mla_decoding(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        self.mha_decoding(exp_sum_ptr, qk_max_ptr, num_partitions)
