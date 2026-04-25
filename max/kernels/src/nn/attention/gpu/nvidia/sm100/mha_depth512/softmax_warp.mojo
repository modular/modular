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
"""Softmax warp group logic for depth=256/512 pair-CTA SM100 attention.

Computes online softmax over Q@K' scores (S in TMEM) and writes the
exponentiated result P to SMEM for SS MMA P@V consumption. Unlike FA4
where P lives in TMEM for TS MMA, this kernel must explicitly transfer
P from registers to SMEM with the correct swizzle layout.

Pair-CTA TMEM column layout (cta_group=2):
    For MMA output [BM, MMA_N]:
      Columns 0 : MMA_N//2       → TMEM rows 0..BM-1
      Columns MMA_N//2 : MMA_N   → TMEM rows BM..MMA_M-1

Depth-dependent behavior:
  split_o=True (d512, MMA_M=128, BM=64): Each M row is served by a thread
    pair (row m and row m+64). exchange_reduce combines cross-thread
    row_max/row_sum.
  split_o=False (d256, MMA_M=256, BM=128): Each thread covers a unique
    M-row with full BN columns. No exchange needed.
"""

from std.math import exp2, recip
from std.math.constants import log2e
from std.memory import bitcast
from std.sys import size_of
import std.gpu.primitives.warp as warp
from std.gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from std.gpu.memory import AddressSpace, fence_async_view_proxy
from std.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    umma_arrive_leader_cta,
)
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from std.gpu.primitives.warp import _vote_nvidia_helper
from std.gpu.primitives.cluster import block_rank_in_cluster
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAddress,
)
from layout import IntTuple
from layout.swizzle import make_swizzle
from layout.tensor_core_async import tile_layout_k_major
from layout.tma_async import RaggedTMA3DTile, SharedMemBarrier
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    SharedMemPointer,
    MBarType,
    TMemTile,
    llvm_opaque_tid,
    add_ftz,
    sub_ftz,
    mul_ftz,
    fma_ftz,
    max_ftz,
    maximum,
    apply_mask,
    peel_mask,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem


# Named barrier ID for softmax warp group cross-thread exchange.
# Must not conflict with barrier IDs used by other warp groups.
comptime _SOFTMAX_EXCHANGE_BARRIER: Int32 = 0


@always_inline
def depth512_scale_write_output[
    output_type: DType,
    qkv_dtype: DType,
    config: Depth512SM100Config[qkv_dtype],
](
    tid: UInt32,
    m_row: UInt32,
    is_lower: Bool,
    inv_row_sum: Float32,
    smem: Depth512AttentionSMem[config=config],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        config.swizzle_mode,
        BM=config.BM,
        BN=config.ov_depth,
        group=config.group if config.fuse_gqa else 1,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """Read O from TMEM, scale by inv_row_sum, write to SMEM, TMA store.

    split_o (d512): Two phases (O_lo, O_hi). Each thread processes ov_depth/4
      physical TMEM cols per phase, with is_lower determining output col base.
    !split_o (d256): Single phase. Each thread processes MMA_M*ov_depth/256
      physical TMEM cols. All threads write to col base 0.
    """
    comptime accum_dtype = DType.float32
    comptime BM = config.BM
    # Physical TMEM cols per O phase.
    comptime o_cols_per_phase = (config.ov_depth // 4) if config.split_o else (
        config.MMA_M * config.ov_depth // 256
    )
    comptime batch_size = 16
    comptime num_batches = o_cols_per_phase // batch_size
    comptime assert o_cols_per_phase % batch_size == 0

    var tmem_addr = smem.tmem_addr_ptr()[]

    # Output SMEM base (reuses Q buffer).
    var o_smem = smem.o_smem[output_type]()
    # O SMEM must match tile_layout_k_major[BM, ov_depth] for TMA store.
    # Decompose col into k-block + inner offset, swizzle only the inner part.
    comptime o_swizzle = make_swizzle[output_type, config.swizzle_mode]()
    comptime o_sw_K = config.swizzle_mode.bytes() // size_of[output_type]()

    # ---- Helper: load from TMEM, scale, write to SMEM --------------------
    @parameter
    @always_inline
    def read_scale_write(
        o_tmem: TmemAddress,
        col_base: Int,
    ):
        comptime for b in range(num_batches):
            comptime col_offset = b * batch_size
            var o_vals = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                dtype=accum_dtype,
                pack=False,
                width=batch_size,
            ]((o_tmem + col_offset).addr)

            # Scale and cast in groups of 8, write to swizzled SMEM.
            comptime for g in range(batch_size // 8):
                comptime base = g * 8
                var vals: SIMD[accum_dtype, 8] = {
                    o_vals[base],
                    o_vals[base + 1],
                    o_vals[base + 2],
                    o_vals[base + 3],
                    o_vals[base + 4],
                    o_vals[base + 5],
                    o_vals[base + 6],
                    o_vals[base + 7],
                }
                vals = vals * inv_row_sum

                var col = col_base + col_offset + base
                var o_k_block = col // o_sw_K
                var o_inner = Int(m_row) * o_sw_K + col % o_sw_K
                (o_smem + o_k_block * BM * o_sw_K + o_swizzle(o_inner)).bitcast[
                    Scalar[DType.uint32]
                ]().store(bitcast[DType.uint32, 4](vals.cast[output_type]()))

    comptime if config.split_o:
        comptime ov_quarter = config.ov_depth // 4
        comptime ov_half = config.ov_depth // 2
        var o_lo_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O))
        var o_hi_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O_hi))
        # Lower threads write first half cols, upper threads write second half.
        var col_base_lo: Int = 0 if is_lower else ov_quarter
        var col_base_hi: Int = ov_half if is_lower else (ov_half + ov_quarter)
        read_scale_write(o_lo_tmem, col_base_lo)
        read_scale_write(o_hi_tmem, col_base_hi)
    else:
        var o_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O))
        read_scale_write(o_tmem, 0)

    # Sync all 128 softmax threads before TMA store.
    # Reuse barrier ID 0 (safe: softmax exchange loop is complete).
    named_barrier[Int32(WARPGROUP_SIZE)](_SOFTMAX_EXCHANGE_BARRIER)

    # TMA store: one elected thread issues all column-chunk stores.
    var e = elect()
    var local_warp_idx = (tid // UInt32(WARP_SIZE)) % 4

    if local_warp_idx == 0:
        if e != 0:
            fence_async_view_proxy()
        comptime swizzle_granularity = config.swizzle_mode.bytes() // size_of[
            output_type
        ]()
        comptime num_col_chunks = config.ov_depth // swizzle_granularity
        comptime for col in range(num_col_chunks):
            if e != 0:
                ragged_tma_store.async_copy_from_col[col](
                    o_smem,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                )
        if e != 0:
            cp_async_bulk_commit_group()

    # Wait for all TMA stores to complete.
    cp_async_bulk_wait_group[0]()


@always_inline
def depth512_softmax[
    MaskType: MHAMask,
    qkv_dtype: DType,
    output_type: DType,
    config: Depth512SM100Config[qkv_dtype],
    page_size: Int,
](
    smem: Depth512AttentionSMem[config=config],
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
    scale: Float32,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        config.swizzle_mode,
        BM=config.BM,
        BN=config.ov_depth,
        group=config.group if config.fuse_gqa else 1,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    comptime accum_dtype = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    # Per-thread S columns: split_o halves the columns (exchange_reduce combines),
    # !split_o each thread sees the full BN (both happen to be 128 currently).
    comptime effective_bn = BN // 2 if config.split_o else BN
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    comptime BM_eff: Int = config.BM_eff()
    comptime PairBM_mask = BM_eff * 2
    comptime f32x2 = SIMD[DType.float32, 2]

    # Batch size for pipelined TMEM loads and exp computation.
    comptime batch_size = 32
    comptime has_remainder = (effective_bn % batch_size) != 0
    comptime first_cols = (
        effective_bn % batch_size
    ) if has_remainder else batch_size

    comptime max_unroll = 8

    # ---- Thread identity -------------------------------------------------
    var tid = llvm_opaque_tid()
    var row = tid % 128  # TMEM row (0-127)
    var m_row = row % UInt32(BM)  # M row index
    # split_o (d512): is_lower distinguishes paired threads (row<64 vs >=64)
    # !split_o (d256): BM=128, so is_lower is always True (all threads unique)
    var is_lower = row < UInt32(BM)

    var cta_rank = block_rank_in_cluster() % 2
    var per_thread_score_row: UInt32
    comptime if fuse_gqa:
        per_thread_score_row = (
            score_row
            + UInt32(cta_rank) * UInt32(BM_eff)
            + m_row // UInt32(group)
        )
    else:
        per_thread_score_row = score_row + cta_rank * UInt32(BM) + m_row

    # Column offset into the full BN score tile.
    # split_o: lower→0, upper→effective_bn. !split_o: always 0.
    var col_offset: UInt32 = 0 if is_lower else UInt32(effective_bn)

    # ---- TMEM addresses --------------------------------------------------
    var tmem_addr = smem.tmem_addr_ptr()[]
    var s_even_tmem = tmem_addr + UInt32(config.TMEM_S_even)
    var s_odd_tmem = tmem_addr + UInt32(config.TMEM_S_odd)

    # ---- Scale -----------------------------------------------------------
    var scale_log2e: Scalar[accum_dtype] = scale * log2e

    # ---- Barriers --------------------------------------------------------
    var mbars = Depth512MBars[config.num_kv_stages, config.split_o](
        smem.mbar_base()
    )
    var pipeline_s_even = mbars.consumer_s_even()
    var pipeline_s_odd = mbars.consumer_s_odd()
    var pipeline_c = mbars.producer_c()
    var po_lo = mbars.po_lo_mbar()

    # ---- SMEM pointers ---------------------------------------------------
    var p_smem = smem.p_smem()
    var correction_smem = smem.correction_smem()
    # P SMEM must match tile_layout_k_major[BM, BN] for the MMA descriptor.
    # The layout is hierarchical: BN/sw_K outer blocks of [BM, sw_K] each.
    # Decompose col into k_block + inner, swizzle only the inner part.
    comptime p_swizzle = make_swizzle[qkv_dtype, config.swizzle_mode]()
    comptime p_sw_K = config.swizzle_mode.bytes() // size_of[qkv_dtype]()

    # ---- S register buffer -----------------------------------------------
    # Holds effective_bn f32 values per thread (128 for both d256 and d512).
    var s = InlineArray[Scalar[accum_dtype], effective_bn](uninitialized=True)

    # ---- Iteration bounds (must match MMA and load warps) ----------------
    var kv_row: UInt32 = mask.start_column[PairBM_mask, BN, page_size](
        score_row
    )

    comptime mask_sets = MaskType.nonfull_sets[PairBM_mask, BN]()
    comptime mask_strategies = MaskType.mask_strategies[PairBM_mask, BN]()
    comptime num_sets = len(mask_sets)

    var mask_iters: StaticTuple[UInt32, num_sets] = {}

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        mask_ends = mask.masked_set_ends[
            BM=PairBM_mask, BN=BN, page_size=page_size
        ](score_row, num_keys)
        mask_iters[0] = mask_ends[0]
        comptime for i in range(1, num_sets):
            mask_iters[i] = mask_ends[i] - mask_ends[i - 1]

    comptime assert num_sets >= 1 and num_sets <= 3

    # ---- Inner helpers ---------------------------------------------------

    @parameter
    @always_inline
    def s_load[i: Int]() -> f32x2:
        return f32x2(s[2 * i], s[2 * i + 1])

    @parameter
    @always_inline
    def s_store[i: Int](v: f32x2):
        s[2 * i] = v[0]
        s[2 * i + 1] = v[1]

    @parameter
    @always_inline
    def mask_batch[
        N: Int, //, mask_strategy: MaskStrategy
    ](mut batch: InlineArray[Scalar[accum_dtype], N], kv_col: UInt32):
        """Apply mask to a batch of score elements."""
        apply_mask[
            mask_strategy=mask_strategy,
            skip_scale=True,
        ](
            batch,
            mask,
            scale_log2e,
            prompt_idx=UInt32(0),
            q_head_idx=UInt32(0),
            kv_tile_start_row=Int32(kv_col),
            max_seq_len=num_keys,
            num_keys=Int32(num_keys),
            score_row=Int32(per_thread_score_row),
        )

    @parameter
    @always_inline
    def exchange_reduce[
        op: StringLiteral,  # "max" or "add"
    ](partial_val: Float32) -> Float32:
        """Exchange partial value between paired threads via correction_smem.

        Uses 2 named_barrier syncs. correction_smem must be free (ensured
        by calling pipeline_c.acquire() before this function).
        """
        # Step 1: Lower half writes its partial value.
        if is_lower:
            correction_smem[m_row] = partial_val
        named_barrier[Int32(WARPGROUP_SIZE)](_SOFTMAX_EXCHANGE_BARRIER)

        # Step 2: Upper half reads partner, computes combined, writes back.
        var combined: Float32
        if not is_lower:
            partner_val = correction_smem[m_row]
            comptime if op == "max":
                combined = max_ftz(partial_val, partner_val)
            else:
                combined = partial_val + partner_val
            correction_smem[m_row] = combined
        else:
            combined = partial_val
        named_barrier[Int32(WARPGROUP_SIZE)](_SOFTMAX_EXCHANGE_BARRIER)

        # Step 3: Lower half reads combined result.
        if is_lower:
            combined = correction_smem[m_row]
        return combined

    # ---- load_mask_max: pipelined S load + mask + max --------------------
    # Follows FA4 pattern: double-buffer TMEM loads across batches so that
    # masking + max of batch N overlaps with the TMEM load of batch N+1.

    @parameter
    @always_inline
    def load_mask_max_impl[
        *, mask_strategy: MaskStrategy
    ](s_tmem: UInt32, kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
        """Load effective_bn columns of S from TMEM, apply mask, compute partial
        row_max as a StaticTuple for reduction.

        Each warp pair loads from the correct TMEM address range:
        - Lower (warps 0-1): columns 0..effective_bn-1 at s_tmem
        - Upper (warps 2-3): columns effective_bn..BN-1 at s_tmem + effective_bn
        """
        # Base TMEM address: both row groups read the same physical columns.
        # Pair-CTA layout maps rows 0-63 → first logical N half, rows 64-127 →
        # second logical N half, but both use the same MMA_N/2 physical cols.
        base_tmem = s_tmem
        # KV column base for masking.
        kv_col_base = kv_row + col_offset

        # --- Pipelined load: load batch 0, start batch 1, process batch 0 ---
        s0 = TMemTile[accum_dtype, BM, first_cols](base_tmem).load_async()

        s1 = TMemTile[accum_dtype, BM, batch_size](
            base_tmem + UInt32(first_cols)
        ).load_async()

        mask_batch[mask_strategy=mask_strategy](s0, kv_col_base)
        vrow_max = maximum[width=max_unroll](s0)
        comptime for _i in range(first_cols):
            s[_i] = s0[_i]

        comptime cols = effective_bn - first_cols + batch_size

        comptime for i in range(cols // (2 * batch_size)):
            comptime offset0 = first_cols + batch_size * (2 * i)
            comptime offset1 = first_cols + batch_size * (2 * i + 1)
            comptime offset2 = first_cols + batch_size * (2 * i + 2)

            comptime if offset1 >= effective_bn:
                # Last batch: s1 is already loaded, just process it.
                mask_batch[mask_strategy=mask_strategy](
                    s1, kv_col_base + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)
                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]
            else:
                # Load next batch (s2) while processing current (s1).
                s2 = TMemTile[accum_dtype, BM, batch_size](
                    base_tmem + UInt32(offset1)
                ).load_async()
                mask_batch[mask_strategy=mask_strategy](
                    s1, kv_col_base + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)
                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]

                comptime if offset2 < effective_bn:
                    s1 = TMemTile[accum_dtype, BM, batch_size](
                        base_tmem + UInt32(offset2)
                    ).load_async()
                mask_batch[mask_strategy=mask_strategy](
                    s2, kv_col_base + UInt32(offset1)
                )
                vrow_max = maximum(s2, vrow_max)
                comptime for _i in range(batch_size):
                    s[offset1 + _i] = s2[_i]

        return vrow_max

    @parameter
    @always_inline
    def init_load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> Float32:
        """Load S, mask, return partial max (scalar)."""
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](s_even_tmem, kv_row)
        )

    @parameter
    @always_inline
    def load_mask_max[
        mask_strategy: MaskStrategy
    ](s_tmem: UInt32, kv_row: UInt32, old_max: Float32) -> Float32:
        """Load S, mask, return partial max combined with old_max."""
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](s_tmem, kv_row),
            old_max,
        )

    # ---- store_exp: compute exp, write P to SMEM, return row_sum ---------
    # Follows FA4 pattern: interleave score_to_logit ahead of exp2 via
    # score_to_logit_ratio, then write P to SMEM in batches.

    @parameter
    @always_inline
    def store_exp(row_max: Float32) -> f32x2:
        comptime exp_simd = 2
        comptime vs_len = effective_bn // exp_simd
        comptime score_to_logit_ratio: Int = 4

        var vscale = f32x2(scale_log2e)
        var vneg_max_scaled = f32x2(-row_max * scale_log2e)

        @parameter
        @always_inline
        def score_to_logit(score: f32x2) -> f32x2:
            return fma_ftz(score, vscale, vneg_max_scaled)

        # Interleaved exp: score_to_logit runs ahead by score_to_logit_ratio.
        @parameter
        @always_inline
        def exp_iter[idx: Int]():
            comptime if idx < vs_len // score_to_logit_ratio:
                comptime for i in range(score_to_logit_ratio):
                    comptime j = score_to_logit_ratio * idx + i
                    s_store[j](score_to_logit(s_load[j]()))
            s_store[idx](exp2(s_load[idx]()))

        # --- Process in batches, write P to SMEM after each ---
        comptime p_batch = vs_len // 4 if vs_len >= 128 else vs_len // 2
        comptime p_batch_elems = p_batch * exp_simd
        comptime num_p_batches = vs_len // p_batch
        comptime p_remainder = vs_len % p_batch
        comptime assert num_p_batches >= 1

        # Helper to write a range of exp values from s[] to P SMEM.
        @parameter
        @always_inline
        def write_p_batch[start_elem: Int, num_elems: Int]():
            comptime for c in range(0, num_elems, 8):
                comptime base = start_elem + c
                var vals = SIMD[accum_dtype, 8](
                    s[base],
                    s[base + 1],
                    s[base + 2],
                    s[base + 3],
                    s[base + 4],
                    s[base + 5],
                    s[base + 6],
                    s[base + 7],
                ).cast[qkv_dtype]()
                var col = Int(col_offset) + base
                var p_k_block = col // p_sw_K
                comptime assert effective_bn % p_sw_K == 0
                comptime r = base % p_sw_K
                var p_inner = Int(m_row) * p_sw_K + r
                (p_smem + p_k_block * BM * p_sw_K + p_swizzle(p_inner)).bitcast[
                    Scalar[DType.uint32]
                ]().store(bitcast[DType.uint32, 4](vals))

        # Batch 0: compute exp.
        comptime for idx in range(p_batch):
            exp_iter[idx]()
        write_p_batch[0, p_batch_elems]()

        # Remaining batches: compute exp, then write to P SMEM.
        comptime for b in range(1, num_p_batches):
            comptime offset = p_batch * b
            comptime el_offset = offset * exp_simd
            comptime for idx in range(offset, offset + p_batch):
                exp_iter[idx]()
            write_p_batch[el_offset, p_batch_elems]()

        comptime if p_remainder > 0:
            comptime offset = p_batch * num_p_batches
            comptime el_offset = offset * exp_simd
            comptime for idx in range(offset, offset + p_remainder):
                exp_iter[idx]()
            write_p_batch[el_offset, p_remainder * exp_simd]()

        # P is fully written to SMEM. Fence + signal PO_lo so the MMA warp
        # can start P@V as early as possible (before the row-sum reduction).
        # Cluster-scope arrive so both CTAs' signals reach the leader.
        fence_async_view_proxy()
        umma_arrive_leader_cta(po_lo)

        # Row sum: 4-way unrolled accumulation over exp values in s.
        var acc0 = s_load[0]()
        var acc1 = s_load[1]()
        var acc2 = s_load[2]()
        var acc3 = s_load[3]()
        comptime for i in range(4, vs_len, 4):
            acc0 = add_ftz(acc0, s_load[i]())
            acc1 = add_ftz(acc1, s_load[i + 1]())
            acc2 = add_ftz(acc2, s_load[i + 2]())
            acc3 = add_ftz(acc3, s_load[i + 3]())
        return add_ftz(add_ftz(acc0, acc1), add_ftz(acc2, acc3))

    # ---- Peeled first iteration ------------------------------------------

    pipeline_s_even.wait()
    tcgen05_fence_after()

    var partial_max: Float32 = peel_mask[
        rebind[StaticTuple[MaskStrategy, num_sets]](mask_strategies),
        init_load_mask_max,
    ](mask_iters, kv_row)

    umma_arrive_leader_cta(pipeline_s_even.consumer_mbar())
    pipeline_s_even.step()

    # pipeline_c.acquire() passes immediately on first use (buffer free).
    pipeline_c.acquire()
    # split_o: exchange_reduce combines paired threads' partial values.
    # !split_o: each thread has the full row, no exchange needed.
    var row_max: Float32
    comptime if config.split_o:
        row_max = exchange_reduce["max"](partial_max)
    else:
        row_max = partial_max

    # Compute exp, write P to SMEM (signals PO_lo inside), get partial sum.
    var partial_sum = store_exp(row_max)
    var global_sum: Float32
    comptime if config.split_o:
        global_sum = exchange_reduce["add"](partial_sum.reduce_add())
    else:
        global_sum = partial_sum.reduce_add()
    var row_sum = f32x2(global_sum, 0)

    # ---- Main loop (alternating S_even / S_odd) --------------------------

    var o_phase: UInt32 = 0

    var s_cur_pipeline = pipeline_s_odd
    var s_cur_tmem = s_odd_tmem
    var s_nxt_pipeline = pipeline_s_even
    var s_nxt_tmem = s_even_tmem

    comptime rescale_threshold: Float32 = Float32(-8) if size_of[
        qkv_dtype
    ]() >= 2 else Float32(0)

    @parameter
    @always_inline
    def main_loop_body[mask_strategy: MaskStrategy]():
        """One iteration of the main softmax loop."""
        old_max = row_max

        # Wait for S, load, mask, compute partial max.
        s_cur_pipeline.wait()
        tcgen05_fence_after()
        partial_max = load_mask_max[mask_strategy](s_cur_tmem, kv_row, old_max)
        umma_arrive_leader_cta(s_cur_pipeline.consumer_mbar())
        s_cur_pipeline.step()

        # Exchange max (correction_smem free after acquire).
        pipeline_c.acquire()
        var new_row_max: Float32
        comptime if config.split_o:
            new_row_max = exchange_reduce["max"](partial_max)
        else:
            new_row_max = partial_max

        diff = sub_ftz(old_max, new_row_max)
        diff = mul_ftz(diff, scale_log2e)

        var correction: Float32
        comptime if rescale_threshold < 0:
            if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                row_max = new_row_max
                correction = exp2(diff)
            else:
                correction = 1
        else:
            row_max = new_row_max
            correction = exp2(diff)

        # Compute exp, write P (signals PO_lo inside), exchange sum.
        partial_sum = store_exp(row_max)
        var local_sum: Float32
        comptime if config.split_o:
            local_sum = exchange_reduce["add"](partial_sum.reduce_add())
        else:
            local_sum = partial_sum.reduce_add()

        # Write correction factor for the correction warp.
        # split_o: only lower half writes to avoid double-write (paired threads).
        # !split_o: all threads have unique m_rows.
        comptime if config.split_o:
            if is_lower:
                correction_smem[m_row] = correction
        else:
            correction_smem[m_row] = correction
        pipeline_c.commit()

        row_sum = fma_ftz(row_sum, f32x2(correction), f32x2(local_sum, 0))
        o_phase ^= 1

        # Swap S buffers.
        var tmp_pipeline = s_cur_pipeline
        s_cur_pipeline = s_nxt_pipeline
        s_nxt_pipeline = tmp_pipeline
        var tmp_tmem = s_cur_tmem
        s_cur_tmem = s_nxt_tmem
        s_nxt_tmem = tmp_tmem

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        comptime for i in range(num_sets):
            comptime mask_strategy = mask_strategies[i]
            var iters: UInt32
            iters = warp.broadcast(mask_iters[i])
            while iters != 0:
                iters -= 1
                kv_row += UInt32(BN)
                main_loop_body[mask_strategy]()
    else:
        while True:
            kv_row += UInt32(BN)
            if kv_row >= num_keys:
                break
            cur_mask_status = mask.status(
                Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                Index[dtype=DType.int32](PairBM_mask, BN),
            )
            if cur_mask_status == TileMaskStatus.FULL_MASK:
                continue
            if cur_mask_status == TileMaskStatus.PARTIAL_MASK:
                main_loop_body[
                    MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS
                ]()
            else:
                main_loop_body[MaskStrategy.OUT_OF_BOUNDS]()

    # ---- Post-loop: wait for final O and write output --------------------

    # Wait for the last P@V to complete.
    # split_o: O_mma_hi fires after O_mma_lo, so waiting O_mma_hi ensures both.
    # !split_o: only O_mma_lo exists.
    comptime o_mma_done_offset = Depth512MBars[
        config.num_kv_stages, config.split_o
    ].O_mma_hi_offset if config.split_o else Depth512MBars[
        config.num_kv_stages, config.split_o
    ].O_mma_lo_offset
    var o_mma_done_mbar: MBarType = mbars.mbar_base + o_mma_done_offset
    o_mma_done_mbar[].wait(o_phase)
    tcgen05_fence_after()

    # Final scaling: inv_row_sum = 1 / total_row_sum.
    var inv_row_sum = recip(row_sum.reduce_add())

    # Scale O from TMEM by inv_row_sum, write to SMEM, TMA store to global.
    if num_output_rows > 0:
        depth512_scale_write_output[output_type, qkv_dtype, config](
            tid,
            m_row,
            is_lower,
            inv_row_sum,
            smem,
            ragged_tma_store,
            num_output_rows,
            out_head_idx,
            out_row_idx,
        )

    # TMEM deallocation: all other warps (correction, MMA, load) are done
    # with TMEM by this point. Only warp 0 needs to deallocate.
    if tid // UInt32(WARP_SIZE) == 0:
        tcgen05_release_allocation_lock[Int32(config.cta_group)]()
        tcgen05_dealloc[Int32(config.cta_group)](tmem_addr, UInt32(512))
