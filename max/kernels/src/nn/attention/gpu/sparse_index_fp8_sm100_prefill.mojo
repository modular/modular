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
"""SM100 (B200) warp-specialized PREFILL variant of the FP8 MLA indexer scorer.

Computes the identical per-(query token, key) logit as the shipped scorer
(`sparse_index_fp8_sm100.fp8_index_score_sm100`) and the scalar
`nn.index_fp8.fp8_index_kernel`:

    score[token, key] = k_scale[key]
                        * Σ_head relu(q[token, head] · k[key]) * q_scale[token, head]

The shipped kernel is K-resident / Q-streaming: one CTA holds a `BM_key`-key tile
and streams every query token past it. That maps a batch-1 prefill onto only
`num_keys / BM_key` CTAs and runs one serial warpgroup, so it is latency-bound
(measured ~6% achieved occupancy: one active warp per scheduler, no
MMA↔epilogue overlap).

This kernel INVERTS which operand persists:

- **Q resident** as the B operand `[MMA_N = N_TOKENS * num_heads, depth]`: a CTA
  owns one N_TOKENS-token block, staged once.
- **K streams** as the A operand `[BM_key = 128, depth]` through a deep SMEM
  prefetch ring, `S^T = K @ Q^T = [key, (token, head)]`, so the epilogue reduces
  over the (token, head) COLUMNS exactly like the shipped kernel (heads stay
  columns; all head counts in {4, 8, 32, 64} work uniformly, no cross-warp
  reduction).
- Grid `(batch, ceil(seq_len / N_TOKENS), num_key_parts)`: one CTA per
  (query-token block, key part). A batch-1 GLM prefill (1024 tokens,
  num_heads=32, N_TOKENS=4) is 256 CTAs on the first two axes alone, so it runs
  unsplit at `num_key_parts == 1`. Decode/MTP inverts that -- a handful of token
  blocks over a long cache -- and grid.z supplies the parallelism instead, each
  CTA streaming `_KEY_TILES_PER_CTA` tiles of its own key window. `num_key_parts`
  is a grid EXTENT, not the realized split: it is sized from the batch maximum,
  so on a ragged batch each CTA narrows it to what its own entry can feed
  (`_MIN_TILES_PER_PART`) and the surplus parts retire immediately.

Warp specialization, mirroring the MSA prefill scorer
(`Kernels/lib/msa/sparse_indexer_prefill.mojo`, PR #91938), on a 256-thread CTA at
every tile:
- WG0 (warps 0-3, threads 0-127) = score/epilogue consumer. Drains each S^T
  stage out of TMEM one row per thread (`tcgen05.ld.32x32b`, warp `w` / lane `l`
  -> row `32 * w + l`, so the 4 warps span all `BM_key` rows), applies the
  branchless relu, sums over each token's head columns entirely within the
  thread, scales by k_scale, and writes one f32 per (token, key) under the fused
  causal guard.
- WG1 (warps 4-7) = producer: warp 4 = MMA (TMEM owner + `K @ Q^T` per K tile),
  warp 5 = TMA (deep K-ring producer). Warps 6-7 are idle and register-dealloc to
  the floor; they exist only so `setmaxnreg` has a whole warpgroup to issue the
  `dec` from, which is what funds the consumer's 216. Role-to-role mbars
  (k_full/k_empty for the K ring, s_full/s_empty for the multi-stage S^T) replace
  the shipped kernel's per-iteration whole-CTA `named_barrier`; the resident Q
  owns no barrier and rides `k_full[0]`.

Scores carry no cross-key reduction (no softmax denominator), and a thread's
only global write is `output[global_token, key_local]`. Key windows are
therefore disjoint output elements: the split needs no combine pass, no
workspace, and no atomics.

Routing lives in `fp8_index_score_sm100`; see the comment there for the
measured thresholds. Two disjoint corners land here: enough token blocks to
fill the grid on their own (long prefill), or too few token blocks but a key
range deep enough that splitting it fills the grid (decode / MTP-decode). The
kernel body supports all head counts uniformly; the route currently admits
`num_heads` in {32, 64}.

NVIDIA SM100 only (SS-UMMA / TMA / tcgen05). Verified against
`nn.index_fp8.fp8_index_naive` via `test_index_fp8` and end-to-end top-k set
match via `test_mla_index_fp8`.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.sync import barrier, named_barrier
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from std.bit import next_power_of_two
from std.math import align_down, align_up, ceildiv, clamp
from std.sys import get_defined_bool, get_defined_int, size_of
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from layout import (
    Coord,
    PointerStorage,
    TensorLayout,
    TensorStorage,
    TileTensor,
)
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
)

from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

from linalg.arch.sm100.mma import smem_descriptor
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    TMemTile,
    elect,
    elect_mma_arrive,
    expect_bytes_pred,
    llvm_opaque_tid,
    splitk_window,
    store_global_pred,
)
from nn.attention.mha_operand import MHAOperand


# Defined locally (not imported from `sparse_index_fp8_sm100`) so that file can
# route to this one without an import cycle; they must stay type-identical to
# its aliases or the `rebind` at its call site stops compiling.
comptime _INDEX_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime QTMATileT[
    dtype: DType, MMA_N: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(MMA_N, 1, depth), _INDEX_SWIZZLE]
comptime KTMATileT[
    dtype: DType, BM_key: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(BM_key, 1, depth), _INDEX_SWIZZLE]


# WG0 (warps 0-3) = 128 epilogue/score consumers; the epilogue drains one TMEM
# lane per thread, so this must equal BM_key. The producer needs only two warps,
# one to issue MMAs and one to issue TMA loads. A second producer PAIR exists only
# to make `setmaxnreg.sync.aligned` legal: it is warpgroup-collective -- every warp
# of a warpgroup must execute it with the same operand -- so an asymmetric 216/40
# cap requires WG1 to be complete, at the price of two permanently idle warps.
#
# TWO register numbers are in play and they answer different questions.
# `setmaxnreg.inc` is the consumer REGION's allocation budget: ptxas allocates the
# code after `USETMAXREG.TRY_ALLOC` against it, and registers above the reported
# count are genuinely used there. The REPORTED count (`Used N registers`) is only
# the driver's occupancy divisor, `min(demand, 65536 / (nthreads * ctas_per_sm))`
# = 128 here. What must fit under the REPORTED count is whatever stays live across
# the role split, before the pool is redistributed -- that, not the region budget,
# is what spills.
#
# 216 + 40 = 2 * 128 returns the producers' share to the consumer exactly, which is
# what makes `TRY_ALLOC` succeed; the launcher asserts that identity at every tile.
# Deleting both `setmaxnreg` ops takes the spill 8 B -> 216 B with 32 `LDL` back in
# the tile loop.
comptime _PROD_WARPS_FORCE = get_defined_int["FP8_INDEX_PROD_WARPS", 0]()
comptime _NUM_SOFTMAX_THREADS = 128

# Hardware barrier id for the consumer warpgroup's two private syncs (the q_scale
# publish in the prologue, the TMEM drain before dealloc). Id 0 belongs to the
# whole-CTA `barrier()`, which `named_barrier` also defaults to, so a
# warpgroup-scoped sync left on the default would share it.
comptime _CONSUMER_BAR: Int32 = 1


@always_inline
def _prefill_prod_warps[MMA_N: Int]() -> Int:
    comptime if _PROD_WARPS_FORCE > 0:
        return _PROD_WARPS_FORCE
    return 4


@always_inline
def _prefill_nthreads[MMA_N: Int]() -> Int:
    return _NUM_SOFTMAX_THREADS + WARP_SIZE * _prefill_prod_warps[MMA_N]()


# `setmaxnreg` needs BOTH warpgroups whole: with a partial producer warpgroup
# there is no legal way to issue the `dec`, and an `inc` without a matching `dec`
# claims registers no warp released.
@always_inline
def _use_setmaxnreg[MMA_N: Int]() -> Bool:
    return _prefill_prod_warps[MMA_N]() == 4


# Column chunk for the TMEM->register drain. Must stay <= 64 or ptxas runs out of
# destination registers for the `tcgen05.ld`; the asserts in the kernel body carry
# the rest of the constraints (it must divide MMA_N, be a multiple of 4, and leave
# `MMA_N // chunk >= 2` for the rotated epilogue). The decode twin holds a
# same-named constant at its own default of 16, under its own constraints; only
# this one is define-driven.
comptime _EPILOGUE_CHUNK = get_defined_int["FP8_INDEX_EPILOGUE_CHUNK", 32]()
# Producer register floor, applied to ALL FOUR of WG1's warps -- MMA, TMA and the
# two idle ones -- because `setmaxnreg.sync.aligned` is warpgroup-granular: every
# warp of a warpgroup must pass the SAME operand, so a per-warp split is UB. The
# consumer claims the rest so its TMEM->register fragment does not spill. Spilling
# is NOT monotonic in the cap, so treat this pair as a sweep point, not a
# derivation.
comptime _NUM_REG_PRODUCER = 40

# S^T TMEM stages: 2 lets the consumer read stage `it` while the MMA writes
# `it+1`, which is what makes deferring the epilogue's WAR fence affordable.
comptime _S_TMEM_STAGES = 2


# The rest of the pipeline is derived from the N-tile, because TMEM decides how
# many CTAs can be co-resident.
#
# The S^T stages cost `_S_TMEM_STAGES * align_up(MMA_N, 32)` of the SM's 512
# TMEM columns, and `tcgen05.alloc` only accepts a POWER-OF-TWO column count. So
# MMA_N=128 costs 256 (exactly half the SM, hence 2 CTAs/SM), while MMA_N=192
# uses 384 and must ALLOCATE 512 -- the whole SM, hence 1 CTA/SM.
#
# Residency is therefore a step function of the N-tile, and crossing 128 columns
# costs half the consumer warps. The epilogue is instruction- and latency-bound and
# its cost does not depend on which operand streams, so consumer warps per SM is
# the lever: measured on a batch-8 107K-key MTP step, MMA_N=192 cut 16.5% of the
# instructions and came out +2.5% in cycles -- a wash. Hence prefer a DIVISOR of
# the token count over a multiple: 3 tokens at nh=32 (96 columns, 2*96 = 192 -> 256
# allocated) takes the same exactness as 6 while staying on the 2-CTA/SM side.
#
# SMEM follows residency: a K stage is 128*128 fp8 = 16KB against a
# `228KB / ctas_per_sm - 1024` budget, so 2 CTAs/SM affords 6 stages plus the
# resident Q, and 1 CTA/SM affords 12.
@always_inline
def _ctas_per_sm[MMA_N: Int]() -> Int:
    return 2 if MMA_N <= 128 else 1


@always_inline
def _k_ring_stages[MMA_N: Int]() -> Int:
    return 6 if MMA_N <= 128 else 12


# Barrier count, shared by the kernel's SMEM layout and the launcher's
# `shared_mem_bytes` sizer so the two cannot drift: k_full(NSTAGE) +
# k_empty(NSTAGE) + s_full(N_S) + s_empty(N_S). The resident Q has no barrier of
# its own -- it rides the first K tile's, see `issue_k[with_q=True]`.
@always_inline
def _n_mbars[MMA_N: Int]() -> Int:
    return 2 * _k_ring_stages[MMA_N]() + 2 * _S_TMEM_STAGES


# Consumer register cap, and it IS the budget ptxas allocates the consumer region
# against -- not a formality above the launch allocation. Measured by sweeping it on
# one 128-column body: 216/176 spill 8 B, falling to 136 B at an `inc` of 128, with
# the reported count pinned at 128 throughout. That body is at 0 spill today, so the
# sweep no longer reproduces as written -- a ceiling is only observable from a body
# that exceeds it, so force the pressure back up before re-running it.
#
# A one-sided dial: raise it and the consumer gets the room, lower it and the
# consumer spills. Do NOT lower it to "save" anything -- the producers are already
# at the 40 they need, and the pool identity has to hold.
@always_inline
def _num_reg_consumer[MMA_N: Int]() -> Int:
    return 216 if MMA_N <= 128 else 256


# The per-thread LAUNCH allocation: what the driver divides into the register file
# for occupancy, and the cap on the REPORTED count. It is NOT the ceiling on the
# consumer region -- `_num_reg_consumer` is. Do NOT read the difference
# `_static_reg_budget - <what the code uses>` as available slack: the two quantities
# live on opposite sides of `TRY_ALLOC`. Allocation is granular in 8, so round down.
@always_inline
def _static_reg_budget[MMA_N: Int]() -> Int:
    return align_down(
        65536 // (_prefill_nthreads[MMA_N]() * _ctas_per_sm[MMA_N]()), 8
    )


# Stage the q_scales in registers once per CTA instead of re-reading them from SMEM
# on every key tile. A thread folds every column of its own key row, so the hoist
# costs exactly `MMA_N` registers; it is on at every shipped width, and every
# instantiation is at zero spill and zero in-loop shared loads.
#
# Raising the register budget does NOT raise the threshold, and that is the
# load-bearing warning: a uniform 168 at 128 columns spills 56-220 B, because with
# more registers ptxas commits to the aggressive schedule that hoists all 8
# `LDTM.x16` across the folds and then does not fit. Slack is capacity, not a
# guarantee. `-D FP8_INDEX_HOIST_QS=0|1` forces it either way for a paired A/B.
comptime _FORCE_HOIST_QS = get_defined_int["FP8_INDEX_HOIST_QS", -1]()


# How the epilogue schedules its TMEM->register chunk loads against the folds. Arm 0
# is the shipped default; 1 and 2 are gated A/B arms and neither measured a win.
#
#   0  load chunk c, fold it, repeat; release the stage after the last fold.
#   1  issue chunk c+1 BEFORE folding chunk c (two fragments live). Buys cover for the
#      load -- ptxas issues both loads of a pair back to back, so half sit only ~4
#      instructions ahead of the `FMNMX` that reads them -- at +16 registers.
#   2  as 1, but release the stage before the LAST fold. Buys the MMA warp a head start
#      by making the CONSUMER stall on a drain it previously never reached, and the
#      consumer is the critical role, so very likely a LOSS.
comptime _EPI_ROTATE = get_defined_int["FP8_INDEX_EPI_ROTATE", 0]()

# Emit the epilogue's guarded global store as a PTX `@%p st.global.b32` rather than
# a Mojo `if`. Default ON; set to 0 to get the branch back for an A/B. See the store
# site for the measurements.
#
# SCOPE WARNING: the measurement that justified this was taken on the MMA_N=96 arm,
# which is at zero spill either way. It is NOT free on the others -- it takes the
# nh=64 MMA_N=128 arms from 0 B to 20 B (ragged) and 0 B to 28 B (paged), and that
# spill is LIVE at the shipped default. Re-measure per instantiation rather than
# treating a change here as free.
comptime _PRED_STORE = get_defined_bool["FP8_INDEX_PRED_STORE", True]()


@always_inline
def _hoist_q_scales[MMA_N: Int]() -> Bool:
    comptime if _FORCE_HOIST_QS >= 0:
        return _FORCE_HOIST_QS != 0
    # A measured table per configuration, NOT `budget - working_set`: ptxas inflates
    # its allocation to fill whatever budget it is handed (at 1 CTA/SM it reports 255
    # of 256 with no hoist at all), so the slack it leaves cannot be predicted
    # arithmetically. EVERY arm below hoists; what separates them is the local-memory
    # traffic that replaces the shared traffic. Rows are keyed on HEAD COUNT, and the
    # counts are scoped to the CONSUMER loop:
    #
    #   config                          spill nh=32      spill nh=64      loop LDL
    #   256 thr, setmaxnreg 216/40        0 / 0 B          0 / 0 B            0
    #   256 thr, setmaxnreg deleted     216 / 228 B      216 / 228 B         32
    #   224 thr (no setmaxnreg)         216 / 228 B      256 / 260 B         32
    #   192 thr, budget 168              56 / 64 B       196 / 220 B        9-30
    #
    # Only the first is a win, and it is the ONLY residency-neutral configuration in
    # which 128 columns pay. The others swap 32 shared loads for up to 32 local loads,
    # which is worse than not hoisting at all.
    comptime if _use_setmaxnreg[MMA_N]():
        return MMA_N <= 128
    # Below here only `-D FP8_INDEX_PROD_WARPS` lands, since the default shape
    # always has `setmaxnreg`. At 2 producer warps the budget is a uniform 168,
    # which fits 96 columns and traps at 128 (see the table above).
    comptime if _static_reg_budget[MMA_N]() >= 168:
        return MMA_N <= 96
    # Only `-D FP8_INDEX_PROD_WARPS=3` lands here: 7 warps is not a whole warpgroup,
    # so it forfeits `setmaxnreg` and ptxas rounds the CTA back up to the 8-warp
    # launch bound. A measured dead end, kept only so the knob stays legal.
    return MMA_N <= 64


# Key tiles a CTA streams before the launcher adds another grid.z part. Deep
# enough to amortize the CTA prologue (TMEM alloc, mbar init, Q staging) and to
# keep the K ring full; the launcher overrides it upward in part count when that
# many tiles per CTA would not fill a wave.
comptime _KEY_TILES_PER_CTA = 16

# Ceiling, in waves at `_ctas_per_sm`, on the part count the amortized arm of
# the `num_key_parts` derivation may request -- see the launcher for why.
comptime _MAX_KEY_PART_WAVES = 4

# Floor on the key tiles a single grid.z part may own, applied INSIDE the kernel
# against the CTA's own per-entry tile count. The launcher can only size the part
# count from `max_num_keys` (the batch maximum), so on a ragged batch every entry
# but the deepest is over-split; this is what stops a shallow entry degenerating
# into 1-tile CTAs. Uniform batches are unaffected -- there the launcher's own
# per-part tile count already clears this floor.
comptime _MIN_TILES_PER_PART = 4


@__name(t"fp8_index_score_prefill_sm100_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
# Cap the launch register count so `_prefill_nthreads[MMA_N]()` threads fit at
# `_ctas_per_sm[MMA_N]()` CTAs/SM (maxntid + minctasm), mirroring MSA and FA4:
# without it the launch requests more than `65536 / nthreads` regs/thread and the
# warpgroup reg-alloc/dealloc below has no reserved pool to redistribute.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_prefill_nthreads[N_TOKENS * num_heads]())
    )
)
# Must agree with `_ctas_per_sm`; spelled inline because the decorator is
# evaluated against the kernel's own parameters (same shape as
# `state_space/mamba2_ssd_scan.mojo:576-579`), and asserted against the helper in
# the body below so the two cannot drift.
@__llvm_metadata(
    `nvvm.minctasm`=SIMDLength(2) if N_TOKENS * num_heads
    <= 128 else SIMDLength(1)
)
def _fp8_index_score_prefill_kernel_sm100[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    *,
    VLStorageType: TensorStorage = PointerStorage[element_width=1],
    QSStorageType: TensorStorage = PointerStorage[element_width=1],
    OutStorageType: TensorStorage = PointerStorage[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[
        DType.uint32, VLLT, ImmutAnyOrigin, Storage=VLStorageType
    ],
    q_s: TileTensor[DType.float32, QSLT, ImmutAnyOrigin, Storage=QSStorageType],
    output: TileTensor[
        DType.float32, OutLT, MutAnyOrigin, Storage=OutStorageType
    ],
    max_num_keys_dev: Int32,
    causal_dev: Int32,
    num_key_parts_dev: Int32,
    out_row_begin_dev: Int32,
    out_row_end_dev: Int32,
):
    comptime assert valid_length.flat_rank == 1
    comptime MMA_N = N_TOKENS * num_heads
    # Two producer warps are the floor: the MMA and TMA roles are separate warps
    # because each spins on its own mbarrier.
    comptime assert (
        _prefill_prod_warps[MMA_N]() >= 2
    ), "the producer needs an MMA warp and a TMA warp"
    # The N-tile is the B operand's extent, so it must be a legal UMMA N
    # (multiple of 16, <= 256 for KIND_F8F6F4 at cta_group=1). Note nothing
    # downstream checks that for us: `UMMAInsDescriptor.create` just encodes
    # `N >> 3` into the descriptor, so an illegal N would be emitted silently.
    comptime assert MMA_N % 16 == 0 and MMA_N <= 256, (
        "MMA_N (N_TOKENS * num_heads) must be a legal UMMA N -- a multiple of"
        " 16 and at most 256; got "
        + String(MMA_N)
    )
    comptime AT = DType.float32
    comptime SW = _INDEX_SWIZZLE
    comptime NSTAGE = _k_ring_stages[MMA_N]()
    comptime N_S = _S_TMEM_STAGES
    comptime CTAS_PER_SM = _ctas_per_sm[MMA_N]()
    # The `nvvm.minctasm` decorator spells this inline; keep them in lockstep.
    comptime assert CTAS_PER_SM == (2 if MMA_N <= 128 else 1)
    # WG1 (producer) thread roles.
    comptime MMA_WARP = 4
    comptime TMA_WARP = 5

    # Index arithmetic runs in SIGNED 32-bit. Every quantity below is bounded by a
    # context length or a token count, and `Int` is 64-bit here, which costs a second
    # register plus an `IMAD.X`/`ISETP...EX` tail on every add and compare -- measured
    # to be what the consumer spills across the `setmaxnreg` boundary.
    #
    # SIGNED, deliberately, against the surrounding SM100 `UInt32` convention:
    # `seq_len`, `block_key_bound`, `n_key_tiles` and `n_tiles_local` are all formed
    # by subtraction and all explicitly tested `<= 0`. Unsigned would underflow those
    # into huge counts, and since `n_tiles_local` is the trip count all three warp
    # roles walk, the failure would be a HANG rather than a wrong answer. Only the
    # store OFFSET goes back to 64-bit, and it has to (see the store).
    var max_num_keys = max_num_keys_dev
    var causal = causal_dev
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)

    var start_of_seq = Int32(valid_length[b])
    var end_of_seq = Int32(valid_length[b + 1])
    var seq_len = end_of_seq - start_of_seq

    # This launch owns global token rows `[out_row_begin, out_row_end)` and writes
    # them to `output` rows `[0, out_row_end - out_row_begin)`. The caller chunks
    # that window to bound the score buffer, which is the whole point of the
    # split; unchunked it is `[0, total_seq_len)` and everything below reduces to
    # the unwindowed form.
    #
    # Clamping to the window here (rather than predicating the store) is what
    # keeps the epilogue free: token blocks are indexed from `tok_lo`, so no block
    # straddles a chunk boundary, no token is scored twice, and the store's
    # liveness test just swaps `seq_len` for `tok_hi`. `seq_len` itself stays the
    # TRUE sequence length -- the causal bound is an absolute position and must
    # not see the window.
    var tok_lo = max(Int32(0), out_row_begin_dev - start_of_seq)
    var tok_hi = min(seq_len, out_row_end_dev - start_of_seq)
    var tok0 = tok_lo + Int32(block_idx.y) * Int32(N_TOKENS)
    # Folded once here so the store's row arithmetic is the same single add it
    # was before the window existed.
    var out_row0 = start_of_seq - out_row_begin_dev

    var num_keys = Int32(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len
    # The host bounds `max_num_keys`, but `num_keys` is per-entry device data it never
    # sees. On the MLA path this holds structurally; on the ragged path it rests on a
    # caller contract, so check the final value where it can be seen. DEFAULT assert
    # mode, NOT "safe": a "safe" assert in this kernel drags in `vprintf` and grew the
    # emitted PTX +26% while perturbing register allocation.
    debug_assert(
        num_keys <= max_num_keys_dev,
        "fp8 index prefill: per-entry num_keys exceeds max_num_keys",
    )

    # Bail uniformly (every thread) before any collective op (TMA mbar / tcgen05
    # alloc); a divergent early return deadlocks them. A token block past the
    # sequence -- or outside this launch's row window -- produces no output (the
    # caller's -inf fill covers those rows).
    #
    # A chunked launch retires most of its CTAs here, since the grid is sized by
    # the whole batch, so hoisting this above the `cache_length` load to save
    # them a global read looks free. It measured neutral (4096 tokens, 5 chunks)
    # -- the chunking overhead is not where those CTAs spend it -- so the load
    # stays where it was.
    if tok0 >= tok_hi or seq_len <= 0:
        return

    # Keys this token block must stream: bounded by the deepest live token of
    # the block under the causal mask (each token still gets its own per-key
    # guard in the epilogue). Non-causal streams every key; causal trims the
    # triangle a zero-prefix fresh prefill leaves off the end.
    var last_tok = min(tok0 + Int32(N_TOKENS), tok_hi) - 1
    var block_key_bound = num_keys - (seq_len - 1 - last_tok) * causal
    # `block_key_bound` can be <= 0 (a causal bound that trims the whole block), so the
    # SUBTRACTION above stays signed. The `ceildiv` does not: `SIMD.__ceildiv__`
    # branches at COMPTIME on the dtype -- signed lowers to `-(x // -d)` with a
    # ~9-instruction correction chain, unsigned to an add and a shift. So the clamp has
    # to sit ABOVE the ceildiv; moving it below leaves the expensive form in place.
    var n_key_tiles = ceildiv(
        UInt32(max(block_key_bound, Int32(0))), UInt32(BM_key)
    )

    # grid.z splits that tile range across CTAs. Scores carry no cross-key reduction,
    # and a thread's store address is `global_token * max_num_keys + it * BM_key + row`,
    # so disjoint tile windows write disjoint elements -- no combine pass and no
    # workspace. `splitk_window` front-loads, so only trailing parts come up empty.
    #
    # The launcher sizes `num_key_parts_dev` from `max_num_keys`, a batch MAXIMUM, so on
    # a ragged batch it over-splits every entry but the deepest. Each CTA therefore
    # narrows the part count to what its OWN tile count can feed, and the surplus CTAs
    # return here, before any collective. `p_eff` depends only on CTA-uniform values, so
    # all three warp roles derive the same trip count from it -- load-bearing, because
    # windowing the consumer alone unbalances the k/s mbar handshakes and HANGS.
    var p_eff = clamp(
        ceildiv(n_key_tiles, UInt32(_MIN_TILES_PER_PART)),
        UInt32(1),
        UInt32(num_key_parts_dev),
    )
    if UInt32(block_idx.z) >= p_eff:
        return
    var win = splitk_window(
        n_key_tiles,
        p_eff,
        UInt32(block_idx.z),
    )
    var tile_begin = Int32(win[0])
    var n_tiles_local = Int32(win[1]) - tile_begin

    # Second uniform bail (an empty trailing part, or a batch entry whose causal
    # bound left it no keys). Uniform for the same reason as the one above, and
    # likewise ahead of every collective.
    if n_tiles_local <= 0:
        return

    # FA4 stateless S = K @ Q^T accumulator. `MMA_M = BM_key = 128 > 64` keeps
    # `use_ws` False (the standard, non-packed TMEM datapath). It carries no
    # handshake and no accumulator staging, so the S mbars and the stage stride
    # below are driven by this kernel. `mma_kind` has no dtype-derived default,
    # so an fp8 operand MUST select the f8f6f4 instruction family.
    comptime QK = SM100TensorAccumulator[
        dtype,
        AT,
        MMA_M=BM_key,
        MMA_N=MMA_N,
        BK=depth,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F8F6F4 if dtype.is_float8() else UMMAKind.KIND_F16,
        swizzle_a=SW,
        swizzle_b=SW,
        transpose_b=True,
        cta_group=1,
        num_stages=1,
    ]
    # Operand descriptor K extent, padded to the MMA_K granularity (equals the
    # accumulator's internal `padded_BK` at depth == 128).
    comptime compute_BK = align_up(depth, 16)
    comptime S_COLS = align_up(MMA_N, 32)
    comptime assert BM_key == _NUM_SOFTMAX_THREADS, (
        "the epilogue drains one TMEM lane per thread, so BM_key must equal the"
        " consumer warpgroup size"
    )
    comptime assert MMA_N % _EPILOGUE_CHUNK == 0
    comptime assert _EPILOGUE_CHUNK <= 64
    # The fold walks columns in groups of four (one 16-byte q-scale load feeding
    # two f32x2 FFMAs), so a group must sit wholly inside one token and its base
    # must be 16-byte aligned. Both routes to this kernel supply num_heads in
    # {32, 64}, but nothing in the body enforced it.
    comptime assert _EPILOGUE_CHUNK % 4 == 0 and num_heads % 4 == 0, (
        "the epilogue folds columns in groups of four, so both _EPILOGUE_CHUNK"
        " and num_heads must be multiples of 4 for a group never to straddle a"
        " token boundary"
    )

    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    var smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
        name="fp8_index_sm100_prefill_smem",
    ]()
    # Q resident (B operand, one token block) | K ring (A operand) | q_scale
    # resident | mbars: k_full(NSTAGE) + k_empty(NSTAGE) + accumulator(2*N_S) |
    # tcgen05 TMEM base slot.
    var q_smem = smem
    var k_smem = smem + q_elems
    var qs_smem = (smem + q_elems + NSTAGE * k_elems).bitcast[Float32]()
    var mbar = (qs_smem + MMA_N).bitcast[SharedMemBarrier]()
    # Offsets are named because the lane-parallel init below maps a thread index
    # to an arrival count, which makes them part of the layout contract rather
    # than a one-off pointer bump. `s_empty` last is load-bearing -- see the init.
    comptime K_FULL_OFF = 0
    comptime K_EMPTY_OFF = K_FULL_OFF + NSTAGE
    comptime S_FULL_OFF = K_EMPTY_OFF + NSTAGE
    comptime S_EMPTY_OFF = S_FULL_OFF + N_S
    comptime N_MBAR = S_EMPTY_OFF + N_S
    comptime assert N_MBAR == _n_mbars[MMA_N]()
    var k_full = mbar + K_FULL_OFF
    var k_empty = mbar + K_EMPTY_OFF
    var s_full = mbar + S_FULL_OFF
    var s_empty = mbar + S_EMPTY_OFF
    var ptr_tmem = (mbar + N_MBAR).bitcast[UInt32]()

    comptime q_flat_layout = tt_row_major[q_elems]()
    comptime k_flat_layout = tt_row_major[k_elems]()

    # Lane-parallel init, mirroring FA4's `FA4MiscMBars.init`: barrier `i` is
    # initialized by thread `i`, so all N_MBAR `mbarrier.init` issue as one `STS.64`
    # with a lane-indexed address instead of N_MBAR serialized stores from thread 0.
    #
    # `s_empty` is the WAR release and the only class with a non-default count (one
    # arrival per consumer thread); `s_full` is armed by a single tcgen05 commit and
    # every producer barrier by one TMA completion, so those take the default of 1.
    # Because the layout puts the `s_empty` stages LAST, the count map is a single
    # compare that ptxas folds into a `SEL` -- keep them last or this grows a branch
    # chain. Indexed by `tid` rather than lane so it stays correct if N_MBAR ever
    # passes WARP_SIZE.
    comptime assert N_MBAR <= _prefill_nthreads[MMA_N]()
    if tid < N_MBAR:
        mbar[tid].init(
            Int32(_NUM_SOFTMAX_THREADS) if tid >= S_EMPTY_OFF else Int32(1)
        )

    # tcgen05 alloc is warp-collective (.sync.aligned): exactly one warp (the MMA warp).
    # Release the lock right after so co-resident CTAs can allocate. `tcgen05.alloc`
    # takes a POWER-OF-TWO column count, so the stages' footprint is rounded up for the
    # allocation (MMA_N=192 uses 384 and allocates 512) while the stage stride stays
    # `S_COLS` -- the waste is dead columns at the top, not a gap between stages.
    comptime TMEM_USED_COLS = N_S * S_COLS
    comptime TMEM_COLS = UInt32(next_power_of_two(TMEM_USED_COLS))
    comptime assert Int(TMEM_COLS) * CTAS_PER_SM <= 512, (
        "S^T stages need "
        + String(TMEM_COLS)
        + " TMEM columns (rounded up to a power of two from "
        + String(TMEM_USED_COLS)
        + "), which exceeds the SM's 512 at CTAS_PER_SM="
        + String(CTAS_PER_SM)
    )
    var wid = warp_id[broadcast=True]()
    barrier()

    comptime k_bytes = k_elems * size_of[dtype]()
    comptime q_bytes = q_elems * size_of[dtype]()

    if wid < 4:
        comptime if _use_setmaxnreg[MMA_N]():
            warpgroup_reg_alloc[_num_reg_consumer[MMA_N]()]()

        # q_scale staging, one f32 per (token, head) column. This sits INSIDE the
        # consumer branch rather than the whole-CTA prologue because it is a dependent
        # global load and no producer warp reads the result, which is what let the
        # prologue's second whole-CTA `barrier()` go: the TMA warp now reaches its first
        # K issue without waiting on 128 cold global loads. A `bar.sync` fences
        # intra-CTA `st.shared`/`ld.shared`, so no `fence_async_view_proxy` is needed.
        #
        # One wave per `_NUM_SOFTMAX_THREADS` columns, with the bound check emitted ONLY
        # for a wave that does not fill, so MMA_N=128 stages with no predicate at all.
        # Waves rather than `tid < MMA_N` because the staging must not reach past the
        # consumer warpgroup: at MMA_N=192 the old form wrote from warps 4 and 5 too.
        @__parameter
        @always_inline
        def stage_qs(col: Int):
            var qs_tok = Int32(col // num_heads)
            if tok0 + qs_tok < seq_len:
                qs_smem[col] = q_s[
                    start_of_seq + tok0 + qs_tok, col % num_heads
                ][0]
            else:
                qs_smem[col] = 0.0

        comptime for w in range(ceildiv(MMA_N, _NUM_SOFTMAX_THREADS)):
            comptime col_base = w * _NUM_SOFTMAX_THREADS
            comptime if col_base + _NUM_SOFTMAX_THREADS > MMA_N:
                if tid + col_base < MMA_N:
                    stage_qs(tid + col_base)
            else:
                stage_qs(tid + col_base)
        named_barrier[Int32(_NUM_SOFTMAX_THREADS + WARP_SIZE)](_CONSUMER_BAR)
        var tmem_addr: UInt32 = ptr_tmem[0]
        # `tcgen05_ld[datapaths=32]` maps warp w, lane l -> accumulator row
        # `WARP_SIZE * w + l`, so this warpgroup's 4 warps cover all BM_key rows
        # one-per-thread. A thread owning a whole key row makes the head sum
        # thread-local (no cross-lane reduction) and puts 32 consecutive
        # `key_local` in each warp, so each store is one 128B transaction.
        # The consumer is warps 0-3, so masking `tid` gives `wid * 32 + lane_id()`
        # identically while reading `tid` once, avoiding a second `S2R`.
        # `llvm_opaque_tid` is FA4's anti-hoist intrinsic; it does NOT bind here
        # (exactly one `%tid.x` read either way, in the prologue ahead of
        # `TRY_ALLOC`, because the q-scale staging above already needs `tid`), and is
        # kept only for the cheaper spelling.
        var row = Int32(llvm_opaque_tid() & (_NUM_SOFTMAX_THREADS - 1))
        var c_state = PipelineState[N_S]()

        # The q_scales a thread needs are CTA-invariant, yet the tile loop below
        # re-read all of them on EVERY key tile: measured at 128 columns as 1,295,264
        # `LDS`, 7.8% of the instruction stream and 36% of the shared-load pipe, with
        # the stalls landing on the consuming `FFMA2` rather than on the `LDS` itself.
        #
        # `qs_reg` is the single access path for the fold either way; only the FILL
        # SITE is conditional. Where the working set fits (see `_hoist_q_scales`) it is
        # filled once here, taking the in-loop `LDS` count to ZERO; otherwise each
        # chunk fills its own four in place below. Read FOUR at a time: adjacent scalar
        # reads coalesce into one 16-byte access and an explicit width-2 load does not
        # re-merge. Every index MUST stay comptime -- a runtime index forces the array
        # to local memory.
        var qs_reg = Array[Scalar[AT], MMA_N](uninitialized=True)
        comptime if _hoist_q_scales[MMA_N]():
            comptime for j in range(MMA_N // 4):
                var qs4 = qs_smem.unsafe_load[width=4, alignment=16](4 * j)
                comptime for e in range(4):
                    qs_reg[4 * j + e] = qs4[e]

        # k_scale for one key row, reused across the block's tokens; the
        # streamed keys mean it can no longer be staged once per CTA the way the
        # resident-key kernel does. `key < num_keys` is what stops an OOB pool
        # row being dereferenced, and it is also the ONLY guard the prefetch
        # below needs: a key that passes it exists in this entry's cache, so the
        # paged pointer is valid even for a tile that another grid.z part owns.
        @__parameter
        @always_inline
        def gather_k_scale(key: Int32) -> Float32:
            if key < num_keys:
                return ks_operand.block_paged_ptr[1](
                    UInt32(b), UInt32(key), UInt32(0), UInt32(0)
                )[0].cast[DType.float32]()
            return 0.0

        # Rotate that gather a whole key tile ahead. It is a two-hop dependent load
        # (block table -> scale row) whose only consumer is the final `FMUL` of each
        # token's fold, so issuing it in the same iteration leaves just the first
        # token's fold to cover the miss -- 37% L2 hit, putting 5.5% of warp-stall
        # samples on that one `FMUL` against 0.2% on the identical one a token later.
        # Do NOT add an `i + 1 < n_tiles_local` guard: it buys nothing over
        # `< num_keys` and would turn a predicated `LDG` into a reconvergence
        # quartet. The price is one speculative gather per CTA.
        var k_scale = gather_k_scale(tile_begin * Int32(BM_key) + row)

        # Raw base of the score buffer, for the `_PRED_STORE` arm only -- it takes
        # a pointer, where `raw_store` takes an offset. Element (0, 0), so this is
        # the storage base with no layout arithmetic and no exposure to
        # `linear_idx_type`, which the store offset deliberately outgrows. Dead
        # code (and free) in the branch arm.
        var out_base = output.ptr

        # `n_tiles_local` is `Int32`, so `range` yields an `Int32` induction
        # variable directly (`range.mojo:495`) -- the whole key-index chain stays
        # 32-bit with no per-iteration cast, which is what the narrowing rule
        # requires. All three warp roles walk this same count.
        for tile_i in range(n_tiles_local):
            var key_local = (tile_begin + tile_i) * Int32(BM_key) + row
            var k_scale_next = gather_k_scale(key_local + Int32(BM_key))

            # `PipelineState.index()` is already `UInt32`; keep it that way
            # rather than round-tripping through 64-bit `Int`.
            var cs = c_state.index()
            s_full[cs].wait(c_state.phase())
            var s_it = tmem_addr + cs * UInt32(S_COLS)

            # Drain this thread's key row in `_EPILOGUE_CHUNK`-column chunks. A token
            # owns a CONTIGUOUS column range (`col // num_heads`), so its sum is final
            # at its last column: store it right there and the accumulator collapses
            # from `SIMD[AT, N_TOKENS]` to one f32x2, rather than keeping every token
            # sum and all 128 q-scales live at once (~250 registers, which spilled).
            #
            # The chunk loads carry no wait between them, so they pipeline against the
            # folds: `tcgen05.ld` register outputs are automatically ordered, and the
            # single `tcgen05_load_wait` is only the WAR fence before the stage is
            # released.
            #
            # `num_heads` and `_EPILOGUE_CHUNK` are both powers of two, so a chunk
            # never straddles a token partway and the completion test stays comptime.
            # A *runtime* token index would spill the accumulator to local memory.
            #
            # Columns fold two at a time so the multiply-accumulate is a single packed
            # `fma.rn.f32x2` (SASS FFMA2). `.fma()` is required over `a * b + c`: LLVM
            # does not contract an f32 pair into one FFMA2. The relu is written as a
            # vector op but PTX has no `max.f32x2` at any ISA version, so it lowers to
            # one FMNMX per column -- that floor is why a pair costs 2 FMNMX + 1 FFMA2.
            @__parameter
            @always_inline
            def issue_frag[c: Int]() -> Array[Scalar[AT], _EPILOGUE_CHUNK]:
                return TMemTile[AT, BM_key, _EPILOGUE_CHUNK](
                    s_it + UInt32(c * _EPILOGUE_CHUNK)
                ).load_async()

            @__parameter
            @always_inline
            def consume_frag[
                c: Int
            ](frag: Array[Scalar[AT], _EPILOGUE_CHUNK], mut acc: SIMD[AT, 2],):
                comptime for i in range(_EPILOGUE_CHUNK // 4):
                    comptime col4 = c * _EPILOGUE_CHUNK + 4 * i
                    comptime if not _hoist_q_scales[MMA_N]():
                        var qs4 = qs_smem.unsafe_load[width=4, alignment=16](
                            col4
                        )
                        comptime for e in range(4):
                            qs_reg[col4 + e] = qs4[e]
                    comptime for h in range(2):
                        comptime col = col4 + 2 * h
                        var raw = SIMD[AT, 2](
                            frag[4 * i + 2 * h], frag[4 * i + 2 * h + 1]
                        )
                        acc = max(raw, SIMD[AT, 2](0)).fma(
                            SIMD[AT, 2](qs_reg[col], qs_reg[col + 1]), acc
                        )
                        comptime if (col + 2) % num_heads == 0:
                            comptime t = col // num_heads
                            var tok_local = tok0 + Int32(t)
                            # Fused causal mask (branchless): token tok_local sees
                            # keys up to cache_len + tok_local, so forbidden slots
                            # are left for the caller (the MLA caller pre-fills
                            # -Float32.MAX) and the separate mask pass is skipped.
                            # The liveness half is memory safety rather than
                            # masking -- a dead token's row belongs to the NEXT
                            # batch entry.
                            #
                            # The guard skips no work worth skipping (0.03%
                            # divergence), but as a BRANCH it costs a
                            # `BSSY`/`BRA`/`NOP`/`BSYNC` quartet per token per key
                            # tile: 4.6% of the instruction stream at 96 columns,
                            # 10.6% at 128. `_PRED_STORE` folds it into a PTX
                            # `@%p st.global.b32` instead, measured at -5.4% to
                            # 0.0% wall clock over five shapes with none
                            # regressing. Predicating only the STORE leaves every
                            # register dependency intact, so ptxas does NOT hoist
                            # the 8 `LDTM.x16` across the folds: zero `STL`/`LDL`
                            # in both arms at MMA_N 96 and 128, nh 32 and 64.
                            var key_bound = (
                                num_keys - (seq_len - 1 - tok_local) * causal
                            )
                            # The OFFSET must be 64-bit: the score buffer is
                            # `total_seq_len * max_num_keys` f32, so the flat index
                            # passes 2^31 at ~13.1K tokens x 163840 keys and 2^32
                            # at 32K x 163840, both reachable by configuration with
                            # no allocation guard on that path. Both factors are
                            # `Int32` though, so this is one `IMAD.WIDE` (32x32->64)
                            # rather than a 64-bit multiply. Do NOT narrow it.
                            comptime if _PRED_STORE:
                                var out_row = out_row0 + tok_local
                                store_global_pred(
                                    out_base
                                    + (
                                        Int(out_row) * Int(max_num_keys)
                                        + Int(key_local)
                                    ),
                                    k_scale * (acc[0] + acc[1]),
                                    Int32(
                                        key_local < key_bound
                                        and tok_local < tok_hi
                                    ),
                                )
                            else:
                                if key_local < key_bound and tok_local < tok_hi:
                                    var out_row = out_row0 + tok_local
                                    output.raw_store(
                                        Int(out_row) * Int(max_num_keys)
                                        + Int(key_local),
                                        k_scale * (acc[0] + acc[1]),
                                    )
                            acc = SIMD[AT, 2](0)

            # WAR release: the MMA warp may overwrite this S^T stage once every thread
            # of the warpgroup has arrived. `tcgen05_load_wait` drains this thread's
            # outstanding TMEM reads and `tcgen05_fence_before` orders them ahead of the
            # arrive. Sites that omit the fence all consume every loaded register BEFORE
            # arriving, so their register dependencies order the WAR incidentally; at
            # `_EPI_ROTATE == 2` this one does not.
            @__parameter
            @always_inline
            def release_stage():
                tcgen05_load_wait()
                tcgen05_fence_before()
                _ = s_empty[cs].arrive()

            var acc = SIMD[AT, 2](0)
            comptime NCHUNK = MMA_N // _EPILOGUE_CHUNK
            comptime if _EPI_ROTATE == 0:
                comptime for c in range(NCHUNK):
                    consume_frag[c](issue_frag[c](), acc)
                release_stage()
            else:
                # Two fragments live, alternating between two names -- the FA4
                # idiom from `mha_depth512/softmax_warp.mojo:525-598`, which rotates
                # `s1`/`s2` the same way. Names rather than an array because a
                # runtime index into a register array spills it to local memory.
                comptime assert NCHUNK >= 2, (
                    "the rotated epilogue needs at least two column chunks; at"
                    " _EPILOGUE_CHUNK="
                    + String(_EPILOGUE_CHUNK)
                    + " that means MMA_N >= "
                    + String(2 * _EPILOGUE_CHUNK)
                )
                var f_a = issue_frag[0]()
                var f_b = issue_frag[1]()
                comptime for c in range(NCHUNK):
                    # The release sits before the LAST fold, not after it. Legal because
                    # chunks 0..NCHUNK-2 are already folded, so the only outstanding
                    # load is chunk NCHUNK-1's -- also the most recently issued, which
                    # makes this correct whether `tcgen05.wait::ld` drains every load or
                    # just the last. Its registers are live, so folding after the stage
                    # is handed back reads registers, not TMEM.
                    comptime if _EPI_ROTATE >= 2 and c == NCHUNK - 1:
                        release_stage()
                    comptime if c % 2 == 0:
                        consume_frag[c](f_a, acc)
                        comptime if c + 2 < NCHUNK:
                            f_a = issue_frag[c + 2]()
                    else:
                        consume_frag[c](f_b, acc)
                        comptime if c + 2 < NCHUNK:
                            f_b = issue_frag[c + 2]()
                comptime if _EPI_ROTATE < 2:
                    release_stage()
            k_scale = k_scale_next
            c_state.step()

        # Single warpgroup-wide drain: the consumer's last `s_empty` arrive
        # happens-before this barrier, so no S^T stage is live when the MMA warp
        # frees TMEM.
        named_barrier[Int32(_NUM_SOFTMAX_THREADS)](_CONSUMER_BAR)
        if wid == 0:
            tcgen05_dealloc[1](tmem_addr, TMEM_COLS)
    else:
        if wid == MMA_WARP:
            tcgen05_alloc[1](ptr_tmem, TMEM_COLS)
            tcgen05_release_allocation_lock[1]()
            # The role runs warp-collectively and elects one lane per issue:
            # `SM100TensorAccumulator.mma` broadcasts the accumulator's TMEM
            # address from lane 0 (`shfl.sync` over the full warp mask), so
            # calling it from a single-lane region hangs the warp on a
            # convergence barrier the other 31 lanes never reach.
            comptime if _use_setmaxnreg[MMA_N]():
                warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()
            named_barrier[Int32(_NUM_SOFTMAX_THREADS + WARP_SIZE)](
                _CONSUMER_BAR
            )
            var tmem_addr: UInt32 = ptr_tmem[0]
            var e = elect()
            var kc_state = PipelineState[NSTAGE]()
            # The MMA warp is the PRODUCER of the S^T stages, so its WAR state starts
            # pre-flipped: `wait(1)` on a fresh mbar tests a phase that has not been
            # reached and falls through, replacing a prologue that pre-armed `s_empty`
            # with 256 arrivals purely to make that same first wait pass. Same device
            # as the K ring's `kp_state` below.
            var sp_state = PipelineState[N_S](0, 1, 0)
            # No separate wait for the resident Q: it lands on `k_full[0]` with the
            # first K tile, so iteration 0's `k_full` wait below covers it. `q_smem`
            # is written once and never recycled, so later iterations reading it
            # behind a re-armed `k_full[0]` is not a hazard.
            #
            # Trip count must match the consumer's and the load warp's exactly: the k/s
            # mbar handshakes and both `PipelineState` phases stay in lockstep only
            # because all three roles walk the same tile window.
            var q = smem_descriptor[
                BMN=MMA_N,
                BK=compute_BK,
                swizzle_mode=SW,
                is_k_major=True,
            ](q_smem)
            var k = smem_descriptor[
                BMN=BM_key,
                BK=compute_BK,
                swizzle_mode=SW,
                is_k_major=True,
            ](k_smem)
            for _ in range(n_tiles_local):
                var s = kc_state.index()
                k_full[s].wait(kc_state.phase())
                var st = sp_state.index()
                # WAR: the consumers must have drained this S stage before the MMA
                # overwrites it. The first pass over each stage falls through on the
                # pre-flipped phase (nothing to drain yet); after that this is the
                # consumer's `release_stage()`.
                s_empty[st].wait(sp_state.phase())
                QK.mma(
                    k + s * UInt32(k_elems),
                    q,
                    tmem_addr + st * UInt32(S_COLS),
                    c_scale=0,
                    elect=e,
                )
                elect_mma_arrive(s_full + st, e)
                # Release K stage s only after the MMA has drained it
                # (tcgen05.commit tracks the async MMA); a plain mbar arrive
                # would let the load warp overwrite K mid-read.
                elect_mma_arrive(k_empty + s, e)
                kc_state.step()
                sp_state.step()
        elif wid == TMA_WARP:
            comptime if _use_setmaxnreg[MMA_N]():
                warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()
            var kp_state = PipelineState[NSTAGE](0, 1, 0)
            var n_prefetch = min(Int32(NSTAGE), n_tiles_local)
            var e = elect()

            # `with_q` carries the resident Q on this tile's barrier as well, and holds
            # for the PEELED FIRST ISSUE ONLY (see the prologue below): Q is staged
            # once, so every later arming of a stage -- including stage 0's refill --
            # expects `k_bytes` alone.
            #
            # One expect-bytes with the bytes SUMMED is the only legal shape, not one
            # call per copy: it lowers to `mbarrier.arrive.expect_tx`, so it IS the
            # producer's single arrival, and against an init count of 1 a second call
            # would drive the pending-arrival count below zero -- out of spec, not
            # merely redundant. Summed, both TMAs deposit into the one tx counter and
            # the barrier fires when the last byte of either lands.
            #
            # The single-lane guard rides an `@%p` on each instruction rather than an
            # `if e != 0:`. That is NOT a branch removal -- ptxas already if-converts
            # the guarded form and both shapes emit the same SASS at the issue. It is
            # worth 8 fewer instructions of uniform-datapath setup per sidecar, and
            # consistency with every other SM100 producer.
            @__parameter
            @always_inline
            def issue_k[
                with_q: Bool = False
            ](it: Int32, state: PipelineState[NSTAGE]):
                var s = state.index()
                # `k_row0` stays `Int`: `async_copy_3d` takes
                # `coords: Tuple[Int, Int, Int]`, so narrowing it only adds a
                # widening cast back at the call.
                var k_row0 = Int(
                    k_operand.row_idx(UInt32(b), UInt32(it * Int32(BM_key)))
                )
                var k_dst = TileTensor[
                    dtype,
                    type_of(k_flat_layout),
                    address_space=AddressSpace.SHARED,
                ](k_smem + s * UInt32(k_elems), k_flat_layout)
                expect_bytes_pred(
                    k_full + s,
                    Int32(k_bytes + q_bytes) if with_q else Int32(k_bytes),
                    e,
                )
                k_tma.async_copy_3d_elect(k_dst, k_full[s], (0, 0, k_row0), e)
                comptime if with_q:
                    var q_dst = TileTensor[
                        dtype,
                        type_of(q_flat_layout),
                        address_space=AddressSpace.SHARED,
                    ](q_smem, q_flat_layout)
                    q_tma.async_copy_3d_elect(
                        q_dst,
                        k_full[s],
                        (0, 0, Int(start_of_seq + tok0) * num_heads),
                        e,
                    )

            # Prologue: fill the first NSTAGE stages (fresh, no k_empty wait).
            # `tile_begin` shifts only the tile ADDRESS -- the ring index/phase
            # sequence is a function of the count, so it is unshifted.
            #
            # Iteration 0 is peeled to carry the resident Q. Always taken:
            # `n_tiles_local >= 1` is guaranteed by the bail above, so
            # `n_prefetch >= 1`. Peeled rather than an `i == 0` test because `i` is
            # a runtime value and `with_q` must be comptime.
            issue_k[with_q=True](tile_begin, kp_state)
            kp_state.step()
            for i in range(Int32(1), n_prefetch):
                issue_k(tile_begin + i, kp_state)
                kp_state.step()
            # Refills: wait k_empty at that stage/phase (MMA done with the prior
            # occupant, tile i-NSTAGE) before reissuing.
            for i in range(n_prefetch, n_tiles_local):
                k_empty[kp_state.index()].wait(kp_state.phase())
                issue_k(tile_begin + i, kp_state)
                kp_state.step()
        else:
            # Idle warps 6-7, and the ONLY reason they are launched: they must
            # dealloc the SAME count as warps 4-5, because
            # `setmaxnreg.sync.aligned` is warpgroup-collective and a differing
            # operand within WG1 is UB. Drop `setmaxnreg` and these warps have no
            # purpose at all.
            comptime if _use_setmaxnreg[MMA_N]():
                warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()


@always_inline
def _prefill_smem_bytes[
    dtype: DType, depth: Int, BM_key: Int, MMA_N: Int
]() -> Int:
    """SMEM byte size for the prefill kernel (shared by launcher + kernel).

    Must be parameterized on `MMA_N`, not a constant: the launcher passes the
    result as `shared_mem_bytes`, so a value that disagrees with the kernel's own
    layout under-allocates and the q_scale region runs into the mbars.
    """
    comptime NSTAGE = _k_ring_stages[MMA_N]()
    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    comptime n_mbars = _n_mbars[MMA_N]()
    return (
        (q_elems + NSTAGE * k_elems) * size_of[Scalar[dtype]]()
        + MMA_N * size_of[Float32]()
        + n_mbars * size_of[SharedMemBarrier]()
        + size_of[UInt32]()
    )


# Route a shape here when a sequence spans at least this many token blocks: the
# token-block grid alone then fills the machine without a key split.
comptime _PREFILL_MIN_TOKEN_TILES = 16

# The other direction: too FEW token blocks to fill the grid, but a key range
# deep enough that splitting it does. This is decode / MTP-decode against a long
# cache, where the K-resident scorer degenerates to one CTA per key tile doing a
# single MMA -- it pays a full CTA prologue (TMEM alloc, mbar init, Q staging,
# k_scale gather) per 128 keys and never pipelines the K stream. Here one CTA
# instead streams `_KEY_TILES_PER_CTA` tiles through the ring behind a resident
# Q tile.
comptime _KEYSPLIT_MAX_TOKEN_TILES = 4
comptime _KEYSPLIT_MIN_KEY_TILES = 64
# Load-bearing beyond this route: the alternate N-tile's block-count clause reads
# `max(token_tiles, _KEYSPLIT_MAX_TOKEN_TILES)`, so at K=4 with N_ALT=3 only a
# `max_seq_len` in {3, 6, 9} can reach the narrower tile -- speculative widths only,
# which is the property every `MMA_N < 128` claim in this file is scoped on. Raising K
# widens that toward genuine prefill, so re-tuning it must re-check the alternate
# tile, not just the key split.


@always_inline
def fp8_index_score_sm100_prefill[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    *,
    VLStorageType: TensorStorage = PointerStorage[element_width=1],
    QSStorageType: TensorStorage = PointerStorage[element_width=1],
    OutStorageType: TensorStorage = PointerStorage[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[
        mut=False, DType.uint32, ..., Storage=VLStorageType
    ],
    q_s: TileTensor[mut=False, DType.float32, ..., Storage=QSStorageType],
    output: TileTensor[DType.float32, ..., Storage=OutStorageType],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    causal: Int,
    ctx: DeviceContext,
    out_row_begin: Int = 0,
) raises:
    """Enqueue the warp-specialized K-streaming prefill scorer into `output`.

    One CTA per (batch, token block, key part). A part streams its own window of
    the token block's causally-reachable keys; the windows are disjoint and carry
    no reduction, so there is no combine pass. `num_key_parts` bounds the split
    rather than fixing it -- a batch entry too shallow to feed that many parts
    narrows it in-kernel and the surplus CTAs retire before any collective.
    Called from `fp8_index_score_sm100` for both the many-token-block prefill
    route and the key-split decode/MTP route.

    `output` holds the global token rows `[out_row_begin, out_row_begin +
    output.dim[0]())`, so a caller that cannot afford the whole
    `total_seq_len x max_num_keys` score matrix can fill it a row-window at a
    time. The default covers every row and is the unwindowed launch.
    """
    # The window bounds the token blocks any entry can contribute, so a chunked
    # launch does not pay for a grid sized to the whole batch.
    var out_rows = Int(output.dim[0]())
    var token_blocks = ceildiv(min(max_seq_len, out_rows), N_TOKENS)
    # Split the key range over grid.z only when the (batch, token block) grid alone
    # leaves SMs idle -- splitting an already-full grid costs pipeline depth for
    # nothing (the K-resident scorer measured -29% doing exactly that). Target
    # `_KEY_TILES_PER_CTA` tiles per CTA, but take more parts if that is what it costs
    # to reach a wave at `_ctas_per_sm`.
    #
    # The wave-fill arm FLOORS: it wants the largest part count whose grid still fits
    # one wave, and `ceildiv` overshoots by construction (at base_ctas=64 it gave
    # 5 -> 320 CTAs = 2 waves where 4 -> 256 = 1 wave delivers the same tiles per CTA).
    #
    # The amortized arm is CAPPED at `_MAX_KEY_PART_WAVES` waves: `max_num_keys` is
    # METADATA, and a captured decode graph freezes it at its capture-time bound (1M
    # for a full-context GLM graph) while the live keys stay orders of magnitude
    # smaller, so a part count proportional to `key_tiles` turns that gap into empty
    # CTAs. Past a few waves the extra parts buy no parallelism even when the
    # bound matches the runtime key range -- capped parts just stream more tiles each, which the
    # K-ring amortizes better than more prologues would.
    comptime sm_count = ctx.default_device_info.sm_count
    comptime MMA_N = N_TOKENS * num_heads
    comptime CTAS_PER_SM = _ctas_per_sm[MMA_N]()
    var base_ctas = batch_size * token_blocks
    var key_tiles = ceildiv(max_num_keys, BM_key)
    var num_key_parts = 1
    if base_ctas < sm_count:
        var wave_parts = (CTAS_PER_SM * sm_count) // base_ctas
        num_key_parts = max(
            1,
            min(
                max(
                    min(
                        ceildiv(key_tiles, _KEY_TILES_PER_CTA),
                        _MAX_KEY_PART_WAVES * max(wave_parts, 1),
                    ),
                    wave_parts,
                ),
                key_tiles,
            ),
        )

    comptime kernel = _fp8_index_score_prefill_kernel_sm100[
        dtype,
        KOperand,
        KSOperand,
        type_of(valid_length.as_immut()).LayoutType,
        type_of(q_s).LayoutType,
        type_of(output).LayoutType,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
        VLStorageType=type_of(valid_length.as_immut()).Storage,
        QSStorageType=q_s.Storage,
        OutStorageType=output.Storage,
    ]
    # The `setmaxnreg` caps redistribute a fixed file: 65536 registers/SM shared
    # by `CTAS_PER_SM` CTAs, and every thread of a warpgroup holds that
    # warpgroup's cap. Counted per thread rather than per warpgroup so it stays
    # right when the producer is not a full warpgroup.
    comptime if _use_setmaxnreg[MMA_N]():
        comptime assert (
            _NUM_SOFTMAX_THREADS * _num_reg_consumer[MMA_N]()
            + WARP_SIZE * _prefill_prod_warps[MMA_N]() * _NUM_REG_PRODUCER
            <= 65536 // CTAS_PER_SM
        ), (
            "the warpgroup register caps must fit the per-SM register file at"
            " CTAS_PER_SM CTAs/SM"
        )
    comptime smem_bytes = _prefill_smem_bytes[dtype, depth, BM_key, MMA_N]()
    # Overrunning the per-SM budget does not fail the launch -- it just seats
    # fewer CTAs, so `CTAS_PER_SM` and the `minctasm` bound derived from it
    # would quietly become a lie. Fail at comptime instead. (~1KB/SM is reserved
    # by the driver, which is why the 1-CTA case caps at 227KB, not 228KB.)
    comptime assert (
        smem_bytes
        <= ctx.default_device_info.shared_memory_per_multiprocessor
        // CTAS_PER_SM
        - 1024
    ), (
        "prefill SMEM ("
        + String(smem_bytes)
        + " B) exceeds the budget for CTAS_PER_SM="
        + String(CTAS_PER_SM)
        + "; lower _k_ring_stages for MMA_N="
        + String(MMA_N)
    )
    # `max_num_keys` crosses the ABI as `Int32` and the kernel's whole key-index chain
    # is 32-bit signed, so the narrowing here is a silent truncation of a caller-supplied
    # `Int`. The bound reserves TWO key tiles of headroom rather than stopping at
    # `Int32.MAX` because the `k_scale` prefetch evaluates `key_local + BM_key` one tile
    # PAST the last real key; at `Int32.MAX` that wraps negative, passes the signed
    # `key < num_keys` guard, and `UInt32(key)` turns it into a ~2.1e9 pool row -- an OOB
    # read rather than the `0.0` the guard should produce.
    debug_assert[assert_mode="safe"](
        max_num_keys <= Int(Int32.MAX) - 2 * BM_key,
        (
            "fp8 index prefill: max_num_keys must leave two key tiles of"
            " headroom under Int32.MAX; the device-side key arithmetic is"
            " 32-bit signed"
        ),
    )
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        k_operand,
        ks_operand,
        valid_length.as_immut(),
        q_s,
        output,
        Int32(max_num_keys),
        Int32(causal),
        Int32(num_key_parts),
        Int32(out_row_begin),
        Int32(out_row_begin + out_rows),
        grid_dim=(
            batch_size,
            token_blocks,
            num_key_parts,
        ),
        block_dim=_prefill_nthreads[MMA_N](),
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        ),
    )
