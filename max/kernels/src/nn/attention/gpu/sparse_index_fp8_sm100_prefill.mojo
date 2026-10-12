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
  part streaming its own key window (`_clc_key_parts`). `num_key_parts` is a
  grid EXTENT, not the realized split: it is sized from the batch maximum, so on
  a ragged batch each item narrows it to what its own entry can feed
  (`_MIN_TILES_PER_PART`) and the surplus parts come up empty.
- CLC work stealing: a ctaid is a work item, not a CTA. A CTA walks a stream of
  items, claiming each next one with `clusterlaunchcontrol.try_cancel`, so an
  empty or finished item costs a cancel rather than a CTA prologue.

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
  warp 5 = TMA (Q slots, the deep K ring, and the CLC steals), warp 6 =
  q-scale transpose (interleaved order only). Warp 7 is idle; it exists only so
  `setmaxnreg` has a whole warpgroup to issue the `dec` from, which is what
  funds the consumer's 216. Role-to-role mbars (k_full/k_empty for the K ring,
  s_full/s_empty for the multi-stage S^T, q_full/qs_full/qs_ready/q_empty for
  the Q slots) replace the shipped kernel's per-iteration whole-CTA
  `named_barrier`.

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

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from max.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    create_tma_descriptor,
)
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.sync import (
    barrier,
    named_barrier,
    named_barrier_arrive,
    syncwarp,
)
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.primitives.cluster import (
    clusterlaunchcontrol_query_cancel_get_first_ctaid_v4,
    clusterlaunchcontrol_query_cancel_is_canceled,
    clusterlaunchcontrol_try_cancel,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from std.bit import next_power_of_two
from std.math import align_down, align_up, ceildiv, clamp, lcm
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    size_of,
)
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from layout import (
    Coord,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    UNKNOWN_VALUE,
    coord,
)
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    TMATensorTile,
    create_split_tma,
    create_tma_tile,
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
    relu_cvt_bf16x2,
    splitk_window,
    store_global_pred,
)
from nn.attention.mha_operand import MHAOperand
from kv_cache.types import flat_scale_window, scale_align_elems
from nn.attention.gpu.nvidia.sm100.attention import SM100_RESERVED_SMEM_BYTES


# Defined locally (not imported from `sparse_index_fp8_sm100`) so that file can
# route to this one without an import cycle; they must stay type-identical to
# its aliases or the `rebind` at its call site stops compiling.
comptime _INDEX_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime QTMATileT[
    dtype: DType, MMA_N: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, coord[MMA_N, 1, depth], _INDEX_SWIZZLE]
comptime KTMATileT[dtype: DType, BM_key: Int, depth: Int] = TMATensorTile[
    dtype,
    coord[1, BM_key, 1, depth],
]
# The k-scale ring's descriptor: a flat `1 x KS_BOX` window on the scale pool,
# unswizzled because one scalar per key has no depth to swizzle. A SECOND
# descriptor rather than a widening of `KTMATileT` because the scales live in
# their own pool with their own paging. `KS_BOX` is one 16-byte alignment unit
# WIDER than the key tile -- see `flat_scale_window`.
comptime KSTMATileT[ks_dtype: DType, KS_BOX: Int] = TMATensorTile[
    ks_dtype, coord[1, KS_BOX]
]
# The q-scales of one token block, `[N_TOKENS, num_heads]` f32 token-major, as
# `q_s` stores them.
comptime QSTMATileT[N_TOKENS: Int, num_heads: Int] = TMATensorTile[
    DType.float32, coord[N_TOKENS, num_heads]
]


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
comptime _NUM_SOFTMAX_THREADS = 128

# Hardware barrier id for the consumer warpgroup's two private syncs (the q_scale
# publish in the prologue, the TMEM drain before dealloc). Id 0 belongs to the
# whole-CTA `barrier()`, which `named_barrier` also defaults to, so a
# warpgroup-scoped sync left on the default would share it.
comptime _CONSUMER_BAR: Int32 = 1

# ===== Measurement knobs =====
# Every knob below defaults to the value this file already folded, so an
# undefined build is the shipped kernel. That is the A/B's null arm and it is
# checked, not assumed: `--emit asm` with no define must match `origin/main`
# byte for byte.
#
# A knob must configure the CHOSEN kernel, never change WHICH kernel is chosen.
# `_ctas_per_sm` takes a residency-class parameter for exactly that reason --
# see its comment and the router's call sites.

# Seats the 1-CTA/SM arm at tiles that would otherwise hold 2. Only takes
# effect at `MMA_N <= 128`, the widest tile two CTAs' TMEM fits.
comptime _CTAS_FORCE = get_defined_int["FP8_INDEX_CTAS_PER_SM", 0]()
# Producer warps. 0 derives to 4, which `setmaxnreg` requires (it is
# warpgroup-collective).
comptime _PROD_WARPS_FORCE = get_defined_int["FP8_INDEX_PROD_WARPS", 0]()
# Forces the `setmaxnreg.inc` operand for a paired sweep, still clamped by
# `reg_cap` so a knob cannot break the register-file identity the launcher
# asserts. 0 leaves the measured table in charge.
comptime _CONS_REG_FORCE = get_defined_int["FP8_INDEX_CONS_REG", 0]()
# Forces the producer floor, which is the term `reg_cap` is derived against.
comptime _PROD_REG_FORCE = get_defined_int["FP8_INDEX_PROD_REG", 0]()
# Forces the K ring depth. 0 derives it from the carveout. Ring depth is
# NON-MONOTONE -- deriving it from a bigger budget measured WORSE in both
# regimes -- so treat a forced value as a sweep point, not an improvement.
comptime _K_STAGES_FORCE = get_defined_int["FP8_INDEX_K_STAGES", 0]()

# CLC work stealing. A CTA walks a stream of work items through a depth-2 ring
# of Q slots, each carrying its item's Q tile, its q-scales (by TMA) and the
# item's metadata, so a later item needs no fresh prologue. The load warp finds
# the next item by cancelling a not-yet-launched ctaid with
# `clusterlaunchcontrol.try_cancel` once the current item's last K load is
# issued, so a CTA claims work only when it is nearly free. A CTA whose own
# item is empty steals one instead of exiting, so an empty ctaid costs a CLC
# round trip rather than a CTA launch.
comptime _Q_SLOTS = 2
# One item's metadata, as `Int32` lanes of one 32-byte record. A zero
# `_MD_N_TILES` is the end-of-stream sentinel.
comptime _MD_WORDS = 8
# The CLC scratch after the records: the 16-byte response, then the entry
# steal's outcome padded to the next 16-byte unit.
comptime _CLC_RESP_WORDS = size_of[UInt128]() // size_of[Int32]()
comptime _CLC_SCRATCH_WORDS = 2 * _CLC_RESP_WORDS
comptime _MD_TOK0 = 0
comptime _MD_TOK_HI = 1
comptime _MD_SEQ_LEN = 2
comptime _MD_NUM_KEYS = 3
comptime _MD_OUT_ROW0 = 4
comptime _MD_TILE_BEGIN = 5
comptime _MD_N_TILES = 6
comptime _MD_KS_OFF = 7


# Producer register floor, applied to ALL FOUR of WG1's warps -- MMA, TMA and the
# two idle ones -- because `setmaxnreg.sync.aligned` is warpgroup-granular: every
# warp of a warpgroup must pass the SAME operand, so a per-warp split is UB. The
# consumer claims the rest so its TMEM->register fragment does not spill.
# Spilling is NOT monotonic in the cap, so treat this as a swept value.
# 216 + 40 = 2 * 128 returns the producers' share to the consumer exactly, which
# is what makes `TRY_ALLOC` succeed; the launcher asserts that identity.
comptime _NUM_REG_PRODUCER = 40

# S^T TMEM ring depth. 2 lets the consumer read stage `it` while the MMA writes
# `it+1`, which is what makes deferring the epilogue's WAR fence affordable.
comptime _S_TMEM_STAGES = 2

# Column chunk for the TMEM->register drain. Must stay <= 64 or ptxas runs out of
# destination registers for the `tcgen05.ld`; it must also divide MMA_N and be a
# multiple of 4. The decode twin holds a same-named constant at its own 16;
# only this one is define-driven.
comptime _EPILOGUE_CHUNK = get_defined_int["FP8_INDEX_EPILOGUE_CHUNK", 32]()

# Columns folded per accumulator step. `SIMD[f32, W].fma` lowers to `W / 2`
# INDEPENDENT `fma.rn.f32x2` on disjoint lanes, for `W - 2` registers and NO
# extra `tcgen05.ld` -- paying only a wider per-token closing reduce.
#
# Legacy Q order: 4 is the measured optimum. On B200 at MMA_N=128, nh=32, min
# over reps, W=4 is -1.5% to -1.75% on prefill and -1.6% on decode, while W=8 is
# neutral (+0.1%) -- its 12 extra `add.f32` buy latency relief the second chain
# already took. So 4 is a crossover, not a point on a slope.
#
# Interleaved Q order: 2. W=4 bought ILP (a second chain per token) at the price
# of the closing `add.f32`s. The interleaved layout supplies the ILP itself --
# adjacent columns are independent tokens -- so the width no longer buys
# anything and 2 is the cheaper spelling: 0.24% faster than W=4 on prefill,
# every cell, against a same-kernel null that agreed with itself to 0.03%.
# That holds only while the tile has enough tokens to fill the lanes: at T=2
# (nh=64) W=2 is ONE dependent chain of 64 FMAs per key tile, which cost 3%
# against legacy order. W=4 (2 chains) and W=8 (4 chains) both win back ~4%;
# 4 measured marginally ahead with an f32 score buffer and 8 with bf16.
#
# 0 derives per tile (`_acc_width`); a positive value forces every tile.
comptime _ACC_WIDTH_FORCE = get_defined_int["FP8_INDEX_ACC_WIDTH", 0]()

# Resident-Q row order. Legacy stages row `t * nh + h` (token-major), so a
# `tcgen05.ld` chunk holds one token and its fold is one chain that closes with a
# `reduce_add` at the token's last column. Interleaved stages row `h * T + t`
# (head-major): adjacent columns are different tokens, so every chunk feeds all
# T token sums at once with one fragment live, and the per-token closing reduce
# shrinks from `ACCW - 1` adds to `ACCW / T - 1` (none at ACCW <= T).
#
# Lane `L` of the accumulator is token `L % T` when it spans `lcm(ACCW, T)`
# lanes, so any T works; the 3-token decode tile carries 6 lanes and pays one
# closing add per token, measured 0.87% faster than legacy on every decode and
# ragged cell (B200, 16 cells). ACCW must stay a power of two for the q-scale
# load alignment. -1 derives (interleave wherever legal), 0 forces legacy, 1
# forces interleave.
comptime _Q_INTERLEAVE_FORCE = get_defined_int["FP8_INDEX_Q_INTERLEAVE", -1]()

# The accumulator width the interleaved order takes; see `_ACC_WIDTH_FORCE`.
comptime _INTERLEAVED_ACCW = _ACC_WIDTH_FORCE if _ACC_WIDTH_FORCE > 0 else 2


def _q_interleave_legal() -> Bool:
    return _INTERLEAVED_ACCW.is_power_of_two()


@inline(.always)
def _q_tma_coords[
    interleave: Bool, num_heads: Int
](tok: Int) -> Tuple[Int, Int, Int]:
    """Returns the Q TMA coordinates (CUDA order) of the tile's first token."""
    comptime if interleave:
        return (0, tok, 0)
    return (0, 0, tok * num_heads)


def _q_interleave[n_tokens: Int]() -> Bool:
    comptime if _Q_INTERLEAVE_FORCE == 0:
        return False
    comptime if _Q_INTERLEAVE_FORCE == 1:
        return _q_interleave_legal()
    # Every token's sum stays live to the tile's last group, so the lane count
    # is the live-register cost: at nh=8/4 (T=16/32) it spills 0 -> 148 B and
    # 100 -> 204 B. Production runs the indexer replicated at nh=32/64, so the
    # TP-sharded counts keep the legacy order.
    return _q_interleave_legal() and lcm(_INTERLEAVED_ACCW, n_tokens) <= 6


def _acc_width[n_tokens: Int, out_dtype: DType]() -> Int:
    comptime if _ACC_WIDTH_FORCE > 0:
        return _ACC_WIDTH_FORCE
    comptime if not _q_interleave[n_tokens]():
        return 4
    comptime if n_tokens == 2:
        return 4 if size_of[out_dtype]() == 4 else 8
    return _INTERLEAVED_ACCW


@inline(.always)
def _prefill_prod_warps() -> Int:
    comptime if _PROD_WARPS_FORCE > 0:
        return _PROD_WARPS_FORCE
    return 4


# Element type of the epilogue fold -- the relu'd scores, the q-scales held in
# `qs_reg`, and the running per-token accumulator.
#
# The relu is the fold's dominant cost and f32 cannot make it cheaper: PTX has
# no `max.f32x2` at any ISA version (ptxas 12.9 for sm_100a rejects it as
# "unexpected instruction types"), so `SIMD[f32, W].max` lowers to one FMNMX per
# column. At bf16 the clamp is a MODIFIER on the narrowing convert, so a lane
# pair costs `1 F2FP.RELU + 1 HFMA2` against `2 FMNMX + 1 FFMA2`, and staging
# `qs_smem` in the same type halves the `LDS` bytes per group and drops the
# `MMA_N / 2` prologue converts. Registers fall 27% at the nh=32 MMA_N=128 tile,
# counted in equivalents (`b32 + 2*b64`, an f32 pair being one `.b64` and a bf16
# pair one `.b32`), with no spill either way.
#
# The cost is accuracy: every accumulate step rounds to an 8-bit mantissa, so a
# `num_heads`-long chain drifts 0.5-0.8% against an f32 fold, moving
# tie-adjacent keys. That is a RANKING decision, which a score tolerance cannot
# judge -- grade it against an exact host top-k before touching the fold
# arithmetic.
comptime _FOLD_DTYPE = get_defined_dtype[
    "FP8_INDEX_FOLD_DTYPE", DType.bfloat16
]()


# The rest of the pipeline is derived from the N-tile, because TMEM decides how
# many CTAs can be co-resident.
#
# The S^T stages cost `s_stages * align_up(MMA_N, 32)` of the SM's 512 TMEM
# columns, and `tcgen05.alloc` only accepts a POWER-OF-TWO column count. So
# MMA_N=128 costs 256 (exactly half the SM, hence 2 CTAs/SM), while MMA_N=192
# uses 384 and must ALLOCATE 512 -- the whole SM, hence 1 CTA/SM.
#
# Residency is therefore a step function of the N-tile, and crossing 128 columns
# costs half the consumer warps. Measured on a batch-8 107K-key MTP step,
# MMA_N=192 cut 16.5% of the instructions and came out +2.5% in cycles -- a wash.
# Hence prefer a DIVISOR of the token count over a multiple: 3 tokens at nh=32
# (96 columns) takes the same exactness as 6 while staying on the 2-CTA/SM side.
#
# `force` selects the RESIDENCY CLASS, and the two callers want different ones:
#
#   -1  read `_CTAS_FORCE` (the default; what the kernel body configures to)
#    0  the PRODUCTION class, knob ignored
#
# The router gates its wide and alternate tiles on COMPARISONS between two
# tiles' residencies. Read through the knob, `-D FP8_INDEX_CTAS_PER_SM=1` makes
# `_ctas_per_sm[192] < _ctas_per_sm[128]` into `1 < 1` and silently DROPS the
# wide tile -- so the arm being measured would not ROUTE like the arm it is
# compared against. Router sites therefore pass `0`. A measurement lever must
# configure the chosen kernel, never change which kernel is chosen.
@inline(.always)
def _ctas_per_sm[MMA_N: Int, force: Int = -1]() -> Int:
    comptime f = _CTAS_FORCE if force == -1 else force
    comptime if f > 0 and MMA_N <= 128:
        return f
    return 2 if MMA_N <= 128 else 1


# Keys a slot COSTS, padded so every slot base keeps the 128-byte alignment TMA
# needs of a shared-memory destination.
@inline(.always)
def _ks_slot_elems[ks_dtype: DType, BM_key: Int]() -> Int:
    comptime esize = size_of[Scalar[ks_dtype]]()
    return align_up(flat_scale_window[ks_dtype, BM_key]() * esize, 128) // esize


# ===== Consumer register cap =====
# `reg_consumer` IS the budget ptxas allocates the consumer region against -- not
# a formality above the launch allocation. A one-sided dial: raise it and the
# consumer gets the room, lower it and it spills. The landscape is a VALLEY, not
# a slope -- more registers buy a more aggressive schedule that then does not fit
# -- so re-sweep rather than extrapolate, and note the cells move with any
# codegen change while the WINDOWS are the stable part.
#
# Measured on B200 at nh=32, 128 columns (in a since-removed 2-warpgroup arm),
# spill-free 192 and 208 differed by 3.5%, so the cap buys schedule (earlier
# `LDTM.x32` issue), not just the absence of spill. Spill predicts a regression;
# its absence predicts nothing, so time a cell before retuning it.


struct IndexPrefillConfig[
    dtype: DType,
    ks_dtype: DType,
    depth: Int,
    BM_key: Int,
    mma_n: Int,
    num_heads: Int,
](TrivialRegisterPassable):
    """Every derived quantity of one prefill instantiation, in one object.

    This is the `FA4Config` pattern from `nvidia/sm100/attention.mojo`, adopted
    for the same reason it exists there: the pipeline's shape, its SMEM
    accounting and its register accounting are ONE derivation, and splitting
    them across free helpers is what lets two of them disagree.

    What the split cost here, concretely: the fixed-SMEM term fed only the
    ring-depth derivation while the launcher passed a separate total, so the two
    had to be edited together by hand -- and a one-sided edit under-allocates
    the launch rather than failing it. Both are now the same accumulation:
    `smem_used == fixed_smem_bytes + k_stages * k_stage_bytes` identically.

    Residency stays OUTSIDE the struct, in `_ctas_per_sm`, because the router
    asks what residency a tile would take before it has a dtype or depth to
    build a config with.

    Parameters:
        dtype: Q/K element type.
        ks_dtype: K-scale element type.
        depth: Head dimension.
        BM_key: Keys per K tile.
        mma_n: The N tile, `n_tokens * num_heads`.
        num_heads: Index heads.
    """

    var ctas_per_sm: Int
    var prod_warps: Int
    var nthreads: Int
    var reg_cap: Int
    var reg_consumer: Int
    var reg_producer: Int
    var static_reg_budget: Int
    var k_stages: Int
    var n_mbars: Int
    var carveout: Int
    var fixed_smem_bytes: Int
    var k_stage_bytes: Int
    var smem_used: Int

    comptime n_tokens = Self.mma_n // Self.num_heads
    # One S^T stage's TMEM columns. `tcgen05.alloc` takes a power of two, so the
    # allocation rounds up once, at the ring rather than per stage.
    comptime s_cols = align_up(Self.mma_n, 32)
    # Tokens one item owns, and the grid.y step. The kernel body and the
    # launcher's grid both read this, so they cannot disagree about which
    # blocks an item covers.
    comptime cta_token_stride = Self.n_tokens
    comptime tmem_cols = next_power_of_two(_S_TMEM_STAGES * Self.s_cols)
    # Barriers priced by `fixed_smem_bytes`: the S^T pair per stage, `q_full`,
    # `qs_full`, `q_empty` and `qs_ready` per Q slot, and `clc_full` for the CLC
    # response. `qs_ready` is priced even in token-major order, which never
    # arms it: 16 bytes buys a config that need not know the order.
    # Named once because `n_mbars` counts the SAME barriers -- pricing a term in
    # one and not the other under-sizes the launch by exactly that many
    # barriers, which lands `ptr_tmem` past the allocation rather than failing
    # anything.
    comptime fixed_mbars = 2 * _S_TMEM_STAGES + 4 * _Q_SLOTS + 1
    # Barriers priced by `k_stage_bytes`, per K ring slot: `k_full`/`k_empty`.
    comptime stage_mbars = 2

    def __init__(out self):
        self.ctas_per_sm = _ctas_per_sm[Self.mma_n]()
        # Four producer warps: two do the work (MMA, TMA) and two are idle, but
        # `setmaxnreg.sync.aligned` is warpgroup-collective, so an asymmetric
        # cap needs WG1 whole. Without it the spill goes 8 B -> 216 B with 32
        # `LDL` back in the tile loop.
        self.prod_warps = _prefill_prod_warps()
        self.nthreads = _NUM_SOFTMAX_THREADS + WARP_SIZE * self.prod_warps
        # The per-thread LAUNCH allocation, which the driver divides into the
        # register file for occupancy. NOT the ceiling on the consumer region --
        # `reg_consumer` is. Must be computed BEFORE `reg_cap`, which is bounded
        # by the pool it implies. Allocation is granular in 8, so round down.
        self.static_reg_budget = align_down(
            65536 // (self.nthreads * self.ctas_per_sm), 8
        )
        # Producer floor, held by every warp of the producer warpgroup.
        if _PROD_REG_FORCE > 0:
            self.reg_producer = _PROD_REG_FORCE
        else:
            self.reg_producer = _NUM_REG_PRODUCER
        # `setmaxnreg` redistributes the CTA's pool: every consumer thread holds
        # the consumer cap and all four producer warps hold `reg_producer`.
        #
        # That pool is `nthreads * static_reg_budget`, NOT `65536 /
        # ctas_per_sm`. The launch allocation rounds DOWN to the 8-register
        # granule, so up to 1023 registers of the file are never handed to the
        # CTA at all. Dividing the architectural size instead returns a cap the
        # pool cannot serve, and `setmaxnreg.inc` then spins forever in
        # `USETMAXREG.TRY_ALLOC` rather than failing.
        self.reg_cap = align_down(
            (
                self.nthreads * self.static_reg_budget
                - WARP_SIZE * self.prod_warps * self.reg_producer
            )
            // _NUM_SOFTMAX_THREADS,
            8,
        )
        # Keyed on the head count because the measured table forces it to be: at
        # 128 columns nh=32 needs 200 or below and nh=4 needs 208 or above, and
        # no value is clean for both. See the consumer-register note above for
        # why taking the cap is not the default move.
        if _CONS_REG_FORCE > 0:
            self.reg_consumer = min(_CONS_REG_FORCE, self.reg_cap)
        elif Self.mma_n == 128 and Self.num_heads == 32:
            self.reg_consumer = min(200, self.reg_cap)
        else:
            self.reg_consumer = min(216, self.reg_cap)

        # SMEM, accumulated ONCE. The ring depth is what the carveout still
        # affords after the fixed regions, and the total is the same
        # accumulation carried to the end -- so the launcher's
        # `shared_mem_bytes` and the depth derivation cannot disagree.
        self.carveout = (
            B200.shared_memory_per_multiprocessor // self.ctas_per_sm
            - SM100_RESERVED_SMEM_BYTES
        )
        # Per Q slot: the resident Q, its f32 q-scales and its metadata
        # record. Then the CLC scratch, the fixed barriers and the TMEM pointer.
        self.fixed_smem_bytes = (
            _Q_SLOTS * Self.mma_n * Self.depth * size_of[Scalar[Self.dtype]]()
            + _Q_SLOTS * Self.mma_n * size_of[Float32]()
            + _Q_SLOTS * _MD_WORDS * size_of[Int32]()
            + _CLC_SCRATCH_WORDS * size_of[Int32]()
            + Self.fixed_mbars * size_of[SharedMemBarrier]()
            + size_of[UInt32]()
        )
        # One K ring slot: the tile, its per-key scales, and the
        # `k_full`/`k_empty` pair that guards it. The barriers and the scales
        # belong in the SLOT price, not in a separate fixed term -- they scale
        # with the stage count, so folding them in is what makes the division
        # exact rather than an estimate that then needs a fudge factor. The
        # scale region is what lets the consumer read `k_scale` with an `LDS`
        # instead of a two-hop dependent `LDG`; it rides the slot's own `k_full`
        # barrier, so it cannot be priced anywhere else.
        #
        self.k_stage_bytes = (
            Self.BM_key * Self.depth * size_of[Scalar[Self.dtype]]()
            + _ks_slot_elems[Self.ks_dtype, Self.BM_key]()
            * size_of[Scalar[Self.ks_dtype]]()
            + Self.stage_mbars * size_of[SharedMemBarrier]()
        )
        # `-D FP8_INDEX_K_STAGES` forces the depth. Ring depth is NON-MONOTONE
        # -- deriving it from a larger budget measured WORSE in both regimes --
        # so a forced value is a sweep point, not an improvement. Still clamped
        # by what the carveout affords, so a knob cannot over-allocate SMEM.
        var derived_stages = (
            self.carveout - self.fixed_smem_bytes
        ) // self.k_stage_bytes
        if _K_STAGES_FORCE > 0:
            self.k_stages = min(_K_STAGES_FORCE, derived_stages)
        else:
            self.k_stages = derived_stages
        # Both counts come from the same two comptime members the SMEM terms
        # price against, so the layout and the byte total cannot disagree.
        self.n_mbars = self.k_stages * Self.stage_mbars + Self.fixed_mbars
        self.smem_used = (
            self.fixed_smem_bytes + self.k_stages * self.k_stage_bytes
        )


# The key split. Stealing makes a part cheap -- an empty or short one costs a
# cancel, not a CTA launch -- so the split is sized for load balance rather
# than against launch count. Measured on B200 against a since-removed
# one-item-per-CTA policy that streamed 16 tiles a part (b8/b16, nh=32):
#
# * Split-shaped launches target `_CLC_KEY_TILES_PER_PART` tiles a part, at
#   least one wave and at most `_CLC_MAX_KEY_PART_WAVES`. Exponential-depth
#   decode was fastest at 10-18 tiles a part (0.79-0.84x the fixed policy,
#   against 0.82-0.88x at the fixed policy's count). Uniform depth wants one
#   wave, but an extra few waves cost it only ~5%.
# * Prefill grids, which the fixed policy never split, split once an item
#   would stream more than `_CLC_PREFILL_TILES_PER_PART` tiles, up to
#   `_CLC_PREFILL_MAX_PARTS`. A deep ragged cache otherwise leaves its longest
#   items as the tail: 4 parts took b8 s512 c8k from 1.02x to 0.84x the fixed
#   policy. Shallow prefill stays unsplit, where 2-8 parts measured 5-40%
#   slower.
#
# `max_num_keys` is METADATA: a captured decode graph freezes it at its
# capture-time bound while the live keys stay far smaller, which is why the
# split-shaped part count is capped in waves rather than proportional to
# `key_tiles`.
comptime _CLC_KEY_TILES_PER_PART = 12
comptime _CLC_MAX_KEY_PART_WAVES = 8
comptime _CLC_PREFILL_TILES_PER_PART = 32
comptime _CLC_PREFILL_MAX_PARTS = 4


def _clc_key_parts(split_shape: Bool, key_tiles: Int, wave_parts: Int) -> Int:
    """Returns the key-part count for a launch.

    Args:
        split_shape: Whether the launch is decode-shaped or under a wave.
        key_tiles: Key tiles of the deepest entry, from `max_num_keys`.
        wave_parts: Parts that fill one wave with the (batch, token block) grid.

    Returns:
        The part count, in `[1, key_tiles]`.
    """
    var parts: Int
    if split_shape:
        var wave = max(wave_parts, 1)
        parts = clamp(
            ceildiv(key_tiles, _CLC_KEY_TILES_PER_PART),
            wave,
            _CLC_MAX_KEY_PART_WAVES * wave,
        )
    else:
        parts = clamp(
            key_tiles // _CLC_PREFILL_TILES_PER_PART, 1, _CLC_PREFILL_MAX_PARTS
        )
    return clamp(parts, 1, max(key_tiles, 1))


# Floor on the key tiles a single grid.z part may own, applied INSIDE the kernel
# against the CTA's own per-entry tile count. The launcher can only size the part
# count from `max_num_keys` (the batch maximum), so on a ragged batch every entry
# but the deepest is over-split; this is what stops a shallow entry degenerating
# into 1-tile CTAs. Uniform batches are unaffected -- there the launcher's own
# per-part tile count already clears this floor.
comptime _MIN_TILES_PER_PART = 4


@fieldwise_init
struct _Item(TrivialRegisterPassable):
    """One (batch entry, token block, key part) work item, decoded.

    `n_tiles <= 0` is an empty item: a token block past the entry or the row
    window, a surplus key part, or a causal bound that left no keys.
    """

    var start_of_seq: Int32
    var seq_len: Int32
    var tok0: Int32
    var tok_hi: Int32
    var out_row0: Int32
    var num_keys: Int32
    var tile_begin: Int32
    var n_tiles: Int32


@inline(.always)
def _decode_item[
    KOperand: MHAOperand,
    //,
    BM_key: Int,
    cta_token_stride: Int,
    kpool: Int,
    _is_cache_length_accurate: Bool,
](
    b: Int,
    token_block: Int32,
    key_part: UInt32,
    valid_length: TileTensor[mut=False, .uint32, ...],
    k_operand: KOperand,
    max_num_keys: Int32,
    causal: Int32,
    num_key_parts: Int32,
    out_row_begin: Int32,
    out_row_end: Int32,
) -> _Item:
    """Decodes the work item at grid coordinates `(b, token_block, key_part)`.

    Every input is CTA-uniform, so every role derives the same item and the
    same trip count from it. That is load-bearing: windowing one role alone
    unbalances the k/s mbar handshakes and HANGS.
    """
    var start_of_seq = Int32(valid_length[b])
    var end_of_seq = Int32(valid_length[b + 1])
    var seq_len = end_of_seq - start_of_seq
    # The launch owns global token rows `[out_row_begin, out_row_end)` and
    # writes them to `output` rows `[0, out_row_end - out_row_begin)`. The
    # caller chunks that window to bound the score buffer; unchunked it is
    # `[0, total_seq_len)` and everything below reduces to the unwindowed form.
    #
    # Clamping to the window here (rather than predicating the store) is what
    # keeps the epilogue free: token blocks are indexed from `tok_lo`, so no
    # block straddles a chunk boundary, no token is scored twice, and the
    # store's liveness test just swaps `seq_len` for `tok_hi`. `seq_len` itself
    # stays the TRUE sequence length -- the causal bound is an absolute
    # position and must not see the window.
    var tok_lo = max(Int32(0), out_row_begin - start_of_seq)
    var tok_hi = min(seq_len, out_row_end - start_of_seq)
    # The item's first token.
    var tok0 = tok_lo + token_block * Int32(cta_token_stride)
    # Folded once here so the store's row arithmetic is a single add.
    var out_row0 = start_of_seq - out_row_begin
    var num_keys = Int32(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len
    # `num_keys` counts tokens, for the causal bounds below. `num_rows` counts
    # what the cache holds, one pooled key per `kpool` tokens. The host bounds
    # `max_num_keys`, but the per-entry count is device data the host never
    # sees, and on the ragged path it rests on a caller contract, so check it
    # here, in cache rows. Keep the default assert mode: a "safe" assert in
    # this kernel pulls in `vprintf`, which inflates the emitted PTX and
    # perturbs register allocation.
    var num_rows = num_keys // Int32(kpool)
    debug_assert(
        num_rows <= max_num_keys,
        "fp8 index prefill: per-entry candidate rows exceed max_num_keys",
    )
    # A token block past the sequence -- or outside the row window -- produces
    # no output (the caller's -inf fill covers those rows).
    if tok0 >= tok_hi or seq_len <= 0:
        return _Item(
            start_of_seq, seq_len, tok0, tok_hi, out_row0, num_keys, 0, 0
        )
    # Keys the item must stream: bounded by the deepest LIVE token under the
    # causal mask, the block's last live token. Each token still gets its own per-key guard in the epilogue; this only
    # trims the triangle a zero-prefix fresh prefill leaves off the end.
    var last_tok = min(tok0 + Int32(cta_token_stride), tok_hi) - 1
    var block_key_bound = (
        num_keys - (seq_len - 1 - last_tok) * causal
    ) // Int32(kpool)
    # `block_key_bound` can be <= 0 (a causal bound that trims the whole
    # block), so the SUBTRACTION above stays signed. The `ceildiv` does not:
    # `SIMD.__ceildiv__` branches at COMPTIME on the dtype -- signed lowers to
    # `-(x // -d)` with a ~9-instruction correction chain, unsigned to an add
    # and a shift. So the clamp has to sit ABOVE the ceildiv.
    var n_key_tiles = ceildiv(
        UInt32(max(block_key_bound, Int32(0))), UInt32(BM_key)
    )
    # grid.z splits that tile range across items. Scores carry no cross-key
    # reduction, and a thread's store address is `global_token * max_num_keys
    # + it * BM_key + row`, so disjoint tile windows write disjoint elements --
    # no combine pass and no workspace. `splitk_window` front-loads, so only
    # trailing parts come up empty.
    #
    # The launcher sizes `num_key_parts` from `max_num_keys`, a batch MAXIMUM,
    # so on a ragged batch it over-splits every entry but the deepest. The
    # item therefore narrows the part count to what its OWN tile count can
    # feed, and the surplus parts come up empty.
    var p_eff = clamp(
        ceildiv(n_key_tiles, UInt32(_MIN_TILES_PER_PART)),
        UInt32(1),
        UInt32(num_key_parts),
    )
    if key_part >= p_eff:
        return _Item(
            start_of_seq, seq_len, tok0, tok_hi, out_row0, num_keys, 0, 0
        )
    var win = splitk_window(n_key_tiles, p_eff, key_part)
    var tile_begin = Int32(win[0])
    return _Item(
        start_of_seq,
        seq_len,
        tok0,
        tok_hi,
        out_row0,
        num_keys,
        tile_begin,
        Int32(win[1]) - tile_begin,
    )


@__name(t"fp8_index_score_prefill_sm100_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(ks_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(qs_tma, `nvvm.grid_constant`)
# Cap the launch register count so the config's thread count fits at its
# `ctas_per_sm` (maxntid + minctasm), mirroring MSA and FA4: without it the
# launch requests more than `65536 / nthreads` regs/thread and the warpgroup
# reg-alloc/dealloc below has no reserved pool to redistribute.
#
# Spelled as a whole config here rather than as two helper calls because these
# two numbers are one decision: `minctasm` sizes the pool that `setmaxnreg`
# redistributes, and the body's caps are derived from the same object.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            IndexPrefillConfig[
                dtype,
                KSOperand.dtype,
                depth,
                BM_key,
                N_TOKENS * num_heads,
                num_heads,
            ]().nthreads
        )
    )
)
# Read off the config, never spelled inline: a stale literal here is not a perf
# bug but a HANG -- `setmaxnreg.inc` spins forever when the caps it is asked for
# exceed the pool this decorator sized.
@__llvm_metadata(
    `nvvm.minctasm`=SIMDLength(
        IndexPrefillConfig[
            dtype,
            KSOperand.dtype,
            depth,
            BM_key,
            N_TOKENS * num_heads,
            num_heads,
        ]().ctas_per_sm
    )
)
def _fp8_index_score_prefill_kernel_sm100[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    out_dtype: DType,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    kpool: Int = 1,
    *,
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    ks_tma: KSTMATileT[
        KSOperand.dtype, flat_scale_window[KSOperand.dtype, BM_key]()
    ],
    qs_tma: QSTMATileT[N_TOKENS, num_heads],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[.uint32, VLLT, ImmutAnyOrigin, Engine=VLEngine],
    q_s: TileTensor[.float32, QSLT, ImmutAnyOrigin, Engine=QSEngine],
    output: TileTensor[out_dtype, OutLT, MutAnyOrigin, Engine=OutEngine],
    max_num_keys_dev: Int32,
    causal_dev: Int32,
    num_key_parts_dev: Int32,
    out_row_begin_dev: Int32,
    out_row_end_dev: Int32,
):
    comptime assert valid_length.flat_rank == 1
    comptime MMA_N = N_TOKENS * num_heads
    # The whole derived pipeline shape, in one object. Every stage count,
    # register cap, SMEM term and epilogue knob below reads off this -- the
    # launcher builds the identical one from the identical parameters, which is
    # what makes `shared_mem_bytes` and the body's layout the same number by
    # construction rather than by matching edits.
    comptime CFG = IndexPrefillConfig[
        dtype,
        KSOperand.dtype,
        depth,
        BM_key,
        N_TOKENS * num_heads,
        num_heads,
    ]()
    # Two producer warps are the floor: the MMA and TMA roles are separate warps
    # because each spins on its own mbarrier.
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
    # The accumulator stays f32 whatever the output is; only the final store
    # converts. bf16 halves the score buffer and its write traffic, and shares
    # f32's exponent range so nothing finite clips.
    comptime assert out_dtype in (
        DType.float32,
        DType.bfloat16,
    ), "the score buffer must be f32 or bf16; got " + String(out_dtype)
    # Fold element type, independent of `out_dtype` -- the store converts from
    # whatever the fold produced.
    comptime FT = _FOLD_DTYPE
    comptime assert FT in (
        DType.float32,
        DType.bfloat16,
    ), "the epilogue fold must be f32 or bf16; got " + String(FT)

    comptime SW = _INDEX_SWIZZLE
    comptime KS_DTYPE = KSOperand.dtype
    comptime KS_ALIGN = scale_align_elems[KS_DTYPE]()
    comptime KS_BOX = flat_scale_window[KS_DTYPE, BM_key]()
    comptime KS_SLOT = _ks_slot_elems[KS_DTYPE, BM_key]()
    comptime NSTAGE = CFG.k_stages
    comptime N_S = _S_TMEM_STAGES
    comptime CTAS_PER_SM = _ctas_per_sm[MMA_N]()
    # The `nvvm.minctasm` decorator reads the same helper; keep them in
    # lockstep -- a mismatch sizes the register pool for the wrong CTA count,
    # which hangs `setmaxnreg.inc` rather than merely costing occupancy.
    # Compared against the CONFIG, not against a literal: the config is what
    # the decorator above reads, so this is the actual lockstep invariant and
    # it holds under a residency knob. A literal here would instead FORBID the
    # forced arm, which is the one being measured.
    comptime assert CTAS_PER_SM == CFG.ctas_per_sm, (
        "the body's residency must match what `nvvm.minctasm` stamped; got"
        " CTAS_PER_SM="
        + String(CTAS_PER_SM)
        + " vs config "
        + String(CFG.ctas_per_sm)
    )
    # At most 128 columns: a tile above 128 takes the whole SM and would run at
    # half the consumer warps, and the q-scale hoist (`mma_n` registers) stops
    # fitting. A 192-column arm existed and was MEASURED SLOWER than splitting
    # the same step across two narrow blocks (B200, 2026-08-31: 1.12x on
    # uniform decode, 1.20x once entry depths are ragged), and was removed.
    # Assert the premise rather than deriving a count from it: a wider tile
    # added later must fail to BUILD here.
    comptime assert MMA_N <= 128, (
        "a tile above 128 columns takes the whole SM and runs at half the"
        " consumer warps, and its q-scales no longer fit in registers. Got"
        " MMA_N="
        + String(MMA_N)
    )
    comptime CONS_WARPS = 4
    comptime CONS_THREADS = _NUM_SOFTMAX_THREADS
    comptime CTA_TOKEN_STRIDE = CFG.cta_token_stride
    # K ring depth. A PIPELINING floor, not a deadlock guard: the wait graph is
    # acyclic at any depth (the consumer only ever waits `s_full`; it arrives on
    # `k_empty` and `s_empty` and waits neither), so below the floor the TMA
    # merely spends every iteration waiting on a consumer instead of running
    # ahead of the MMA.
    #
    # One MMA per K tile, so the MMA having finished tile `t` means every issue
    # `<= t - N_S` is drained, giving a fully-started frontier of
    # `t + 1 - N_S`. A slot is released at the START of a tile (`k_empty`
    # arrives before the fold), so that frontier gates a refill and the TMA may
    # lead by `NSTAGE + 1 - N_S`. Two tiles of lead is the tight bound; the
    # `+ 2` carries one stage of margin, free at every depth this kernel derives
    # and still valid if that arrive ever moves after the fold.
    comptime assert NSTAGE >= N_S + 2, (
        "the consumer releases K slots now, so the ring must be deeper than the"
        " MMA's lag behind the fully-started frontier (N_S = "
        + String(N_S)
        + ") plus two to pipeline; got NSTAGE="
        + String(NSTAGE)
    )
    # Producer warpgroup thread roles, immediately after the consumer warps:
    # MMA, TMA, q-scale transpose, idle. The producer warpgroup is always four
    # warps (whole, for `setmaxnreg`).
    comptime MMA_WARP = CONS_WARPS
    comptime TMA_WARP = MMA_WARP + 1
    comptime XPOSE_WARP = TMA_WARP + 1

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
    var tid = thread_idx.x

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
    comptime EPI_CHUNK = _EPILOGUE_CHUNK
    comptime assert MMA_N % EPI_CHUNK == 0
    comptime assert EPI_CHUNK <= 64
    # Columns folded per accumulator step; see `_ACC_WIDTH_FORCE`.
    comptime ACCW = _acc_width[N_TOKENS, out_dtype]()
    # The fold walks columns in groups of four (one 16-byte q-scale load feeding
    # two f32x2 FFMAs), so a group must sit wholly inside one token and its base
    # must be 16-byte aligned. Both routes to this kernel supply num_heads in
    # {32, 64}, but nothing in the body enforced it.
    comptime assert EPI_CHUNK % 4 == 0 and num_heads % 4 == 0, (
        "the epilogue folds columns in groups of four, so both the epilogue"
        " chunk and num_heads must be multiples of 4 for a group never to"
        " straddle a token boundary"
    )
    comptime Q_INTERLEAVE = _q_interleave[N_TOKENS]()
    comptime assert _Q_INTERLEAVE_FORCE != 1 or Q_INTERLEAVE, (
        "FP8_INDEX_Q_INTERLEAVE=1 needs a power-of-two ACCW; got ACCW="
        + String(ACCW)
    )
    # Interleaved, lane `L` of `acc` always holds token `L % N_TOKENS`: a group
    # starting at column `col` lands on lanes `[col % ACC_LANES, + ACCW)`.
    # `lcm` is what makes that hold at any T -- `T | ACC_LANES` fixes the
    # token of every lane, and `ACCW | ACC_LANES` keeps a group from wrapping.
    comptime ACC_LANES = lcm(ACCW, N_TOKENS) if Q_INTERLEAVE else ACCW
    # `ACC_LANES` need not be a power of two (6 at T=3), which a SIMD width
    # must be, so the accumulator is `ACC_LANES / ACCW` group-wide slices.
    comptime ACC_SLICES = ACC_LANES // ACCW
    comptime assert ACCW % 2 == 0 and EPI_CHUNK % ACCW == 0, (
        "the fold pairs columns into f32x2 and a group must not straddle a"
        " `tcgen05.ld` chunk; got ACCW="
        + String(ACCW)
    )

    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    var smem = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=128,
        name="fp8_index_sm100_prefill_smem",
    ]()
    # Q slots (B operand) | K ring (A operand) | k_scale ring | q_scale slots
    # | metadata records | CLC scratch | mbars | tmem ptr.
    #
    # The k_scale ring is slot-for-slot the K ring's: slot `s` holds the scales
    # of the key rows the MMA reads from `k_smem` slot `s`, landed by a second
    # TMA on that slot's own `k_full`. It sits immediately after the K ring so
    # its base inherits the 128-B alignment TMA needs, and `KS_SLOT` is padded
    # to keep every later slot on that alignment. A slot stages `KS_BOX` scales,
    # one alignment unit more than the tile's `BM_key`, because the window can
    # only start on a 16-byte boundary (`_ks_align_elems`); the consumer skips
    # the leading `ks_off` residual.
    #
    # The Q, q_scale and metadata regions each hold `Q_SLOTS` items, so the
    # next item's Q lands under the current one. The q-scales stay f32: the
    # TMA copies `q_s`'s bytes and cannot narrow, so the hoist into `qs_reg`
    # converts instead.
    comptime Q_SLOTS = _Q_SLOTS
    var q_smem = smem
    var k_smem = smem + Q_SLOTS * q_elems
    var ks_smem = (k_smem + NSTAGE * k_elems).bitcast[Scalar[KS_DTYPE]]()
    var qs_smem = (ks_smem + NSTAGE * KS_SLOT).bitcast[Float32]()
    var md_smem = (qs_smem + Q_SLOTS * MMA_N).bitcast[Int32]()
    # The CLC response, 16-byte aligned because the records before it are
    # 32 bytes each, then the entry steal's outcome.
    var clc_resp = (md_smem + Q_SLOTS * _MD_WORDS).bitcast[UInt128]()
    var clc_found = md_smem + Q_SLOTS * _MD_WORDS + _CLC_RESP_WORDS
    var mbar = (md_smem + Q_SLOTS * _MD_WORDS + _CLC_SCRATCH_WORDS).bitcast[
        SharedMemBarrier
    ]()
    # Offsets are named because the lane-parallel init below maps a thread index
    # to an arrival count, which makes them part of the layout contract rather
    # than a one-off pointer bump.
    #
    # ONE S^T ring of `N_S` slots: the MMA warp issues one MMA per K tile into
    # consecutive slots, so K tile `t` lands on slot `t mod N_S`.
    #
    # Ordering is a layout contract, not taste: each class that carries a
    # non-default arrival count must be ONE contiguous range, and `s_empty`
    # must stay LAST -- see the init below.
    comptime K_FULL_OFF = 0
    comptime K_EMPTY_OFF = K_FULL_OFF + NSTAGE
    comptime S_FULL_OFF = K_EMPTY_OFF + NSTAGE
    comptime Q_FULL_OFF = S_FULL_OFF + N_S
    comptime QS_FULL_OFF = Q_FULL_OFF + Q_SLOTS
    comptime Q_EMPTY_OFF = QS_FULL_OFF + Q_SLOTS
    comptime QS_READY_OFF = Q_EMPTY_OFF + Q_SLOTS
    comptime CLC_FULL_OFF = QS_READY_OFF + Q_SLOTS
    comptime S_EMPTY_OFF = CLC_FULL_OFF + 1
    comptime N_MBAR = S_EMPTY_OFF + N_S
    comptime assert N_MBAR == CFG.n_mbars, (
        "body barrier layout disagrees with the config's `n_mbars`; got "
        + String(N_MBAR)
        + " vs "
        + String(CFG.n_mbars)
    )
    var k_full = mbar + K_FULL_OFF
    var k_empty = mbar + K_EMPTY_OFF
    var s_full = mbar + S_FULL_OFF
    var s_empty = mbar + S_EMPTY_OFF
    var q_full = mbar + Q_FULL_OFF
    var qs_full = mbar + QS_FULL_OFF
    var q_empty = mbar + Q_EMPTY_OFF
    var qs_ready = mbar + QS_READY_OFF
    var clc_full = mbar + CLC_FULL_OFF
    var ptr_tmem = (mbar + N_MBAR).bitcast[UInt32]()

    comptime q_flat_layout = tt_row_major[q_elems]()
    comptime k_flat_layout = tt_row_major[k_elems]()
    comptime ks_flat_layout = tt_row_major[KS_BOX]()
    comptime qs_flat_layout = tt_row_major[MMA_N]()

    # Threads that must release a K slot before the TMA may refill it: the MMA
    # warp (one elected arrive) plus every consumer thread that reads a
    # `k_scale` out of that slot. Each thread arrives for itself because
    # `mbarrier.arrive` carries the release that publishes its own `LDS` -- a
    # batched arrive would not.
    comptime K_EMPTY_ARRIVES = 1 + _NUM_SOFTMAX_THREADS
    # A Q slot is free once the MMA has drained its Q (one commit) and every
    # consumer thread has hoisted its q-scales and read its metadata.
    comptime Q_EMPTY_ARRIVES = 1 + CONS_THREADS
    # The transposer's lanes each arrive for their own stores.
    comptime QS_READY_ARRIVES = WARP_SIZE

    # Lane-parallel init, mirroring FA4's `FA4MiscMBars.init`: barrier `i` is
    # initialized by thread `i`, so all N_MBAR `mbarrier.init` issue as one `STS.64`
    # with a lane-indexed address instead of N_MBAR serialized stores from thread 0.
    #
    # Four classes carry a non-default count: `k_empty` (the MMA plus the
    # consumers that read the slot's scales), `q_empty` (the MMA plus the
    # consumers), `qs_ready` (the transposer's lanes) and `s_empty` (one arrival
    # per consumer thread). `s_full` is
    # armed by a single tcgen05 commit and every remaining producer barrier by
    # one TMA completion, so those take 1. Each class is one contiguous range
    # and `s_empty` sits at the BACK, so the count map stays a short `SEL`
    # chain rather than a branch chain -- keep the ranges contiguous or this
    # grows. Indexed by `tid` rather than lane so it stays correct if N_MBAR
    # ever passes WARP_SIZE.
    comptime assert N_MBAR <= CFG.nthreads
    # The load warp initializes `clc_full` itself: it is the only role that
    # touches it, and it does so before the CTA barrier.
    if tid < N_MBAR and tid != CLC_FULL_OFF:
        mbar[tid].init(
            Int32(K_EMPTY_ARRIVES) if (
                tid >= K_EMPTY_OFF and tid < S_FULL_OFF
            ) else Int32(Q_EMPTY_ARRIVES) if (
                tid >= Q_EMPTY_OFF and tid < QS_READY_OFF
            ) else Int32(
                QS_READY_ARRIVES
            ) if (
                tid >= QS_READY_OFF and tid < CLC_FULL_OFF
            ) else Int32(
                _NUM_SOFTMAX_THREADS
            ) if (
                tid >= S_EMPTY_OFF
            ) else Int32(
                1
            )
        )

    # tcgen05 alloc is warp-collective (.sync.aligned): exactly one warp (the MMA warp).
    # Release the lock right after so co-resident CTAs can allocate. `tcgen05.alloc`
    # takes a POWER-OF-TWO column count, so the stages' footprint is rounded up for the
    # allocation (MMA_N=96 uses 192 and allocates 256) while the stage stride stays
    # `S_COLS` -- the waste is dead columns at the top, not a gap between stages.
    #
    # Residency is NOT decided here: `_ctas_per_sm` decides it and
    # `nvvm.minctasm` stamps it, and the assert only catches the two
    # disagreeing.
    comptime TMEM_USED_COLS = N_S * S_COLS
    comptime TMEM_COLS = UInt32(next_power_of_two(TMEM_USED_COLS))
    comptime assert Int(TMEM_COLS) == CFG.tmem_cols, (
        "body TMEM footprint disagrees with the config's `tmem_cols`; got "
        + String(TMEM_COLS)
        + " vs "
        + String(CFG.tmem_cols)
    )
    comptime assert Int(TMEM_COLS) * CTAS_PER_SM <= 512, (
        "S^T stages need "
        + String(TMEM_COLS)
        + " TMEM columns (rounded up to a power of two from "
        + String(TMEM_USED_COLS)
        + "), which exceeds the SM's 512 at CTAS_PER_SM="
        + String(CTAS_PER_SM)
    )

    # The mbarrier init above is plain per-thread stores, so it runs before the
    # header loads below and overlaps their global round trip instead of
    # queuing behind it; a CTA that retires early just discards it. On a
    # 6-token decode that moves the CTA barrier ~100 ns earlier.
    # The CTA's first work item. Every field is CTA-uniform, so all three
    # roles derive the same tile window from it.
    var b = block_idx.x

    @inline(.always)
    def decode(bb: Int, token_block: Int32, key_part: UInt32) {imm} -> _Item:
        # ctaids are consumed in launch order, so map them onto token blocks
        # last-first: a causal block's key range grows with its index, and the
        # longest items then start first rather than forming the tail.
        return _decode_item[
            BM_key, CTA_TOKEN_STRIDE, kpool, _is_cache_length_accurate
        ](
            bb,
            Int32(grid_dim.y) - 1 - token_block,
            key_part,
            valid_length,
            k_operand,
            max_num_keys_dev,
            causal,
            num_key_parts_dev,
            out_row_begin_dev,
            out_row_end_dev,
        )

    # Warp-collective: one elected lane arms the barrier and issues.
    @inline(.always)
    def issue_cancel() {imm}:
        var e = elect()
        expect_bytes_pred(clc_full, Int32(size_of[UInt128]()), e)
        if e != 0:
            clusterlaunchcontrol_try_cancel(clc_resp, clc_full.bitcast[Int64]())

    # Waits the cancel in flight and returns the first non-empty item it and
    # any re-issued cancels win, or an empty item once one fails. It returns
    # only after a FAILED cancel has been waited or with a won item, so no
    # cancel is in flight when the stream ends, and none is issued after a
    # failure (undefined behavior). A won empty ctaid is the case its own CTA
    # would have bailed on.
    @inline(.always)
    def resolve_steal(mut bb: Int, mut phase: UInt32) {imm} -> _Item:
        while True:
            clc_full[0].wait(phase)
            phase ^= 1
            if clusterlaunchcontrol_query_cancel_is_canceled(clc_resp) == 0:
                return _Item(0, 0, 0, 0, 0, 0, 0, 0)
            var cta = clusterlaunchcontrol_query_cancel_get_first_ctaid_v4(
                clc_resp
            )
            # The response was read through the generic proxy; the next
            # cancel rewrites it through the async one.
            fence_async_view_proxy()
            bb = Int(cta[0])
            var it = decode(bb, Int32(cta[1]), cta[2])
            if it.n_tiles > 0:
                return it
            issue_cancel()

    var item = decode(b, Int32(block_idx.y), UInt32(block_idx.z))
    var start_of_seq = item.start_of_seq
    var seq_len = item.seq_len
    var tok0 = item.tok0
    var tok_hi = item.tok_hi
    var out_row0 = item.out_row0
    var num_keys = item.num_keys
    var tile_begin = item.tile_begin
    var n_tiles_local = item.n_tiles
    # An empty item steals one instead of retiring, below; the CTA exits only
    # when that steal finds nothing. The exit is uniform (every thread, after
    # the CTA barrier) and ahead of any collective op (TMA mbar / tcgen05
    # alloc), since a divergent early return deadlocks them.
    var own_empty = n_tiles_local <= 0
    var wid = warp_id[broadcast=True]()
    # Parity of the next `clc_full` completion; the load warp's alone.
    var clc_phase: UInt32 = 0
    if wid == TMA_WARP:
        if lane_id() == 0:
            clc_full[0].init(1)
            fence_mbarrier_init()
        syncwarp()
        # Only the load warp needs the item itself -- every other role reads it
        # off the Q slot -- so it steals alone and the CTA barrier below
        # publishes just whether it found one.
        if own_empty:
            issue_cancel()
            var it = resolve_steal(b, clc_phase)
            if lane_id() == 0:
                clc_found[0] = it.n_tiles
            start_of_seq = it.start_of_seq
            seq_len = it.seq_len
            tok0 = it.tok0
            tok_hi = it.tok_hi
            out_row0 = it.out_row0
            num_keys = it.num_keys
            tile_begin = it.tile_begin
            n_tiles_local = it.n_tiles
    barrier()
    if own_empty and clc_found[0] <= 0:
        return

    comptime k_bytes = k_elems * size_of[dtype]()
    comptime q_bytes = q_elems * size_of[dtype]()
    comptime qs_bytes = MMA_N * size_of[Float32]()
    comptime ks_bytes = KS_BOX * size_of[KS_DTYPE]()

    if wid < CONS_WARPS:
        warpgroup_reg_alloc[CFG.reg_consumer]()
        # The MMA warp arrives once it has allocated TMEM, so this publishes
        # the TMEM address.
        named_barrier[Int32(CONS_THREADS + WARP_SIZE)](_CONSUMER_BAR)
        var tmem_addr: UInt32 = ptr_tmem[0]
        # `tcgen05_ld[datapaths=32]` maps warp w, lane l -> accumulator row
        # `WARP_SIZE * (w % 4) + l`, so the consumer warpgroup's 4 warps cover
        # all BM_key rows one-per-thread. A thread owning a whole key row makes
        # the head sum thread-local (no cross-lane reduction) and puts 32
        # consecutive `key_local` in each warp, so each store is one 128B
        # transaction. `llvm_opaque_tid` is FA4's anti-hoist intrinsic; it does
        # NOT bind here (exactly one `%tid.x` read either way), and is kept only
        # for the cheaper spelling.
        var row = Int32(llvm_opaque_tid() & (_NUM_SOFTMAX_THREADS - 1))
        # `PipelineState[N].step()` is `index += 1; if index == N { index = 0;
        # phase ^= 1 }`. The consumer walks the S^T ring in issue order, so
        # after `k` tiles it sits on the slot and lap parity of issue `k` --
        # what the MMA wrote.
        var c_state = PipelineState[N_S]()

        # Which K ring slot holds this warpgroup's current tile. The MMA warp
        # walks tiles 0,1,2,... assigning slots 0,1,2,... mod NSTAGE, so local
        # tile `j` is always in slot `j % NSTAGE` -- and this cursor is built and
        # stepped exactly like `c_state` so it lands on the same tile's slot at
        # every iteration. Only the INDEX is read (the slot's data is already
        # published by `s_full`), so the phase is unused here.
        var k_state = PipelineState[NSTAGE]()

        # How many keys the producer's alignment round-down put in front of the
        # item's first key, and so where a slot's tile actually starts. The
        # per-tile term the producer adds (`it * BM_key`) is a multiple of
        # `KS_ALIGN`, so the residual is the batch entry's own and is the same on
        # every tile; the load warp computes it once per item. It is zero on
        # every paged operand (`page_size % BM_key == 0` forces it) and non-zero
        # only on a ragged batch whose per-entry key count is not a multiple of
        # `KS_ALIGN`.
        var ks_off: Int32 = 0

        # The q_scales a thread needs are item-invariant, and re-reading them
        # from SMEM on every key tile measured at 128 columns as 1,295,264
        # `LDS`, 7.8% of the instruction stream and 36% of the shared-load pipe,
        # with the stalls landing on the consuming `FFMA2` rather than on the
        # `LDS` itself. So each item hoists them into `qs_reg` once, taking the
        # in-loop `LDS` count to ZERO. A thread folds every column of its own
        # key row, so the hoist costs exactly `mma_n` registers.
        #
        # Raising the register budget does NOT make a wider hoist fit: a
        # uniform 168 at 128 columns spills 56-220 B, because with more
        # registers ptxas commits to the aggressive schedule that hoists all 8
        # `LDTM.x16` across the folds and then does not fit. Slack is capacity,
        # not a guarantee.
        #
        # Held as LANE PAIRS so that at `FT == bfloat16` one slot is one packed
        # register: `MMA_N` scales become `MMA_N / 2`. At f32 the pairing is
        # pure spelling -- same storage, same `fma.rn.f32x2` operands.
        var qs_reg = Array[SIMD[FT, 2], MMA_N // 2](uninitialized=True)

        # The epilogue stores in place today because a token's sum is final at
        # its last column, which collapses the accumulator to one f32x2. That
        # collapse is NOT what this undoes -- the fold still collapses per token
        # and `acc` still resets. What is deferred is only the STORE, so the
        # cost is `N_TOKENS` finished f32 values rather than the `N_TOKENS`
        # running accumulators plus all `MMA_N` q-scales (~250 registers) that
        # the original design rejected.
        #
        # Every index MUST stay comptime, exactly as for `qs_reg`: a runtime
        # index forces the array to local memory and the arm becomes a spill
        # experiment instead of a scheduling one.

        # `k_scale` now comes out of SMEM, staged by the TMA warp into this
        # tile's own K ring slot. What it replaces was a TWO-HOP dependent global
        # load (block table -> scale row) issued by every consumer thread and
        # rotated one key tile ahead to cover its own miss: 37% L2 hit, with 5.5%
        # of warp-stall samples landing on the `FMUL` that consumed it against
        # 0.2% on the identical one a token later. An `LDS` has none of that
        # variance, and the producer warp that now issues the load is 57-70%
        # idle.
        #
        # A row past `num_rows` reads pool data rather than the `0.0` the old
        # guard returned. Safe for the same reason the K tile's own out-of-range
        # rows are: the epilogue store is gated on `key_local < key_bound`, and
        # `key_bound <= num_rows` always, so such a score is folded and then
        # discarded, never written.
        # Raw base of the score buffer.
        var out_base = output.ptr

        # One item per pass: each item is read off its slot's metadata, its
        # q-scales re-hoisted, then the slot freed for the load warp. Every ring
        # cursor carries over.
        @__parameter
        @inline(.always)
        def consume_item():
            # `n_tiles_local` is `Int32`, so `range` yields an `Int32` induction
            # variable directly (`range.mojo:495`) -- the whole key-index chain
            # stays 32-bit with no per-iteration cast, which is what the narrowing
            # rule requires.
            for tile_i in range(n_tiles_local):
                var key_local = (tile_begin + tile_i) * Int32(BM_key) + row

                # `PipelineState.index()` is already `UInt32`; keep it that way
                # rather than round-tripping through 64-bit `Int`.
                var cs = c_state.index()
                s_full[cs].wait(c_state.phase())
                var s_it = tmem_addr + cs * UInt32(S_COLS)

                # No `k_full` wait here: the MMA warp waited it before issuing, and
                # `s_full` is armed by that MMA's completion, so the scales are
                # already visible transitively. Read then release immediately --
                # before the fold, not after -- so the TMA gets the slot back as
                # early as possible now that the consumer sits in its WAR loop.
                # `mbarrier.arrive`'s release orders the `LDS` ahead of the arrive
                # for later GENERIC-proxy accesses only. The refill is an async-proxy
                # TMA write, so the load warp fences after its wait (see the refill
                # loop) -- without it the TMA can land the slot's next tile before
                # this read returns.
                var ks = k_state.index()
                var k_scale = ks_smem[
                    ks * UInt32(KS_SLOT) + UInt32(ks_off + row)
                ].cast[.float32]()
                _ = k_empty[ks].arrive()

                # Legacy order: a token owns a CONTIGUOUS column range (`col //
                # num_heads`), so its sum is final at its last column: store it right
                # there and the accumulator collapses to one `SIMD[AT, ACCW]`, rather
                # than keeping every token sum and all `MMA_N` q-scales live at once
                # (~250 registers, which spilled). Interleaved order: lane `L` of
                # `acc` is token `L % N_TOKENS` throughout, so the same
                # `ACC_LANES` registers hold every token's sum and all T stores
                # happen after the last group.
                #
                # `num_heads` and `EPI_CHUNK` are both powers of two, so a chunk
                # never straddles a token partway and the completion test stays
                # comptime. A *runtime* token index would spill `acc` to local
                # memory.
                @inline(.always)
                def consume_group[
                    col: Int
                ](
                    frag: Array[Scalar[AT], EPI_CHUNK],
                    mut acc: Array[SIMD[FT, ACCW], ACC_SLICES],
                ) {mut qs_reg, imm}:
                    # `ACCW / 2` disjoint lane pairs either way, so the chains
                    # differ but the accumulate count does not.
                    comptime base = col % EPI_CHUNK
                    comptime k = (col % ACC_LANES) // ACCW
                    comptime if FT == DType.bfloat16:
                        # `relu_cvt_bf16x2` takes the HIGH lane first, the
                        # reverse of the SIMD index order.
                        #
                        # `FT` comes from `get_defined_dtype`, which the parser
                        # does not fold, so `SIMD[FT, 2]` needs a `rebind` even
                        # inside the arm that pins `FT` to bf16.
                        comptime for h in range(ACCW // 2):
                            var folded = relu_cvt_bf16x2(
                                frag[base + 2 * h + 1], frag[base + 2 * h]
                            ).fma(
                                rebind[SIMD[.bfloat16, 2]](
                                    qs_reg[col // 2 + h]
                                ),
                                rebind[SIMD[.bfloat16, 2]](
                                    acc[k].slice[2, offset=2 * h]()
                                ),
                            )
                            acc[k] = acc[k].insert[offset=2 * h](
                                rebind[SIMD[FT, 2]](folded)
                            )
                    else:
                        # `.fma()` is required over `a * b + c`: LLVM does not
                        # contract an f32 pair into one FFMA2. The relu is a
                        # vector op but PTX has no `max.f32x2` at any ISA
                        # version, so it lowers to one FMNMX per column.
                        #
                        # `frag` stays element-addressed: it is a
                        # register-resident `tcgen05.ld` result, and taking a
                        # pointer to it to load a pair would put it in local
                        # memory.
                        var raw = SIMD[FT, ACCW]()
                        var qsv = SIMD[FT, ACCW]()
                        comptime for e in range(ACCW):
                            raw[e] = rebind[Scalar[FT]](frag[base + e])
                        comptime for h in range(ACCW // 2):
                            qsv = qsv.insert[offset=2 * h](qs_reg[col // 2 + h])
                        acc[k] = max(raw, SIMD[FT, ACCW](0)).fma(qsv, acc[k])
                    # Legacy closes a token at its last column; interleaved, every
                    # token is still open until the tile's last group.
                    comptime closes = (
                        col + ACCW == MMA_N
                    ) if Q_INTERLEAVE else ((col + ACCW) % num_heads == 0)
                    comptime if closes:
                        comptime for s in range(
                            N_TOKENS if Q_INTERLEAVE else 1
                        ):
                            comptime t = s if Q_INTERLEAVE else col // num_heads
                            # The token sum is taken in f32 whatever `FT` is.
                            var tok_sum: Scalar[AT]
                            comptime if Q_INTERLEAVE:
                                tok_sum = acc[t // ACCW][t % ACCW].cast[AT]()
                                comptime for l in range(
                                    t + N_TOKENS, ACC_LANES, N_TOKENS
                                ):
                                    tok_sum += acc[l // ACCW][l % ACCW].cast[
                                        AT
                                    ]()
                            else:
                                tok_sum = acc[0].cast[AT]().reduce_add()
                            var tok_local = tok0 + Int32(t)
                            # Fused causal mask (branchless): token tok_local sees
                            # keys up to cache_len + tok_local, so forbidden slots
                            # are left unwritten and the separate mask pass is
                            # skipped. What carries the mask downstream is the
                            # top-k's per-row bound, not a sentinel:
                            # `topk_row_bounds_kernel` computes this same
                            # `indexer_key_bound` and the bounded top-k pads past
                            # it in-register, so the buffer is left
                            # uninitialized. The `topk_gpu` arm (top_k > 2048) is
                            # the exception -- it scans the whole row behind the
                            # caller's -Float32.MAX fill, so an over-range write
                            # there WOULD be selected.
                            # The liveness half is memory safety rather than
                            # masking -- a dead token's row belongs to the NEXT
                            # batch entry.
                            #
                            # The guard skips no work worth skipping (0.03%
                            # divergence), but as a BRANCH it costs a
                            # `BSSY`/`BRA`/`NOP`/`BSYNC` quartet per token per key
                            # tile: 4.6% of the instruction stream at 96 columns,
                            # 10.6% at 128. store_global_pred folds it into a PTX
                            # `@%p st.global.bXX` instead, measured at -5.4% to
                            # 0.0% wall clock over five shapes with none
                            # regressing. Predicating only the STORE leaves every
                            # register dependency intact, so ptxas does NOT hoist
                            # the 8 `LDTM.x16` across the folds: zero `STL`/`LDL`
                            # in both arms at MMA_N 96 and 128, nh 32 and 64.
                            var key_bound = (
                                num_keys - (seq_len - 1 - tok_local) * causal
                            ) // Int32(kpool)
                            # The OFFSET must be 64-bit: the score buffer is
                            # `total_seq_len * max_num_keys` elements, so the flat
                            # index passes 2^31 at ~13.1K tokens x 163840 keys and
                            # 2^32 at 32K x 163840, both reachable by configuration
                            # with no allocation guard on that path. Both factors
                            # are `Int32` though, so this is one `IMAD.WIDE`
                            # (32x32->64) rather than a 64-bit multiply. Do NOT
                            # narrow it.
                            var out_row = out_row0 + tok_local
                            store_global_pred(
                                out_base
                                + (
                                    Int(out_row) * Int(max_num_keys)
                                    + Int(key_local)
                                ),
                                (k_scale * tok_sum).cast[out_dtype](),
                                Int32(
                                    key_local < key_bound and tok_local < tok_hi
                                ),
                            )
                        comptime if not Q_INTERLEAVE:
                            acc[0] = SIMD[FT, ACCW](0)

                # Drain this thread's key row in `EPI_CHUNK`-column chunks. The
                # chunk loads carry no wait between them, so they pipeline against
                # the folds: `tcgen05.ld` register outputs are automatically
                # ordered, and the single `tcgen05_load_wait` is only the WAR fence
                # before the stage is released.
                var acc = Array[SIMD[FT, ACCW], ACC_SLICES](
                    fill=SIMD[FT, ACCW](0)
                )
                comptime for c in range(MMA_N // EPI_CHUNK):
                    var frag = TMemTile[AT, BM_key, EPI_CHUNK](
                        s_it + UInt32(c * EPI_CHUNK)
                    ).load_async()
                    comptime for g in range(EPI_CHUNK // ACCW):
                        consume_group[c * EPI_CHUNK + ACCW * g](frag, acc)
                tcgen05_load_wait()
                tcgen05_fence_before()
                _ = s_empty[cs].arrive()

                c_state.step()
                k_state.step()

        # Interleaved, the slot is consumable only once warp 6 has transposed
        # it; `qs_ready` also forwards the end-of-stream sentinel. The CTA's
        # FIRST item is the exception: nothing precedes it for warp 6 to hide
        # behind, and its MMA needs only Q, so the consumers transpose it
        # themselves while that MMA runs instead of paying warp 6's extra hop.
        var first = True
        var q_state = PipelineState[Q_SLOTS]()
        while True:
            var qslot = q_state.index()
            var own_xpose = Q_INTERLEAVE and first
            if Q_INTERLEAVE and not own_xpose:
                qs_ready[qslot].wait(q_state.phase())
            else:
                qs_full[qslot].wait(q_state.phase())
            var md = (md_smem + qslot * UInt32(_MD_WORDS)).unsafe_load[
                width=_MD_WORDS, alignment=32
            ]()
            n_tiles_local = md[_MD_N_TILES]
            if n_tiles_local == 0:
                break
            tok0 = md[_MD_TOK0]
            tok_hi = md[_MD_TOK_HI]
            seq_len = md[_MD_SEQ_LEN]
            num_keys = md[_MD_NUM_KEYS]
            out_row0 = md[_MD_OUT_ROW0]
            tile_begin = md[_MD_TILE_BEGIN]
            ks_off = md[_MD_KS_OFF]
            # Interleaved, the hoist reads the slot already in column order; see
            # the transposer role for why it is permuted in SMEM.
            var qs_slot = qs_smem + qslot * UInt32(MMA_N)
            if own_xpose:
                var src = Int(row)
                var v = Float32(0)
                if src < MMA_N:
                    v = qs_slot[src]
                named_barrier[Int32(CONS_THREADS)](_CONSUMER_BAR)
                if src < MMA_N:
                    qs_slot[(src % num_heads) * N_TOKENS + src // num_heads] = v
                named_barrier[Int32(CONS_THREADS)](_CONSUMER_BAR)
            # Read FOUR at a time: adjacent scalar reads coalesce into one
            # 16-byte access and an explicit width-2 load does not re-merge.
            # Every index MUST stay comptime -- a runtime index forces the array
            # to local memory.
            comptime for j in range(MMA_N // 4):
                var qs4 = qs_slot.unsafe_load[
                    width=4, alignment=4 * size_of[Float32]()
                ](4 * j)
                qs_reg[2 * j] = qs4.slice[2, offset=0]().cast[FT]()
                qs_reg[2 * j + 1] = qs4.slice[2, offset=2]().cast[FT]()
            _ = q_empty[qslot].arrive()
            consume_item()
            q_state.step()
            first = False

        # Drain across the consumer warps: every consumer's last `s_empty`
        # arrive happens-before this barrier, so no S^T stage is live when the
        # MMA warp frees TMEM.
        named_barrier[Int32(CONS_THREADS)](_CONSUMER_BAR)
        # Re-read rather than test `wid`: keeping it live across the fold
        # loop spilled it (4 B STL, reloaded by every role's dispatch test).
        if llvm_opaque_tid() < UInt32(WARP_SIZE):
            tcgen05_dealloc[1](tmem_addr, TMEM_COLS)
    else:
        if wid == MMA_WARP:
            # Release registers BEFORE the TMEM alloc: the consumers'
            # `setmaxnreg.inc` blocks until this warp's `.dec`. Deallocating
            # after the alloc put its latency on the consumers' path, ~0.13-0.19
            # us per launch on decode.
            warpgroup_reg_dealloc[CFG.reg_producer]()
            tcgen05_alloc[1](ptr_tmem, TMEM_COLS)
            tcgen05_release_allocation_lock[1]()
            # The role runs warp-collectively and elects one lane per issue:
            # `SM100TensorAccumulator.mma` broadcasts the accumulator's TMEM
            # address from lane 0 (`shfl.sync` over the full warp mask), so
            # calling it from a single-lane region hangs the warp on a
            # convergence barrier the other 31 lanes never reach.
            # Nothing the consumers publish gates the MMA, so this warp only
            # ARRIVES -- handing the consumers the TMEM address -- and runs on.
            syncwarp()
            named_barrier_arrive[Int32(CONS_THREADS + WARP_SIZE)](_CONSUMER_BAR)
            var tmem_addr: UInt32 = ptr_tmem[0]
            var e = elect()
            var kc_state = PipelineState[NSTAGE]()
            # The MMA warp is the PRODUCER of the S^T stages, so its WAR state starts
            # pre-flipped: `wait(1)` on a fresh mbar tests a phase that has not been
            # reached and falls through, replacing a prologue that pre-armed `s_empty`
            # with 256 arrivals purely to make that same first wait pass. Same device
            # as the K ring's `kp_state` below.
            var sp_state = PipelineState[N_S](0, 1, 0)
            var k = smem_descriptor[
                BMN=BM_key,
                BK=compute_BK,
                swizzle_mode=SW,
                is_k_major=True,
            ](k_smem)

            # One item per pass, like the consumers: the tile count and Q come
            # off the item's slot, and the slot is freed behind the item's last
            # MMA. `kc_state` and `sp_state` carry over.
            var q_state = PipelineState[Q_SLOTS]()
            while True:
                var qslot = q_state.index()
                q_full[qslot].wait(q_state.phase())
                n_tiles_local = md_smem[
                    qslot * UInt32(_MD_WORDS) + UInt32(_MD_N_TILES)
                ]
                if n_tiles_local == 0:
                    break
                var q0 = smem_descriptor[
                    BMN=MMA_N,
                    BK=compute_BK,
                    swizzle_mode=SW,
                    is_k_major=True,
                ](q_smem + qslot * UInt32(q_elems))
                # Trip count must match the consumer's and the load warp's
                # exactly: the k/s mbar handshakes and every `PipelineState`
                # phase stay in lockstep only because all roles walk the same
                # tile window.
                for _ in range(n_tiles_local):
                    var s = kc_state.index()
                    k_full[s].wait(kc_state.phase())
                    # One MMA per K tile into the ring, so the ring's issue
                    # counter IS the tile counter. `sp_state` tracks it and is
                    # re-read and stepped here rather than hoisted.
                    var st = sp_state.index()
                    # WAR: the consumers must have drained this S stage before the
                    # MMA overwrites it. The first pass over each stage falls through
                    # on the pre-flipped phase (nothing to drain yet); after that
                    # this is the consumer's `release_stage()`.
                    s_empty[Int(st)].wait(sp_state.phase())
                    QK.mma(
                        k + s * UInt32(k_elems),
                        q0,
                        tmem_addr + st * UInt32(S_COLS),
                        c_scale=0,
                        elect=e,
                    )
                    elect_mma_arrive(s_full + Int(st), e)
                    sp_state.step()
                    # Release K stage s only after the MMA has drained it
                    # (tcgen05.commit tracks the async MMA); a plain mbar arrive
                    # would let the load warp overwrite K mid-read.
                    elect_mma_arrive(k_empty + s, e)
                    kc_state.step()
                # `tcgen05.commit` tracks every MMA issued before it, so the
                # slot's Q is drained once this arrives.
                elect_mma_arrive(q_empty + qslot, e)
                q_state.step()
        elif wid == TMA_WARP:
            warpgroup_reg_dealloc[CFG.reg_producer]()
            var kp_state = PipelineState[NSTAGE](0, 1, 0)
            var e = elect()

            # One expect-bytes with the K tile's and its scales' bytes SUMMED is
            # the only legal shape, not one call per copy: it lowers to
            # `mbarrier.arrive.expect_tx`, so it IS the producer's single arrival,
            # and against an init count of 1 a second call would drive the
            # pending-arrival count below zero -- out of spec, not merely
            # redundant. Summed, both TMAs deposit into the one tx counter and the
            # barrier fires when the last byte of either lands.
            #
            # The single-lane guard rides an `@%p` on each instruction rather than an
            # `if e != 0:`. That is NOT a branch removal -- ptxas already if-converts
            # the guarded form and both shapes emit the same SASS at the issue. It is
            # worth 8 fewer instructions of uniform-datapath setup per sidecar, and
            # consistency with every other SM100 producer.
            @inline(.always)
            def issue_k(it: Int32, state: PipelineState[NSTAGE]) {imm}:
                var s = state.index()
                # Split rather than folded: the fold spans the whole shared
                # slab and leaves the signed 32-bit TMA coordinate on a large
                # pool. See `MHAOperand.kv_tma_coords`.
                var k_row0, k_block = k_operand.kv_tma_coords(
                    UInt32(b), UInt32(it * Int32(BM_key))
                )
                var k_dst = TileTensor[
                    dtype, type_of(k_flat_layout), address_space=.SHARED
                ](k_smem + s * UInt32(k_elems), k_flat_layout)
                expect_bytes_pred(k_full + s, Int32(k_bytes + ks_bytes), e)
                k_tma.async_copy_4d_elect(
                    k_dst, k_full[s], (0, 0, Int(k_row0), Int(k_block)), e
                )
                # The scales ride the K tile's barrier: the bytes are summed
                # into the one `expect_tx`, so the slot opens when the last byte
                # of either copy lands and the consumer needs no wait of its own
                # -- `s_full` already happens-after this.
                #
                # The scales' own row index, NOT `k_row0`: a paged scale pool
                # resolves its block through `scales_lookup_table` and strides
                # by its own block pitch, so reusing the K row silently reads
                # another block whenever the two LUTs differ.
                #
                # Rounded DOWN to `KS_ALIGN`: a scale "row" is a single scalar,
                # so this index IS the innermost TMA coordinate and must land on
                # a 16-byte boundary, which a ragged batch's per-entry base does
                # not. The window is `KS_ALIGN` keys wider to cover the shift and
                # the consumer skips the residual. (`k_row0` needs no such fix --
                # a K row is a whole `depth`-wide key, so it is always aligned.)
                var ks_row0, ks_blk = ks_operand.scale_tma_coords(
                    UInt32(b), UInt32(it * Int32(BM_key))
                )
                var ks_dst = TileTensor[
                    KS_DTYPE,
                    type_of(ks_flat_layout),
                    address_space=.SHARED,
                ](ks_smem + s * UInt32(KS_SLOT), ks_flat_layout)
                ks_tma.async_copy_elect(
                    ks_dst, k_full[s], (Int(ks_row0), Int(ks_blk)), e
                )

            # Item stream: publish the item into the next Q slot, stream its K
            # tiles, steal the next one. A zero-tile sentinel ends the stream for
            # the MMA and the consumers.
            #
            # One K loop for the whole stream. `kp_state` starts pre-flipped, so
            # a slot's first-lap `k_empty` wait falls through on the fresh
            # barrier; after that it is the refill wait, across items as within
            # one.
            var q_state = PipelineState[Q_SLOTS](0, 1, 0)
            while True:
                var qslot = q_state.index()
                q_empty[qslot].wait(q_state.phase())
                # The consumers read the slot's q-scales with `LDS` (generic
                # proxy) before arriving; the TMA below rewrites it through the
                # async proxy. Same hazard as the K ring.
                fence_async_view_proxy()
                var ks_off = Int32(
                    ks_operand.row_idx(
                        UInt32(b), UInt32(tile_begin * Int32(BM_key))
                    )
                    & UInt32(KS_ALIGN - 1)
                )
                # Stored by the lane that arrives below: each `arrive.expect_tx`
                # is a release, so it publishes the record to whoever waits that
                # barrier.
                if e != 0:
                    (md_smem + qslot * UInt32(_MD_WORDS)).unsafe_store[
                        alignment=32
                    ](
                        SIMD[DType.int32, _MD_WORDS](
                            tok0,
                            tok_hi,
                            seq_len,
                            num_keys,
                            out_row0,
                            tile_begin,
                            n_tiles_local,
                            ks_off,
                        )
                    )
                var tok_row = Int(start_of_seq + tok0)
                # Q and q-scales on SEPARATE barriers: the MMA needs only Q, and
                # the consumers only the q-scales.
                expect_bytes_pred(q_full + qslot, Int32(q_bytes), e)
                q_tma.async_copy_3d_elect(
                    TileTensor[
                        dtype, type_of(q_flat_layout), address_space=.SHARED
                    ](q_smem + qslot * UInt32(q_elems), q_flat_layout),
                    q_full[qslot],
                    _q_tma_coords[Q_INTERLEAVE, num_heads](tok_row),
                    e,
                )
                expect_bytes_pred(qs_full + qslot, Int32(qs_bytes), e)
                qs_tma.async_copy_elect(
                    TileTensor[
                        DType.float32,
                        type_of(qs_flat_layout),
                        address_space=.SHARED,
                    ](qs_smem + qslot * UInt32(MMA_N), qs_flat_layout),
                    qs_full[qslot],
                    (0, tok_row),
                    e,
                )
                q_state.step()
                for i in range(n_tiles_local):
                    k_empty[kp_state.index()].wait(kp_state.phase())
                    # Write-after-read across proxies: the consumers read this
                    # slot's k-scales with `LDS` (generic proxy) before arriving,
                    # and the refill below writes it through the async proxy. The
                    # wait acquires their releases; this fence orders those reads
                    # before the TMA write. One thread pays it, not 128.
                    fence_async_view_proxy()
                    issue_k(tile_begin + i, kp_state)
                    kp_state.step()
                # Claim the next item only now, once this CTA is nearly free: the
                # K ring still holds up to `k_stages` of this item's tiles, which
                # covers the response latency.
                issue_cancel()
                var nxt = resolve_steal(b, clc_phase)
                if nxt.n_tiles <= 0:
                    break
                start_of_seq = nxt.start_of_seq
                seq_len = nxt.seq_len
                tok0 = nxt.tok0
                tok_hi = nxt.tok_hi
                out_row0 = nxt.out_row0
                num_keys = nxt.num_keys
                tile_begin = nxt.tile_begin
                n_tiles_local = nxt.n_tiles
            var qslot = q_state.index()
            q_empty[qslot].wait(q_state.phase())
            if e != 0:
                md_smem[qslot * UInt32(_MD_WORDS) + UInt32(_MD_N_TILES)] = 0
                _ = q_full[qslot].arrive()
                _ = qs_full[qslot].arrive()
        elif wid == XPOSE_WARP:
            warpgroup_reg_dealloc[CFG.reg_producer]()
            # The slot lands token-major, `[t * num_heads + h]`, as `q_s`
            # stores it; the interleaved fold wants column order `h * T + t`.
            # Permuting at the hoist instead puts the two columns of each
            # `FFMA2` in two different loads, and ptxas then re-pairs them with
            # ~88 moves on EVERY tile (measured 1.16x geomean slower). Done by
            # the consumers it costs two named barriers per item on their
            # critical path; here it runs while they fold the previous item,
            # measured 1-2% faster on prefill.
            # One warp needs no barrier between its reads and its writes:
            # `syncwarp` orders them, so the permute stays in place.
            comptime if Q_INTERLEAVE:
                comptime PER_LANE = ceildiv(MMA_N, WARP_SIZE)
                var lane = Int(lane_id())
                var q_state = PipelineState[Q_SLOTS]()
                # The consumers transpose the first item, which is never the
                # sentinel: an empty CTA returns before any role starts. Still
                # arrive for it, unread: `qs_ready` must complete once per item,
                # or slot 0's barrier runs a phase behind the slot and its
                # second item waits on a parity that never completes.
                #
                # And still WAIT for it: a parity wait cannot tell "phase 1
                # done" from "phase 0 not yet done", so if this warp first met
                # `qs_full[0]` at the slot's NEXT item, it would pass while the
                # first item's TMA is still in flight and take the first item
                # for that one -- transposing it under the consumers and
                # handing them its stale metadata.
                qs_full[0].wait(0)
                _ = qs_ready[0].arrive()
                q_state.step()
                while True:
                    var qslot = q_state.index()
                    qs_full[qslot].wait(q_state.phase())
                    if (
                        md_smem[qslot * UInt32(_MD_WORDS) + UInt32(_MD_N_TILES)]
                        == 0
                    ):
                        _ = qs_ready[qslot].arrive()
                        break
                    var qs_slot = qs_smem + qslot * UInt32(MMA_N)
                    var v = SIMD[DType.float32, next_power_of_two(PER_LANE)](0)
                    comptime for i in range(PER_LANE):
                        var src = lane + i * WARP_SIZE
                        if (i + 1) * WARP_SIZE <= MMA_N or src < MMA_N:
                            v[i] = qs_slot[src]
                    syncwarp()
                    comptime for i in range(PER_LANE):
                        var src = lane + i * WARP_SIZE
                        if (i + 1) * WARP_SIZE <= MMA_N or src < MMA_N:
                            qs_slot[
                                (src % num_heads) * N_TOKENS + src // num_heads
                            ] = v[i]
                    _ = qs_ready[qslot].arrive()
                    q_state.step()
        else:
            # Idle warp, and the ONLY reason it is launched: it must
            # dealloc the SAME count as the working warps, because
            # `setmaxnreg.sync.aligned` is warpgroup-collective and a differing
            # operand within the producer warpgroup is UB.
            warpgroup_reg_dealloc[CFG.reg_producer]()


# Route a shape here when a sequence spans at least this many token blocks: the
# token-block grid alone then fills the machine without a key split.
comptime _PREFILL_MIN_TOKEN_TILES = 16

# The other direction: too FEW token blocks to fill the grid, but a key range
# deep enough that splitting it does. This is decode / MTP-decode against a long
# cache, where the K-resident scorer degenerates to one CTA per key tile doing a
# single MMA -- it pays a full CTA prologue (TMEM alloc, mbar init, Q staging,
# k_scale gather) per 128 keys and never pipelines the K stream. Here an item
# instead streams a part of the key range (`_clc_key_parts`) through the ring
# behind a resident Q tile.
comptime _KEYSPLIT_MAX_TOKEN_TILES = 4
comptime _KEYSPLIT_MIN_KEY_TILES = 64
# Load-bearing beyond this route: the alternate N-tile's block-count clause reads
# `max(token_tiles, _KEYSPLIT_MAX_TOKEN_TILES)`, so at K=4 with N_ALT=3 only a
# `max_seq_len` in {3, 6, 9} can reach the narrower tile -- speculative widths only,
# which is the property every `MMA_N < 128` claim in this file is scoped on. Raising K
# widens that toward genuine prefill, so re-tuning it must re-check the alternate
# tile, not just the key split.


@inline(.always)
def _is_keysplit_shape[
    num_heads: Int, BM_key: Int
](max_seq_len: Int, max_num_keys: Int) -> Bool:
    """Whether a shape is decode-shaped: few token blocks, deep key range.

    One definition, read by both the router (which admits such a shape to the
    Q-resident kernel) and the launcher (which relaxes its key-split guard for
    it). Deliberately measured against the DEFAULT `128 // num_heads` tile
    rather than the tile actually launched: a wider tile makes any shape look
    like fewer token blocks, and a 17-token prefill batch on the 8-token tile
    would otherwise read as decode.

    Parameters:
        num_heads: Query index heads.
        BM_key: Keys per tile.

    Args:
        max_seq_len: Batch maximum of new query tokens.
        max_num_keys: Row stride of the score buffer; an upper bound on any
            entry's key count, and under graph capture a frozen one.

    Returns:
        True when the token-block grid alone cannot fill the machine but the
        key range is deep enough that splitting it can.
    """
    return (
        ceildiv(max_seq_len, 128 // num_heads) <= _KEYSPLIT_MAX_TOKEN_TILES
        and ceildiv(max_num_keys, BM_key) >= _KEYSPLIT_MIN_KEY_TILES
    )


@inline(.always)
def _create_prefill_q_tma[
    dtype: DType, //, num_heads: Int, depth: Int, N_TOKENS: Int
](
    ctx: DeviceContext,
    q_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    num_q_tokens: Int,
    out res: QTMATileT[dtype, N_TOKENS * num_heads, depth],
) raises:
    """Builds the resident-Q descriptor in the row order the kernel folds in.

    Q is `[num_q_tokens, num_heads, depth]` contiguous. Interleaved, the box is
    `(num_heads, N_TOKENS, depth)` over global strides `(depth, num_heads *
    depth, 1)`, so TMA writes SMEM row `h * N_TOKENS + t`. It is still one copy
    of `MMA_N` 128-byte rows, so it is wrapped in the same `QTMATileT` as the
    legacy descriptor: the tile type only drives the copy count and offsets,
    and both are one copy at offset 0.
    """
    comptime if _q_interleave[N_TOKENS]():
        # One SMEM row must be exactly one swizzle atom, or the legacy tile
        # type's padded row would stop matching the descriptor's box.
        comptime assert (
            depth * size_of[dtype]() == _INDEX_SWIZZLE.bytes()
        ), "interleaved Q needs depth rows of exactly one 128B swizzle atom"
        var q_buf = DeviceBuffer(
            ctx, q_ptr.address_space_cast[.GENERIC](), 1, owning=False
        )
        var desc = create_tma_descriptor[dtype, 3, _INDEX_SWIZZLE](
            q_buf,
            IndexList[3](num_heads, num_q_tokens, depth),
            IndexList[3](depth, num_heads * depth, 1),
            IndexList[3](num_heads, N_TOKENS, depth),
        )
        res = QTMATileT[dtype, N_TOKENS * num_heads, depth](desc)
    else:
        res = create_split_tma[
            coord[N_TOKENS * num_heads, 1, depth],
            coord[UNKNOWN_VALUE, 1, depth],
            _INDEX_SWIZZLE,
        ](ctx, q_ptr, num_q_tokens * num_heads)


def fp8_index_score_sm100_prefill[
    out_dtype: DType,
    //,
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    kpool: Int = 1,
    *,
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    q_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    num_q_tokens: Int,
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[mut=False, .uint32, ..., Engine=VLEngine],
    q_s: TileTensor[mut=False, .float32, ..., Engine=QSEngine],
    output: TileTensor[out_dtype, ..., Engine=OutEngine],
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

    `out_dtype` is inferred from `output` and may be f32 or bf16; the
    accumulator is f32 either way and only the final store converts.
    """
    # The window bounds the token blocks any entry can contribute, so a chunked
    # launch does not pay for a grid sized to the whole batch.
    var out_rows = Int(output.dim[0]())
    comptime sm_count = ctx.default_device_info.sm_count
    comptime MMA_N = N_TOKENS * num_heads
    comptime CTAS_PER_SM = _ctas_per_sm[MMA_N]()
    comptime CFG = IndexPrefillConfig[
        dtype,
        KSOperand.dtype,
        depth,
        BM_key,
        MMA_N,
        num_heads,
    ]()
    comptime CTA_TOKEN_STRIDE = CFG.cta_token_stride
    var token_blocks = ceildiv(min(max_seq_len, out_rows), CTA_TOKEN_STRIDE)
    # `base_ctas` counts the CTAs that will WORK, not the ones the grid totals.
    # `batch_size * ceildiv(max_seq_len, N_TOKENS)` is the total, and on a ragged
    # batch nearly all of the excess decodes to an empty item
    # (`tok0 >= tok_hi`) and does no work. A batch that
    # launches past `sm_count` while working well under it would otherwise be read
    # as a full machine, decline the split, and leave the deepest entry streaming
    # its whole key range on ONE CTA. The token total is exact (`output` is
    # `[total_seq, max_num_keys]` by both entries' contract) and the `min` makes
    # this a pure RELAXATION: `base_ctas` can only fall, so every uniform batch
    # keeps its grid byte for byte and no shape loses a split it already had.
    #
    # The second clause lets a DECODE-shaped launch split even when the token-block
    # grid already fills the machine. Without it, at batch >= sm_count no key part
    # is ever assigned, so a ragged CONTEXT mix -- uniform query width, wildly
    # different cache depths -- runs the deepest entry on one CTA beside SMs that
    # finished early, and the host cannot see that skew because per-entry depths
    # are device data. Measured worth 2.3x-4.5x at batch >= 148. It is restricted
    # to `_is_keysplit_shape` because relaxing it at prefill widths would multiply
    # an already-full grid for no extra parallelism.
    # Each entry occupies `ceildiv(seq_len, N_TOKENS)` blocks, and summing that
    # over the batch without per-entry lengths is exactly the second term, which
    # over-counts by at most `batch_size - 1`. The `max(1, ...)` guards an empty
    # batch only; the wave-fill part count below divides by this. The first term is bounded
    # by the row window (`out_rows`), so a chunked score-matrix launch sizes
    # its grid to the rows it fills rather than the whole batch (see #96716);
    # an unwindowed caller has `out_rows >= max_seq_len` and the `min` is a no-op.
    var base_ctas = max(
        1,
        min(
            batch_size * ceildiv(min(max_seq_len, out_rows), CTA_TOKEN_STRIDE),
            (Int(output.dim[0]()) + batch_size * (CTA_TOKEN_STRIDE - 1))
            // CTA_TOKEN_STRIDE,
        ),
    )
    var key_tiles = ceildiv(max_num_keys, BM_key)
    var split_shape = base_ctas < sm_count or _is_keysplit_shape[
        num_heads, BM_key
    ](max_seq_len, max_num_keys)
    var num_key_parts = _clc_key_parts(
        split_shape, key_tiles, (CTAS_PER_SM * sm_count) // base_ctas
    )

    comptime kernel = _fp8_index_score_prefill_kernel_sm100[
        dtype,
        KOperand,
        KSOperand,
        type_of(valid_length.as_imm()).LayoutType,
        type_of(q_s).LayoutType,
        output.dtype,
        type_of(output).LayoutType,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
        kpool,
        VLEngine=type_of(valid_length.as_imm()).Engine,
        QSEngine=q_s.Engine,
        OutEngine=output.Engine,
    ]
    # Validate where the config is BUILT, ahead of the derived asserts below --
    # the SMEM check fires on an over-forced ring depth first and reports only a
    # byte count, which hides which knob caused it.
    # The `setmaxnreg` caps redistribute a fixed file: 65536 registers/SM shared
    # by `CTAS_PER_SM` CTAs, and every thread of a warpgroup holds that
    # warpgroup's cap. Counted per thread rather than per warpgroup so it stays
    # right when the producer is not a full warpgroup.
    comptime assert (
        _NUM_SOFTMAX_THREADS * CFG.reg_consumer
        + WARP_SIZE * CFG.prod_warps * CFG.reg_producer
        <= 65536 // CFG.ctas_per_sm
    ), (
        "the warpgroup register caps must fit the per-SM register file at"
        " CTAS_PER_SM CTAs/SM"
    )
    comptime smem_bytes = CFG.smem_used
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
        + "; lower the K ring depth for MMA_N="
        + String(MMA_N)
    )
    # `max_num_keys` crosses the ABI as `Int32` and the kernel's whole key-index
    # chain is 32-bit signed, so the narrowing here is a silent truncation of a
    # caller-supplied `Int`. One key tile of headroom rather than stopping at
    # `Int32.MAX`: `key_local` already reaches one tile past `num_keys` on the
    # last tile of a non-BM-aligned range, and at `Int32.MAX` that wraps negative
    # and passes the signed `key_local < key_bound` store guard. It used to be
    # two tiles because the consumer speculatively gathered `k_scale` one tile
    # ahead; the k-scale TMA reads only the tile it is staging, so that extra
    # tile is no longer touched.
    comptime KEY_INDEX_HEADROOM = BM_key
    debug_assert[assert_mode="safe"](
        max_num_keys <= Int(Int32.MAX) - KEY_INDEX_HEADROOM,
        (
            "fp8 index prefill: max_num_keys must leave one key tile of"
            " headroom under Int32.MAX; the device-side key"
            " arithmetic is 32-bit signed"
        ),
    )
    # Built here, not beside `k_tma` in `fp8_index_score_sm100`: the K-resident
    # scorer needs no scale descriptor, and this is host work paid per launch.
    var ks_tma = ks_operand.create_index_scale_tma_tile[BM_key](ctx)
    var q_tma = _create_prefill_q_tma[num_heads, depth, N_TOKENS](
        ctx, q_ptr, num_q_tokens
    )
    if Int(q_s.dim[1]()) != num_heads:
        raise Error("fp8 index prefill: q_s must be [tokens, num_heads]")
    # Token-major `[tokens, num_heads]` f32, one box per token block. A box
    # running past the last token is zero-filled, and a token past its entry
    # reads the next entry's scales, which only feed unstored sums.
    var qs_tma = create_tma_tile[N_TOKENS, num_heads](ctx, q_s)
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        ks_tma,
        qs_tma,
        k_operand,
        ks_operand,
        valid_length.as_imm(),
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
        block_dim=CFG.nthreads,
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        ),
    )
