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
"""MSA combine (LSE-merge) kernel for SM100 (B200).

The block-major forward (`msa_2q._msa_sm100_block_major`) emits, per selected
(query, kv-block) edge, a `split_slot` partial: `O_partial[split_slot, q, h, :]`
(the block-local softmax(scale*Q.Kᵀ)*V) and `LSE_partial[split_slot, q, h]` (the
block's natural-log log-sum-exp).  A query that selected `C` distinct blocks has
its `C` partials in slots `0..C-1`.  This kernel reduces them across slots into
the final O via the standard streaming-softmax merge:

  m = max_s LSE_s;  w_s = exp(LSE_s - m);  Z = sum_s w_s
  O = (1/Z) * sum_s w_s * O_partial[s]   (fp32 accum -> BF16)
  LSE = ln(Z) + m

`C` comes from `split_counts[b, qloc, h]`.  Degenerate `C == 0` (no block
selected this query) or `Z == 0`/non-finite -> O = 0, LSE = -inf (the
numerator-zero convention the forward + oracle share).

Row-packed grid: one CTA reduces a `tile_m = 64` tile of flat (q,h) rows (batch
on grid.z), ~64x fewer CTAs than one-per-(q,h).  O_partial is cp.async-pipelined;
LSE is one cp.async load with the `s < count` mask folding unused slots to -inf;
the split reduce is a warp-shuffle in the base-2 domain the fwd emits, with a
`max_valid_split` short-circuit.  The fwd stored each natural depth col at a
STG.128 fake col (see `real_to_fake`), so the write-back does a fake->real SMEM
scatter then STG.128, landing O in natural depth order.
"""

from std.bit import next_power_of_two
from std.math import ceildiv, exp2, log
from std.math.constants import log2e
from std.memory import bitcast
from std.sys import size_of

from std.gpu import barrier, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, FuncAttribute
from std.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from std.gpu.primitives.grid_controls import (
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
import std.gpu.primitives.warp as warp

from std.utils.fast_div import FastDiv
from std.utils.numerics import min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# O_partial column permute (fwd<->combine contract)
# ===-----------------------------------------------------------------------===#
#
# The fwd epilogue owns a strided set of depth columns per thread (frag_col +
# col*8 pairs); a natural-column store is a scatter that's L1TEX-bound.  Instead
# it packs its 32 owned BF16 into 16B STG.128 at a FAKE column = real_to_fake of
# the natural column, which makes each lane-group's 32 owned cols land in 32
# contiguous fake columns (4 coalesced STG.128).  The combine reads back at the
# same fake position so the merged O[d] stays in natural depth order.  Both sides
# MUST use this one pair -- if they disagree the output is silently permuted.
# Verified bijective over [0,128) (see the comptime asserts below).
@always_inline
def real_to_fake(c: Int) -> Int:
    var fc = (c % 8) - (c % 2)
    var col = c // 8
    var cg = col // 4
    var s = col % 4
    var elem = c % 2
    return cg * 32 + (fc // 2) * 8 + s * 2 + elem


@always_inline
def fake_to_real(f: Int) -> Int:
    var cg = f // 32
    var r = f % 32
    var lane = r // 8
    var s = (r % 8) // 2
    var elem = (r % 8) % 2
    return cg * 32 + s * 8 + lane * 2 + elem


def _verify_permute() -> Bool:
    # Both round-trips identity over the whole domain => the pair is a bijection
    # and a mutual inverse (no separate onto-check needed).
    comptime for c in range(128):
        comptime assert (
            fake_to_real(real_to_fake(c)) == c
        ), "fake_to_real(real_to_fake(c)) != c"
        comptime assert (
            real_to_fake(fake_to_real(c)) == c
        ), "real_to_fake(fake_to_real(c)) != c"
    return True


comptime _permute_ok = _verify_permute()


# ===-----------------------------------------------------------------------===#
# Combine geometry
# ===-----------------------------------------------------------------------===#

comptime _TILE_M = 64  # flat (q,h) rows per CTA
comptime _NUM_THREADS = 256
comptime _STAGES = 2  # O_partial cp.async ring depth


@always_inline
def combine_max_splits[topk: Int]() -> Int:
    # max_splits = next_pow2(max(topk, 2)) but never below 16.
    return max(next_power_of_two(max(topk, 2)), 16)


# ===-----------------------------------------------------------------------===#
# Combine kernel
# ===-----------------------------------------------------------------------===#


@__name(t"sm100_msa_combine_depth{depth}_{output_type}")
def _msa_combine[
    output_type: DType,
    depth: Int,
    max_splits: Int,
](
    o_ptr: UnsafePointer[
        Scalar[output_type], MutAnyOrigin
    ],  # [total_q,head_q,D]
    lse_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],  # [total_q,head_q]
    o_partial_ptr: UnsafePointer[
        Scalar[output_type], MutAnyOrigin
    ],  # [topK, total_q, head_q, D] (depth in fake-col order)
    lse_partial_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],  # [topK, total_q, head_q]
    split_counts: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [batch, max_seqlen_q, head_kv]
    cu_seqlens_q: UnsafePointer[Int32, MutAnyOrigin],  # [batch+1]
    head_q: Int,
    head_kv: Int,
    max_seqlen_q: Int,
    total_q: Int,
    num_splits: Int,  # = topk (runtime); slots num_splits..max_splits-1 unused
):
    """Row-packed split LSE merge.  One CTA == one `tile_m`-row tile of the flat
    (q_local, h) space for one batch (batch on grid.z); 256 threads.  Reduces
    `O_partial`/`LSE_partial` over slots `0..count-1` per row (`count =
    split_counts[b, qloc, kh]`) into the final O / LSE."""
    # PDL: wait for the fwd grid (which signals via launch_dependent_grids) so
    # O_partial/LSE_partial are visible before any read.  All threads, before
    # any early return -- mirrors MLA combine `wait_on_dependent_grids`.
    wait_on_dependent_grids()

    comptime tile_m = _TILE_M
    comptime nthr = _NUM_THREADS
    comptime stages = _STAGES

    # O cp.async TV map: 8 BF16 (16B) per thread, 16 threads cover one 128-col
    # row, 16 thread-rows per pass -> 4 tile-rows per thread.
    comptime gmem_thr_per_row_o = depth // 8  # 16
    comptime num_rows = tile_m // (nthr // gmem_thr_per_row_o)  # 4
    # LSE s2r: 64 threads cover the tile's 64 rows; 4 contiguous lanes own one
    # row's `max_splits` splits cooperatively -> `splits_per_thread` each.
    comptime threads_per_col = nthr // tile_m  # 4
    comptime splits_per_thread = max_splits // threads_per_col

    var tid = Int(thread_idx.x)
    var m_block = Int(block_idx.x)
    var batch_idx = Int(block_idx.z)

    var group = head_q // head_kv

    # Per-batch flat-row window.  `offset` = first abs query of this batch.
    var offset = Int(cu_seqlens_q[batch_idx])
    var seqlen = Int(cu_seqlens_q[batch_idx + 1]) - offset
    var max_idx = seqlen * head_q

    # Early exit: whole tile past this batch's flat-row count (idle CTA / tail).
    if m_block * tile_m >= max_idx:
        return

    # magic divmod -- no MUFU.RCP.  idx < 2^31 so the unsigned divmod is
    # bit-exact vs the signed // the oracle uses.
    comptime FastUInt = Scalar[FastDiv[DType.uint32].uint_type]
    var head_q_div = FastDiv[DType.uint32](head_q)
    var group_div = FastDiv[DType.uint32](group)

    var slot_row_stride = total_q * head_q  # split stride of O/LSE partials

    # ---- Dynamic SMEM: carve sLSE | sMaxValidSplit | sO ring | sO_perm -------
    # sLSE      f32  (max_splits, tile_m)         row-major (split outer)
    # sMaxValid i32  (tile_m,)
    # sO        bf16 (stages, tile_m, depth)      ring, real-contiguous fake col
    # sO_perm   bf16 (tile_m, depth+16)           +16 pad de-conflicts banks
    comptime sLSE_elems = max_splits * tile_m
    comptime sMV_elems = tile_m
    comptime sO_elems = stages * tile_m * depth
    comptime perm_stride = depth + 16

    var smem_f32 = external_memory[
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
        alignment=128,
        name="msa_combine_dynamic_shared_memory",
    ]()
    var sLSE = smem_f32  # f32
    # sMaxValidSplit packed after sLSE (both 4B); align to 128B boundary.
    comptime mv_off_f32 = ((sLSE_elems + 31) // 32) * 32
    var sMaxValid = (smem_f32 + mv_off_f32).bitcast[Int32]()
    # bf16 region starts after the f32 region (128B aligned).
    comptime bf16_base_f32 = mv_off_f32 + ((sMV_elems + 31) // 32) * 32
    var smem_bf16 = (smem_f32 + bf16_base_f32).bitcast[Scalar[output_type]]()
    var sO = smem_bf16
    var sO_perm = smem_bf16 + sO_elems

    # --- LSE_partial gmem -> smem (cp.async), s < count mask ---
    # LSE copy TV: 64 threads/pass cover the 64 tile rows; thread owns the
    # `splits_per_thread` splits {a, a+threads_per_col, ...} of its row, looping
    # the split-pass.  (1 f32 = 4B per cp.async.)
    var lse_lane = tid % tile_m  # this pass's tile row [0,64)
    var lse_pass0 = tid // tile_m  # split-group base [0,threads_per_col)
    comptime passes = max_splits // (nthr // tile_m)  # max_splits/4

    var idx_l = m_block * tile_m + lse_lane
    var dead_l = idx_l >= max_idx
    var qloc_l = 0
    var qh_l = 0
    var rcount_l = 0
    if not dead_l:
        qloc_l = Int(FastUInt(idx_l) / head_q_div)
        var h_l = idx_l - qloc_l * head_q
        qh_l = (offset + qloc_l) * head_q + h_l
        rcount_l = Int(
            split_counts[
                (batch_idx * max_seqlen_q + qloc_l) * head_kv
                + Int(FastUInt(h_l) / group_div)
            ]
        )

    comptime for p in range(passes):
        var s = lse_pass0 + p * (nthr // tile_m)  # this thread's split index
        var dst = sLSE + s * tile_m + lse_lane  # sLSE[s, lse_lane]
        if (not dead_l) and s < num_splits and s < rcount_l:
            var src = (
                lse_partial_ptr + s * slot_row_stride + qh_l
            ).address_space_cast[AddressSpace.GLOBAL]()
            async_copy[size=4](src, dst)
        else:
            dst[0] = min_or_neg_inf[DType.float32]()
    async_copy_commit_group()

    # --- Per-row metadata hoist + prologue O_partial cp.async stages ---
    # O TV: thread (tid) -> trow = tid//16, fake-col base = (tid%16)*8.  Each
    # thread owns rows {o_trow, o_trow+16, o_trow+32, o_trow+48}.  Per-row
    # metadata (valid / count / O_partial element base) is split-independent, so
    # hoist it once into scalar rmem here; the split loop just indexes it -- no
    # per-split divmod or split_counts re-LDG.
    # Plain scalar arrays indexed by comptime m, NOT a SIMD-lane stash (that
    # round trip miscompiled the qh/count lanes).
    var o_trow = tid // gmem_thr_per_row_o
    var o_fakebase = (tid % gmem_thr_per_row_o) * 8

    @parameter
    @always_inline
    def o_row_local[m: Int]() -> Int:
        return o_trow + m * (nthr // gmem_thr_per_row_o)

    var row_valid = InlineArray[Bool, num_rows](fill=False)
    var row_count = InlineArray[Int32, num_rows](fill=0)
    var row_qhd = InlineArray[Int, num_rows](fill=0)
    comptime for m in range(num_rows):
        var idx = m_block * tile_m + o_row_local[m]()
        if idx < max_idx:
            var qloc = Int(FastUInt(idx) / head_q_div)
            var h = idx - qloc * head_q
            row_valid[m] = True
            row_count[m] = split_counts[
                (batch_idx * max_seqlen_q + qloc) * head_kv
                + Int(FastUInt(h) / group_div)
            ]
            row_qhd[m] = ((offset + qloc) * head_q + h) * depth

    @parameter
    @always_inline
    def load_o_partial(split: Int, stage: Int):
        # Each owned row: one 16B cp.async (8 bf16) at the thread's fake-col run,
        # or zero-fill if this split is beyond the row's count.
        comptime for m in range(num_rows):
            var dst = (
                sO
                + stage * tile_m * depth
                + o_row_local[m]() * depth
                + o_fakebase
            )
            if row_valid[m] and split < Int(row_count[m]):
                var gbase = (
                    row_qhd[m] + split * slot_row_stride * depth + o_fakebase
                )
                var src = (o_partial_ptr + gbase).address_space_cast[
                    AddressSpace.GLOBAL
                ]()
                async_copy[size=16](src, dst)
            else:
                dst.store[width=8](SIMD[output_type, 8](0))

    comptime for stage in range(stages - 1):
        if stage < num_splits:
            load_o_partial(stage, stage)
        async_copy_commit_group()

    # --- Wait LSE + first O stage, then s2r LSE smem -> regs ---
    async_copy_wait_group(Int32(stages - 1))
    barrier()

    # This thread's row + owned splits (s2r map): row = tid//4, splits =
    # {lane_s, lane_s+4, ...}.  Load into a register SIMD.
    var s2r_row = tid // threads_per_col
    var s2r_lane = tid % threads_per_col
    var r_lse = SIMD[DType.float32, splits_per_thread](
        min_or_neg_inf[DType.float32]()
    )
    comptime for k in range(splits_per_thread):
        var s = s2r_lane + k * threads_per_col
        r_lse[k] = sLSE[s * tile_m + s2r_row]

    # --- Reduce over splits, max_valid_split, fold 1/Z into the scales ---
    comptime LOG2E = Float32(log2e)
    var lse_max = min_or_neg_inf[DType.float32]()
    comptime for k in range(splits_per_thread):
        lse_max = max(lse_max, r_lse[k])
    lse_max = warp.lane_group_max[num_lanes=threads_per_col, stride=1](lse_max)

    # Largest split COORD with a finite LSE (init -1) -> short-circuit bound.
    var max_valid = Int32(-1)
    comptime for k in range(splits_per_thread):
        var s = Int32(s2r_lane + k * threads_per_col)
        if r_lse[k] != min_or_neg_inf[DType.float32]():
            max_valid = max(max_valid, s)
    max_valid = warp.lane_group_max[num_lanes=threads_per_col, stride=1](
        max_valid
    )

    var lse_max_cur = (
        Float32(0) if lse_max == min_or_neg_inf[DType.float32]() else lse_max
    )
    var lse_sum = Float32(0)
    comptime for k in range(splits_per_thread):
        var scale = exp2(r_lse[k] * LOG2E - lse_max_cur * LOG2E)
        lse_sum += scale
        r_lse[k] = scale
    lse_sum = warp.lane_group_sum[num_lanes=threads_per_col, stride=1](lse_sum)

    var final_lse = min_or_neg_inf[DType.float32]()
    var inv_sum = Float32(0)
    if max_valid >= 0 and lse_sum > Float32(0) and lse_sum == lse_sum:
        final_lse = log(lse_sum) + lse_max
        inv_sum = Float32(1) / lse_sum
    comptime for k in range(splits_per_thread):
        r_lse[k] *= inv_sum  # normalized weight w_s/Z (0 for masked splits)

    # Write the normalized scales back to sLSE for the accumulate loop's read.
    comptime for k in range(splits_per_thread):
        var s = s2r_lane + k * threads_per_col
        sLSE[s * tile_m + s2r_row] = r_lse[k]

    # Only the s=0 owner (lane 0 of the group) writes max_valid + final LSE.
    if s2r_lane == 0:
        sMaxValid[s2r_row] = max_valid
        var idx = m_block * tile_m + s2r_row
        if idx < max_idx:
            var qloc = Int(FastUInt(idx) / head_q_div)
            var h = idx - qloc * head_q
            lse_ptr[(offset + qloc) * head_q + h] = final_lse

    barrier()

    # --- Read O_partial (ring) + fp32 accumulate ---
    var thr_max_valid = Int(-1)
    comptime for m in range(num_rows):
        if row_valid[m]:
            thr_max_valid = max(thr_max_valid, Int(sMaxValid[o_row_local[m]()]))

    var acc = SIMD[DType.float32, num_rows * 8](0)
    var stage_load = stages - 1
    var stage_compute = 0
    var s = 0
    while s <= thr_max_valid:
        # scales for this split (the normalized weights, from smem).
        var scl = SIMD[DType.float32, num_rows](0)
        comptime for m in range(num_rows):
            scl[m] = sLSE[s * tile_m + o_row_local[m]()]

        # Prefetch the next stage; ring-rotate the load slot.
        var split_to_load = s + stages - 1
        if split_to_load <= thr_max_valid:
            load_o_partial(split_to_load, stage_load)
        async_copy_commit_group()
        stage_load = 0 if stage_load == stages - 1 else stage_load + 1

        # Wait for the stage we're about to read (no barrier: own-slot only).
        async_copy_wait_group(Int32(stages - 1))
        comptime for m in range(num_rows):
            var sbase = (
                stage_compute * tile_m * depth
                + o_row_local[m]() * depth
                + o_fakebase
            )
            # One unconditional ld.shared.b128 of the owned 8-col run; guard only
            # the fp32 accumulate.  A predicated width=8 bf16 load scalarized to
            # 8x LDS.U16 -- force b128 by
            # loading 4x u32 through the bitcast pointer, reinterpret to bf16.
            var po = bitcast[output_type, 8](
                (sO + sbase).bitcast[Scalar[DType.uint32]]().load[width=4]()
            ).cast[DType.float32]()
            if row_valid[m] and scl[m] > Float32(0):
                comptime for v in range(8):
                    acc[m * 8 + v] += scl[m] * po[v]
        stage_compute = 0 if stage_compute == stages - 1 else stage_compute + 1
        s += 1

    async_copy_wait_group(0)
    barrier()

    # --- fake->real SMEM scatter (pack pairs as one 32-bit STS) ---
    comptime for m in range(num_rows):
        if row_valid[m]:
            comptime for vp in range(4):  # 8 owned cols -> 4 pairs
                var v = vp * 2
                var real = fake_to_real(o_fakebase + v)
                var pair = SIMD[output_type, 2](
                    acc[m * 8 + v].cast[output_type](),
                    acc[m * 8 + v + 1].cast[output_type](),
                )
                (sO_perm + o_row_local[m]() * perm_stride + real).store[
                    width=2
                ](pair)

    barrier()

    # --- read sO_perm (real order) + STG.128 to gmem ---
    comptime for m in range(num_rows):
        if row_valid[m]:
            # ld.shared.b128 (not 8x LDS.U16): load 4x u32 through the bitcast
            # pointer, reinterpret to the 8 owned bf16.
            var rO = bitcast[output_type, 8](
                (sO_perm + o_row_local[m]() * perm_stride + o_fakebase)
                .bitcast[Scalar[DType.uint32]]()
                .load[width=4]()
            )
            # O lands in natural depth order after the fake->real round-trip;
            # the thread owns the 8-col run at o_fakebase. One STG.128 (invert
            # the load's u32x4 reinterpret) instead of 8x STG.U16.
            (o_ptr + row_qhd[m] + o_fakebase).bitcast[
                Scalar[DType.uint32]
            ]().store[width=4](bitcast[DType.uint32, 4](rO))


# ===-----------------------------------------------------------------------===#
# Dispatch
# ===-----------------------------------------------------------------------===#


@always_inline
def msa_combine_dispatch[
    output_type: DType,
    //,
    depth: Int,
    max_splits: Int = 32,
](
    o: DeviceBuffer[output_type],  # [total_q, head_q, depth]
    lse: DeviceBuffer[DType.float32],  # [total_q, head_q]
    o_partial: DeviceBuffer[output_type],  # [topK, total_q, head_q, depth]
    lse_partial: DeviceBuffer[DType.float32],  # [topK, total_q, head_q]
    split_counts: UnsafePointer[Int32, MutAnyOrigin],
    cu_seqlens_q: UnsafePointer[Int32, MutAnyOrigin],
    batch: Int,
    head_q: Int,
    head_kv: Int,
    max_seqlen_q: Int,
    total_q: Int,
    topk: Int,  # = num_splits (runtime); must be <= max_splits
    ctx: DeviceContext,
) raises:
    """Launches the MSA combine: grid `(ceil(max_seqlen_q*head_q/64), 1,
    batch)`, one CTA per row-tile per batch, 256 threads each.  Reduces the
    `O_partial`/`LSE_partial` split slots into the final O / LSE.  `max_splits`
    is the comptime SMEM/register sizing (>= the runtime `topk`); the default 32
    covers all currently dispatched topk values."""
    comptime tile_m = _TILE_M
    comptime nthr = _NUM_THREADS
    comptime stages = _STAGES
    comptime assert (
        tile_m * max_splits
    ) % nthr == 0, "(tile_m * max_splits) must be a multiple of num_threads"
    comptime assert depth % 8 == 0, "depth must be a multiple of 8 (STG.128)"

    # max_splits sizes the SMEM/regs; a larger runtime topk silently drops the
    # tail split slots.
    debug_assert(topk <= max_splits, "msa_combine: topk exceeds max_splits")

    # Dynamic SMEM byte budget (matches the kernel's carve).
    comptime mv_off = ((max_splits * tile_m + 31) // 32) * 32 * 4
    comptime bf16_base = mv_off + ((tile_m + 31) // 32) * 32 * 4
    comptime sO_b = stages * tile_m * depth * size_of[Scalar[output_type]]()
    comptime perm_b = tile_m * (depth + 16) * size_of[Scalar[output_type]]()
    comptime smem_bytes = bf16_base + sO_b + perm_b

    var grid_x = ceildiv(max_seqlen_q * head_q, tile_m)

    ctx.enqueue_function[_msa_combine[output_type, depth, max_splits]](
        o.unsafe_ptr(),
        lse.unsafe_ptr(),
        o_partial.unsafe_ptr(),
        lse_partial.unsafe_ptr(),
        split_counts,
        cu_seqlens_q,
        head_q,
        head_kv,
        max_seqlen_q,
        total_q,
        topk,
        grid_dim=(grid_x, 1, batch),
        block_dim=(nthr, 1, 1),
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        ),
        attributes=pdl_launch_attributes(),
    )
