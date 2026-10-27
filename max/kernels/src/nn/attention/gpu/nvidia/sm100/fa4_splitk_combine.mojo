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
"""FA4 SM100 traditional (workspace / unfused) split-K combine kernel.

Merges the `P` per-partition partial outputs the FA4 1Q attention kernel wrote
to a global workspace into the final attention output, using fused Log-Sum-Exp
(LSE) weights for numerical stability.

Workspace layout (produced by the attention kernel's `do_partition` egress):
  * `o_partial`   : `[P, num_rows_q, num_q_heads, ov_depth]` (intermediate
                    dtype), each partition holds its LOCALLY-normalized
                    `O_p / l_p`.
  * `lse_partial` : `[P, num_rows_q, num_q_heads]` (f32), fused per-row LSE in
                    log2 domain (`lse_p = log2(l_p) + m_p`).

Output layout (matches the non-split ragged store): `[num_rows_q, num_q_heads,
ov_depth]`.

Per output row-head `(token, q_head)` (flattened to `rh = token*num_q_heads +
q_head`), reducing over partitions `p`:
  * `m*         = max_p lse_p`
  * `global_lse = log2(Sum_p exp2(lse_p - m*)) + m*`
  * `scale_p    = exp2(lse_p - global_lse)` (folds the `1 / denom`
                  normalization in: `exp2(lse - log2(denom) - m) ==
                  exp2(lse - m) / denom`)
  * `O[d]       = Sum_p scale_p * o_partial[p, rh, d]`

Everything stays in the log2 (base-2) domain to match the FA4 softmax
(`exp2`/`log2`); crossing into natural-exp would silently corrupt the result.

## Perf structure (adapted from `mla_decode_combine.mojo`'s
`mla_combine_kernel` / `mla_combine_kernel_split_parallel`)

`_COMBINE_WARPS` warps per output row-head, each folding every
`_COMBINE_WARPS`-th partition, summed through shared memory at the end.
Within a warp `ov_depth` is split across lanes via a vectorized `vec_size` /
`elems_per_thread` partition (128-bit loads where `ov_depth` and the
intermediate dtype allow it). The LSE reduction (row max, log-sum-exp,
per-partition `scale_p`) is computed ONCE into per-lane registers -- not the
3x redundant global-memory re-read (once for the max, once for the
denominator, once per depth-element inside the accumulation loop) the original
correctness-first version did. The weighted accumulation loop broadcasts each
partition's scale from its owning lane via `warp.shuffle_idx` and prefetches
the warp's NEXT partition's `o_partial` vector into registers while the
CURRENT partition's contribution is folded into the running sum (software
pipelining), so the load latency of the next partition overlaps the FMA work
of the current one.

Two kernel variants share this body, selected by the comptime `P_STATIC`
parameter:
  * `P_STATIC > 0`: `num_partitions` landed on a rung of the shared
    `splitk_p_ladder` that `dispatch.mojo`'s `_bucket_ws` snaps production `P`
    onto. Both the LSE and accumulation loops are `comptime for`-unrolled over
    the exact `P_STATIC`, which also keeps the per-lane `local_lse` array
    register-indexed (`divmod(p, WARP_SIZE)` folds at comptime).
  * `P_STATIC == 0`: an off-rung `P`. Reachable two ways -- a test force-knob,
    and `_bucket_ws` capping its bucketed value at `ws_p_ceiling`, which is
    not itself a rung (B200: 37 through the sweep, then decaying). The same vectorized
    + prefetched accumulation shape runs over a RUNTIME `num_partitions` bound,
    paying a dynamically-indexed `local_lse`. The array is sized at a fixed
    comptime ceiling (`P_MAX`, covering every current-generation `sm_count`) so
    this fallback compiles exactly once, independent of the `P` it is handed.

Target hardware family: NVIDIA SM100 (B200 / B300).

PDL (Item 4): this kernel is a Programmatic Dependent Launch *consumer* of the
attention producer. It calls `wait_on_dependent_grids()` (all threads of its
block, before the `rh >= RH` guard) and launches with
`pdl_launch_attributes(MHA_PDL_LEVEL)`. The attention producer does NOT emit a
terminal `launch_dependent_grids()` for the `do_partition` config (it suppresses
its prologue trigger instead, see kernel.mojo); this `wait` therefore releases on
the producer's grid *completion*, which orders after the workspace egress store.
That completion-based release (rather than a post-store terminal trigger) is
deliberate: a legal trigger must be issued by ALL threads of the CTA
(`grid_controls.mojo:105`), but the attention kernel's warp-specialized invalid-tile
early returns leave no divergence-free post-store convergence point, so a terminal
trigger would require restructuring the verified control flow. The PROGRAMMATIC_
STREAM_SERIALIZATION attribute still co-schedules this combine so its launch +
prologue overlap attention's tail (measured 1.06-1.13x on P=148 combine-heavy
shapes). With `-D MHA_PDL=false` all of this is comptime-pruned back to plain
stream ordering. (An earlier attempt -- a post-store terminal trigger at prologue
AND terminal -- double-issued `launch_dependent_grids()` and tripped
CUDA_ERROR_ILLEGAL_INSTRUCTION; the current design issues no producer trigger for
this config at all.)
"""

from std.math import ceildiv, exp2, log2, max, min
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx, warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.memory import unsafe_stack_allocation
from std.sys import align_of, get_defined_int, size_of
from std.utils.numerics import min_or_neg_inf
from nn.attention.gpu.nvidia.sm100.attention import MHA_PDL_LEVEL
from nn.attention.gpu.nvidia.sm100.attention_utils import splitk_p_ladder

import max.gpu.primitives.warp as warp

# LSE and reduction accumulate in f32 (matches the workspace `lse_partial`
# dtype and the FA4 softmax accumulator).
comptime _ACC = DType.float32

# Comptime ceiling for the runtime-`P` (`P_STATIC == 0`) fallback's per-lane
# LSE register array. Covers every production SM count today (B200=148,
# B300=160) with headroom, independent of the actual runtime `num_partitions`
# -- the fallback kernel is compiled exactly once regardless of the `P` it is
# asked to reduce at launch time. See `fa4_splitk_combine` for the ladder
# that routes production shapes to a `P_STATIC > 0` specialization instead.
#
# Overridable via `-D FA4_COMBINE_P_MAX=<N>`: the LSE load loop, the
# max/sum reductions, and the accumulation loop below are all sized off this
# constant, so raising it correctly covers a larger off-ladder `num_partitions`
# (at the cost of a bigger `local_lse` register array shared by every
# off-ladder shape compiled into the same binary). The default is left at
# `160` -- unchanged production behavior -- and `fa4_splitk_combine` asserts
# against whatever value is in effect so an un-raised ceiling fails loudly
# instead of silently corrupting the reduction or reading out of bounds.
comptime P_MAX = get_defined_int["FA4_COMBINE_P_MAX", 160]()

# Warps per row-head. Each warp folds every `_COMBINE_WARPS`-th partition, so
# the dependent L2-latency chain is `P / _COMBINE_WARPS` loads long, not `P`.
comptime _COMBINE_WARPS = 8


@__name(
    t"sm100_mha_splitk_combine_d{ov_depth}_{output_type}_{intermediate_type}_p{P_STATIC}",
)
def _fa4_splitk_combine_kernel[
    output_type: DType,
    ov_depth: Int,
    intermediate_type: DType = output_type,
    P_STATIC: Int = 0,
](
    output_ptr: MutPointer[Scalar[output_type], MutAnyOrigin],
    o_partial_ptr: MutPointer[Scalar[intermediate_type], MutAnyOrigin],
    lse_partial_ptr: MutPointer[Scalar[_ACC], MutAnyOrigin],
    num_partitions: UInt32,
    rows_heads: UInt32,
):
    """Reduces `P` per-partition workspace partials into one output row-head.

    Launched with `_COMBINE_WARPS` warps per output row-head. Each warp loads
    every partition's fused LSE into per-lane registers, computes the row's
    max and log-sum-exp, then accumulates the vectorized, prefetched,
    scale-weighted sum of its share of the partitions; the warps' sums are
    added through shared memory.

    Parameters:
        output_type: Element type of the final output buffer.
        ov_depth: Attention output head depth. Depths that tile exactly
            across the warp with a POWER-OF-2 vector width (every production
            depth in `[64, 256]` that is a multiple of `WARP_SIZE`: 64/128/256)
            use the vectorized path; depths with a non-power-of-2 `vec_size`
            (96, 160, 192, 224) or a tail (72, 80) fall to a correct
            lane-strided scalar path.
        intermediate_type: Element type of the `o_partial` workspace buffer
            (defaults to `output_type`).
        P_STATIC: Compile-time partition count to unroll over. `0` selects
            the runtime-`num_partitions` fallback (defaults to `0`).

    Args:
        output_ptr: Pointer to the final output buffer, `[rows_heads,
            ov_depth]`.
        o_partial_ptr: Pointer to the per-partition workspace output,
            `[P, rows_heads, ov_depth]`.
        lse_partial_ptr: Pointer to the per-partition fused LSE workspace,
            `[P, rows_heads]`, f32, log2 domain.
        num_partitions: Runtime partition count to reduce. Must equal
            `P_STATIC` when `P_STATIC > 0`.
        rows_heads: `num_rows_q * num_q_heads`, the output row-head count.
    """
    # `ov_depth` need not be a multiple of `WARP_SIZE`. Production depths
    # (64/128/256) tile EXACTLY across the warp as vectorized 128-bit loads
    # (`full_vec`, derived below) with a power-of-2 width; depths whose
    # `vec_size` is non-power-of-2 (96/160/192/224: a `width=3`/`5`/`6`/`7`
    # bf16 op lowers to a misaligned 4-byte sub-store) or with a tail (72, 80)
    # fall to a correct lane-strided scalar path.

    # PDL (consumer/acquire). Issued by ALL threads of the block (one warp,
    # `WARP_SIZE`) uniformly, BEFORE the `rh >= RH` guard, per the primitive
    # contract ("must be called by all threads in a thread block"). Blocks
    # until the attention producer signals — for the `do_partition` config that
    # producer suppresses its prologue launch-dependents (kernel.mojo), so this
    # `wait` releases on the producer's COMPLETION, by which point the egress
    # store to `o_partial`/`lse_partial` has landed. With `MHA_PDL=false` this
    # is comptime-pruned and the combine runs under plain stream ordering.
    comptime if MHA_PDL_LEVEL > PDLLevel.OFF:
        wait_on_dependent_grids()

    var rh: Int = block_idx.x
    var RH: Int = Int(rows_heads)
    if rh >= RH:
        return

    comptime assert WARP_SIZE % _COMBINE_WARPS == 0
    var lane_idx: Int = Int(lane_id())
    var warp_idx: Int = Int(warp_id())
    var partition_stride: Int = RH * ov_depth

    # `o_partial` vectorization: 128-bit loads where `ov_depth` and the
    # intermediate dtype allow it (matches `mla_decode_combine.mojo`'s
    # `mla_combine_kernel_split_parallel`, a single warp covering the full
    # depth).
    comptime max_vec = ov_depth // WARP_SIZE
    comptime vec_size = min(max_vec, 16 // size_of[intermediate_type]())
    comptime elems_per_thread = ov_depth // (WARP_SIZE * vec_size)
    comptime in_alignment = align_of[SIMD[intermediate_type, vec_size]]()
    comptime out_alignment = align_of[SIMD[output_type, vec_size]]()

    # `full_vec` is True iff the warp-vectorized tiling covers `ov_depth`
    # EXACTLY with a POWER-OF-2 vector width. Every production depth in
    # [64, 256] that is a multiple of `WARP_SIZE` (64/128/256) yields a
    # power-of-2 `vec_size` (2/4/8) and vectorizes cleanly. depth=96 is a
    # multiple of `WARP_SIZE` but yields `vec_size = 3` (non-power-of-2): a
    # `width=3` bf16 load/store is a 6-byte op whose LLVM lowering includes a
    # 4-byte sub-store at an address only 2-byte aligned for odd lanes ->
    # `CUDA_ERROR_MISALIGNED_ADDRESS`. depths 72/80 already fall to the
    # scalar path via the exact-cover check. Requiring a power-of-2
    # `vec_size` routes depth=96 (and 160/192/224) to the correct scalar
    # lane-strided path too. The scalar path owns depth indices
    # `{lane, lane+WARP_SIZE, ...}`, at most `max_d_per_lane` per lane.
    comptime full_vec = (
        WARP_SIZE * vec_size * elems_per_thread == ov_depth
        and (vec_size & (vec_size - 1)) == 0
    )
    comptime max_d_per_lane = ceildiv(ov_depth, WARP_SIZE)

    var oaccum_base = o_partial_ptr.unsafe_offset(rh * ov_depth).as_imm()

    # Effective partition count: the comptime `P_STATIC` when this is a
    # ladder-rung specialization, otherwise the runtime `num_partitions`.
    var p_count: Int = Int(num_partitions)
    comptime if P_STATIC > 0:
        p_count = P_STATIC

    # Load this row's fused LSE into per-lane registers once; the scale
    # derivation below overwrites them in place.
    comptime num_lse_per_thread = ceildiv(
        P_STATIC, WARP_SIZE
    ) if P_STATIC > 0 else ceildiv(P_MAX, WARP_SIZE)

    var local_lse = Array[Scalar[_ACC], num_lse_per_thread](
        fill=min_or_neg_inf[_ACC]()
    )

    comptime for k in range(num_lse_per_thread):
        comptime p_base = k * WARP_SIZE
        var p = p_base + lane_idx
        if p < p_count:
            local_lse[k] = lse_partial_ptr[p * RH + rh]

    var thread_max: Scalar[_ACC] = local_lse[0]
    comptime for k in range(1, num_lse_per_thread):
        thread_max = max(thread_max, local_lse[k])
    var m = warp.max(thread_max)

    # All-empty row (every partition's LSE is `-inf`, e.g. a fully
    # out-of-window split-K partition set): `lse_p - m* == -inf - -inf ==
    # NaN` would poison the reduction below. A valid query token always
    # attends >= 1 key, so this should not fire in production, but empty
    # partitions make it reachable -- short-circuit to a 0 write before
    # doing any further (wasted) reduction work. Padding lanes/slots beyond
    # `p_count` are ALSO `-inf` (never overwritten above), so they never
    # mask a genuinely non-empty row: the max only reads `-inf` here when
    # every REAL partition is `-inf` too.
    if m == min_or_neg_inf[_ACC]():
        comptime if full_vec:
            comptime for i in range(elems_per_thread):
                output_ptr.unsafe_store[
                    width=vec_size, alignment=out_alignment
                ](
                    rh * ov_depth
                    + lane_idx * vec_size
                    + i * (WARP_SIZE * vec_size),
                    SIMD[output_type, vec_size](0),
                )
        else:
            comptime for j in range(max_d_per_lane):
                var d = lane_idx + j * WARP_SIZE
                if d < ov_depth:
                    output_ptr[rh * ov_depth + d] = Scalar[output_type](0)
        return

    var thread_sum: Scalar[_ACC] = 0
    comptime for k in range(num_lse_per_thread):
        thread_sum += exp2(local_lse[k] - m)
    var sum_exp = warp.sum(thread_sum)
    var global_lse = log2(sum_exp) + m

    # Overwrite the LSE registers in place with each partition's scale
    # factor -- no separate storage, no shared memory. Lanes/slots beyond
    # `p_count` still hold `-inf`, and `exp2(-inf - global_lse) == 0`, so
    # they naturally contribute nothing below.
    #
    # The `is_valid` (`scale != 0`) select below is NOT defensive -- it is the
    # sole reason `o_partial` needs no initialization. `launch_workspace`
    # (dispatch.mojo) deliberately allocates the workspace WITHOUT filling it,
    # because the attention kernel's empty-partition handler skips the O store
    # while still writing `lse_p = -inf`. Every such slot therefore reaches here
    # as uninitialized memory paired with `scale == 0`, and the select is what
    # substitutes a literal 0 instead of evaluating `0 * NaN`. Removing it
    # reintroduces nondeterministic NaN output; `-D FA4_WS_POISON=1` NaN-fills
    # the workspace to keep that honest.
    comptime for k in range(num_lse_per_thread):
        local_lse[k] = exp2(local_lse[k] - global_lse)

    # Per-warp partial sums; the scales are already globally normalized, so
    # the cross-warp reduction is a plain sum.
    var smem = unsafe_stack_allocation[
        _COMBINE_WARPS * ov_depth, _ACC, address_space=.SHARED
    ]()
    var smem_warp = smem.unsafe_offset(warp_idx * ov_depth)

    # Scale-weighted accumulation across partitions, then store. `full_vec`
    # depths (power-of-2 `vec_size`, exact cover: 64/128/256) use the vectorized
    # + prefetched path; non-power-of-2-width (96/160/192/224) and tail (72, 80)
    # depths use a correct lane-strided scalar path.
    #
    # DO NOT merge the `P_STATIC > 0` and runtime-`P` arms below into one shared
    # loop body. They read as copy-paste, but the static arm resolves
    # `divmod(p, WARP_SIZE)` at COMPTIME, which is what keeps `local_lse[k]` a
    # register access. Threading `p` through a shared helper makes `k` runtime,
    # turning that register array into a dynamically-indexed local-memory array
    # -- the exact cost the static specialization exists to avoid.
    comptime if full_vec:
        var datas = Array[SIMD[intermediate_type, vec_size], elems_per_thread](
            fill=SIMD[intermediate_type, vec_size](0)
        )
        if warp_idx < p_count:
            comptime for i in range(elems_per_thread):
                datas[i] = oaccum_base.unsafe_load[
                    width=vec_size, alignment=in_alignment
                ](
                    warp_idx * partition_stride
                    + lane_idx * vec_size
                    + i * (WARP_SIZE * vec_size)
                )

        var result = Array[SIMD[_ACC, vec_size], elems_per_thread](
            fill=SIMD[_ACC, vec_size](0)
        )

        @inline(.always)
        def fold(p: Int, scale: Scalar[_ACC]) {mut datas, mut result, imm}:
            var is_valid = SIMD[.bool, vec_size](fill=scale != Scalar[_ACC](0))
            comptime for i in range(elems_per_thread):
                var clean = is_valid.select(
                    datas[i].cast[_ACC](), SIMD[_ACC, vec_size](0)
                )
                result[i] = result[i] + scale * clean
                if p + _COMBINE_WARPS < p_count:
                    datas[i] = oaccum_base.unsafe_load[
                        width=vec_size, alignment=in_alignment
                    ](
                        (p + _COMBINE_WARPS) * partition_stride
                        + lane_idx * vec_size
                        + i * (WARP_SIZE * vec_size)
                    )

        comptime if P_STATIC > 0:
            # `_COMBINE_WARPS` divides `WARP_SIZE`, so a step's LSE register
            # index `k` is comptime and only the source lane is runtime.
            comptime for j in range(ceildiv(P_STATIC, _COMBINE_WARPS)):
                comptime p_base = j * _COMBINE_WARPS
                comptime k = p_base // WARP_SIZE
                var p = p_base + warp_idx
                if p < P_STATIC:
                    fold(
                        p,
                        warp.shuffle_idx(
                            local_lse[k],
                            UInt32(p_base % WARP_SIZE + warp_idx),
                        ),
                    )
        else:
            for p in range(warp_idx, p_count, _COMBINE_WARPS):
                var k, src_lane = divmod(p, WARP_SIZE)
                fold(p, warp.shuffle_idx(local_lse[k], UInt32(src_lane)))

        comptime for i in range(elems_per_thread):
            smem_warp.unsafe_store(
                lane_idx * vec_size + i * (WARP_SIZE * vec_size), result[i]
            )
    else:
        # Scalar lane-strided fallback for odd depths (72, 80) the warp cannot
        # evenly vectorize. Correct but unvectorized -- these are non-production
        # test depths, so perf is irrelevant. Each lane owns depth indices
        # `{lane, lane+WARP_SIZE, ...} < ov_depth`. The `scale == 0` skip is the
        # scalar spelling of the vectorized path's `is_valid` select, and carries
        # the same no-init contract: it is what keeps an uninitialized
        # `o_partial` slot from injecting `0 * NaN`, not an optional guard.
        var acc = Array[Scalar[_ACC], max_d_per_lane](fill=Scalar[_ACC](0))
        for p in range(warp_idx, p_count, _COMBINE_WARPS):
            var k, src_lane = divmod(p, WARP_SIZE)
            var scale = warp.shuffle_idx(local_lse[k], UInt32(src_lane))
            if scale != Scalar[_ACC](0):
                comptime for j in range(max_d_per_lane):
                    var d = lane_idx + j * WARP_SIZE
                    if d < ov_depth:
                        acc[j] = (
                            acc[j]
                            + scale
                            * oaccum_base[p * partition_stride + d].cast[_ACC]()
                        )

        comptime for j in range(max_d_per_lane):
            var d = lane_idx + j * WARP_SIZE
            if d < ov_depth:
                smem_warp[d] = acc[j]

    barrier()
    for d in range(Int(thread_idx.x), ov_depth, _COMBINE_WARPS * WARP_SIZE):
        var total = Scalar[_ACC](0)
        comptime for w in range(_COMBINE_WARPS):
            total += smem[w * ov_depth + d]
        output_ptr[rh * ov_depth + d] = total.cast[output_type]()


@inline(.always)
def fa4_splitk_combine[
    output_type: DType,
    ov_depth: Int,
    intermediate_type: DType = output_type,
](
    ctx: DeviceContext,
    output_ptr: MutPointer[Scalar[output_type], MutAnyOrigin],
    o_partial_ptr: MutPointer[Scalar[intermediate_type], MutAnyOrigin],
    lse_partial_ptr: MutPointer[Scalar[_ACC], MutAnyOrigin],
    num_partitions: UInt32,
    num_rows_q: UInt32,
    num_q_heads: UInt32,
) raises:
    """Launches the workspace split-K combine over `num_rows_q * num_q_heads`
    output row-heads, `_COMBINE_WARPS` warps each.

    Dispatches to a `P_STATIC > 0` comptime-unrolled kernel specialization
    when `num_partitions` lands on the same partition-count ladder
    `dispatch.mojo`'s `_bucket_ws` uses to pick production `P`; otherwise
    falls through to a `P_STATIC == 0` runtime-`P` kernel that still gets
    the vectorized + prefetched accumulation, just not the full unroll.

    Parameters:
        output_type: Element type of the final output buffer.
        ov_depth: Attention output head depth.
        intermediate_type: Element type of the `o_partial` workspace buffer
            (defaults to `output_type`, matching every current call site).

    Args:
        ctx: Device context used to enqueue the kernel.
        output_ptr: Pointer to the final output buffer.
        o_partial_ptr: Pointer to the per-partition workspace output.
        lse_partial_ptr: Pointer to the per-partition fused LSE workspace.
        num_partitions: Runtime partition count to reduce (the ACTUAL count;
            see `dispatch.mojo`'s `launch_workspace` for why this can differ
            from the over-launched grid's `max_num_partitions`).
        num_rows_q: Number of query rows.
        num_q_heads: Number of query heads.
    """
    var rows_heads: UInt32 = num_rows_q * num_q_heads
    comptime nthreads = _COMBINE_WARPS * WARP_SIZE

    # The SHARED rung ladder `dispatch.mojo`'s `_bucket_ws` snaps production `P`
    # onto, so every rung gets a comptime-unrolled specialization here. Reading
    # it from `splitk_p_ladder` rather than restating it is what keeps the two
    # sides from drifting -- a hand-copied list already lost `_bucket_ws`'s
    # sub-12 rungs once, silently demoting production `P in {4,6,8,10}` to the
    # generic runtime-`P` kernel below.
    #
    # Note `_bucket_ws` can still return an OFF-rung `P`: it caps its bucketed
    # value at `ws_p_ceiling`, and that ceiling is not itself a rung (B200:
    # 37 through the sweep, then decaying). Those shapes land on the fallback.
    comptime sm_count = ctx.default_device_info.sm_count

    @inline(.always)
    def enqueue[P_STATIC: Int]() raises {imm}:
        comptime kernel = _fa4_splitk_combine_kernel[
            output_type, ov_depth, intermediate_type, P_STATIC=P_STATIC
        ]
        ctx.enqueue_function[kernel](
            output_ptr,
            o_partial_ptr,
            lse_partial_ptr,
            num_partitions,
            rows_heads,
            grid_dim=(Int(rows_heads), 1, 1),
            block_dim=(nthreads, 1, 1),
            attributes=pdl_launch_attributes(MHA_PDL_LEVEL),
        )

    comptime _COMBINE_P_LADDER = splitk_p_ladder[sm_count]()
    comptime for C in _COMBINE_P_LADDER:
        if num_partitions == UInt32(C):
            enqueue[C]()
            return

    # Off-ladder `P`: fall through to the runtime-`P` kernel (compiled once,
    # shared by every off-rung shape). Its LSE array/loops only cover
    # `[0, P_MAX)` -- fail loudly instead of silently corrupting the
    # reduction (or reading OOB) if a caller's `P` exceeds the ceiling.
    debug_assert[assert_mode="safe"](
        num_partitions <= UInt32(P_MAX),
        "fa4_splitk_combine: off-ladder num_partitions=",
        num_partitions,
        " exceeds the runtime-P fallback's P_MAX=",
        P_MAX,
        (
            " ceiling. Add a `splitk_p_ladder` rung, or raise the ceiling with"
            " -D FA4_COMBINE_P_MAX=<N>."
        ),
    )
    enqueue[0]()
