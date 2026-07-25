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

"""Phase A correctness matrix: SM100 FA4 warp-specialized (BM=32, MMA_M=32).

Target hardware family: NVIDIA SM100 (B200).

## What this guards

The whole `.ws` MMA_M=32 datapath: the shared KV sub-tile ring (producer +
consumer), the packed-TMEM P@V, the per-quarter WS softmax, and the two-level
(per-WG Level-1 + cross-WG Level-2) LSE combine wired into `fa4_softmax`'s
epilogue (Md). A very short prompt (`valid_length <= 32`) routes the dispatch
(sm100/dispatch.mojo) to the `BM=32` WS config; the output is compared against
`mha_gpu_naive` over the full `[0, num_keys)` range with the SAME mask.

The `T = ceil(num_keys / 256)` WS-tile count (driven by `cache_length`) exercises
both epilogue arms and the odd-tail alias-swap:
  * **T==1** (`num_keys <= 256`, one 256-key WS tile): WG0 alone runs the 4-way
    `fa4_ws_intracta_combine` fast path (WG1 takes the num_q==1 early-out).
  * **T in {2, 3}** (`256 < num_keys <= 768`): both softmax warpgroups run their
    even/odd streams and merge via the hierarchical two-level combine; T==3 fires
    the odd-tail alias-swap (mma_warp.mojo) + the combine odd-parity phase.

## T <= 3 regime cap (KNOWN T >= 4 main-loop bug)

The WS 1Q shared-ring MAIN LOOP (`main_iters >= 1`, first exercised at T >= 4)
has an illegal-address bug: `fa4_mma`'s steady-state full body
(mma_warp.mojo ~659-724) and `fa4_load`'s producer main loop (~680-700) are dead
for T <= 3 (peel + odd tail only). B200 confirmed: T{1,3} green across masks and
depths, but T==5 (all-full nk=1280) faults `CUDA_ERROR_ILLEGAL_ADDRESS`. Until
that is fixed, sm100/dispatch.mojo caps the WS route at `max_cache_valid_length
<= 3*BN` (T <= 3); longer KV routes to the proven non-WS path. This test stays
within T <= 3 for the WS-exercising cases; a T==5 shape is included ONLY to
confirm the dispatch cap routes it off WS (it compares green via the non-WS path).

## The all-empty-warpgroup hypothesis: SETTLED (guard NOT needed)

The two-level combine has NO all-empty guard: a fully-empty warpgroup (every
assigned KV tile FULL_MASK-skipped, `row_max` staying at its `-inf` init) would
make Level-1 compute `exp2(-inf - -inf) = NaN`. The adversarial windowed shapes
below force one WG's whole owned-tile span out of the visible window (T==2 ->
WG0 owns only tile 0; T==3 -> WG1 owns only tile 1). B200: all pass with
`nan_count == 0` -- `start_column` alignment drops the leading empty tile(s) so
the owned-tile-span is never entirely FULL_MASK-skipped for a built-in mask.
The Level-1 all-empty guard is therefore UNNECESSARY and deliberately NOT added.

Random V keeps `block0`/`block1` distinct so a packed-TMEM depth-block column
swap shows up in `max_abs_diff` and the per-depth-block imbalance check.

## NaN detection

`max_abs_diff > atol` alone does NOT catch NaN (`NaN > x` is False in IEEE), so
an explicit `isnan` sweep flags any NaN output.

## Tolerance

`atol = 0.04`, matching `test_mha_sm100_1q_splitk_lse` (bf16-MMA + f32-softmax vs
an f32 naive over the same bf16 Q/K).

## Phase B: attention sink

`use_sink=True` cases route with `sink=True` and a `[num_q_heads]` per-head sink
tensor (values spread to ~[-2, 6), straddling the ~3-magnitude score range so the
per-head sinks exercise the clamp band `sink >< per-quarter local max` across the
64 heads). The WS 8-way key split folds the sink mass ONCE, in WG0 quarter 0
(softmax_warp.mojo `fold_sink and warp_idx == 0`), mirroring split-K partition 0;
a per-quarter fold would 4x-count it. Compared against `mha_gpu_naive[sink=True]`
over the same per-head sinks. The `SlidingWindow[96]` sink cases put the sink
carrier in an otherwise-fully-masked WG.
"""

from std.collections import Set
from std.math import ceildiv, isnan, rsqrt
from std.random import rand, random_ui64, seed
from std.sys import get_defined_int

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import (
    MHAMask,
    CausalMask,
    NullMask,
    ChunkedMask,
    SlidingWindowCausalMask,
)

from std.utils import IndexList


comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def execute_ws_bm32_test[
    mask_t: MHAMask,
    head_size: Int,
    num_q_heads: Int = 64,
    page_size: Int = 128,
    use_sink: Bool = False,
    large_sink: Bool = False,
    raise_on_fail: Bool = True,
](
    mask: mask_t, cache_length: Int, valid_length: Int, ctx: DeviceContext
) raises:
    # gpt-oss-20b-like shape: 8 kv-heads, `num_q_heads` q-heads (group =
    # num_q_heads // 8), like the split-K LSE test. `valid_length <= 32` routes
    # the dispatch to the WS BM=32 config.
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=head_size)
    comptime dtype = DType.bfloat16
    comptime kv_heads = kv_params.num_heads
    comptime group = num_q_heads // kv_params.num_heads
    comptime num_layers = 1
    comptime layer_idx = 0
    # WS KV tile is BN=256 (even -> WG0, odd -> WG1). Diagnostic only.
    comptime WS_BN = 256

    # `valid_length > 32` reaches WS only when the config is force-selected
    # (`-D FA4_FORCE_CONFIG=1`, the Phase C2 multi-tile target); under the auto
    # route (<=32) it would silently fall to the baseline path, so this would no
    # longer be a WS test. Keep the guard active for the default (auto) target.
    comptime _force_config = get_defined_int["FA4_FORCE_CONFIG", 0]()
    if valid_length > 32 and _force_config != 1:
        raise Error(
            "WS BM=32 route requires valid_length <= 32 (or -D"
            " FA4_FORCE_CONFIG=1 to force WS)"
        )

    var num_keys = cache_length + valid_length
    var total_length = valid_length
    var batch_size = 1
    var T = ceildiv(num_keys, WS_BN)

    print(
        "test_mha_sm100_ws_bm32: mask=",
        mask_t.name(),
        " depth=",
        head_size,
        " q_heads=",
        num_q_heads,
        " group=",
        group,
        " page_size=",
        page_size,
        " sink=",
        use_sink,
        " valid_length=",
        valid_length,
        " num_keys=",
        num_keys,
        " T(WS tiles)=",
        T,
        sep="",
    )

    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()
    comptime sink_layout = Layout.row_major(UNKNOWN_VALUE)

    var scale = rsqrt(Float32(head_size))

    seed(0x5151)

    # --- Row offsets (single sequence) ---
    var input_row_offsets = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    input_row_offsets[0] = 0
    input_row_offsets[1] = UInt32(valid_length)
    var input_row_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_dev, input_row_offsets)
    var input_row_offsets_lt = LayoutTensor[
        mut=False, DType.uint32, row_offsets_layout
    ](
        input_row_offsets_dev,
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    # --- Q (ragged: [total_length, num_q_heads, head_size]) ---
    var q_size = total_length * num_q_heads * head_size
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    var q_host_tt = LayoutTensor[dtype, q_ragged_layout](
        q_host.unsafe_ptr(),
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    random(q_host_tt)
    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    var q_lt = LayoutTensor[mut=False, dtype, q_ragged_layout](
        q_dev,
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )

    # --- Paged KV blocks (shared physical storage for both runs) ---
    var num_paged_blocks = ceildiv(num_keys, page_size) * batch_size + 4
    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_size,
    )
    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * head_size
    )
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](kv_block_size)
    var kv_block_host_tt = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_host.unsafe_ptr(),
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
    )
    random(kv_block_host_tt)
    var kv_block_dev = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(kv_block_dev, kv_block_host)
    var kv_block_paged_lt = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_dev,
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
    )

    # --- Full lookup table (unique physical block per logical page) ---
    var full_pages = ceildiv(num_keys, page_size)
    var lut_cols = padded_lut_cols(full_pages)
    var paged_lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size * lut_cols
    )
    var lut_set = Set[Int]()
    for blk in range(full_pages):
        var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
        while randval in lut_set:
            randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
        lut_set.add(randval)
        paged_lut_host[blk] = UInt32(randval)

    # --- Cache lengths ---
    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    cache_lengths_host[0] = UInt32(cache_length)
    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    var cache_lengths_lt = LayoutTensor[
        mut=False, DType.uint32, cache_lengths_layout
    ](
        cache_lengths_dev,
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )

    var paged_lut_dev = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * lut_cols
    )
    ctx.enqueue_copy(paged_lut_dev, paged_lut_host)
    var paged_lut_lt = LayoutTensor[mut=False, DType.uint32, paged_lut_layout](
        paged_lut_dev,
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, lut_cols)
        ),
    )

    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        # Disjoint k/v views share one blocks buffer's origin; declare the block
        # origin UnsafeAnyOrigin to opt out of the nested-origin exclusivity check
        # (mirrors test_mha_sm100_1q_splitk_lse / test_mha_sm100_1q_sink).
        kv_block_paged_lt.as_unsafe_any_origin(),
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(valid_length),
        UInt32(num_keys),
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    # --- Per-head sink weights (random when use_sink) ---
    # `num_q_heads` entries: fa4_softmax / mha_gpu_naive index the sink by query
    # head. Spread rand [0,1) to ~[-2, 6): the ~3-magnitude score range sits
    # inside this band, so across the 64 heads some have sink > a quarter's local
    # max and some below -- exercising the WG0-quarter-0 clamp band (Risk 5). The
    # same `sinks_lt` feeds both the WS run and the naive oracle.
    var sinks_host = ctx.enqueue_create_host_buffer[dtype](num_q_heads)
    if use_sink:
        rand(sinks_host.as_span())
        for h in range(num_q_heads):
            comptime if large_sink:
                # Clamp-stress (~+10): far above the ~3-magnitude scores, so the
                # sink pins Gmax and only WG0-quarter-0 (partition 0) may clamp to
                # it -- a per-quarter or per-partition over-count would show as a
                # large denominator/error vs naive. Spread [8, 12).
                sinks_host[h] = (
                    sinks_host[h].cast[DType.float32]() * Float32(4.0)
                    + Float32(8.0)
                ).cast[dtype]()
            else:
                sinks_host[h] = (
                    sinks_host[h].cast[DType.float32]() * Float32(8.0)
                    - Float32(2.0)
                ).cast[dtype]()
    else:
        sinks_host.as_span().fill(Scalar[dtype](0))
    var sinks_dev = ctx.enqueue_create_buffer[dtype](num_q_heads)
    ctx.enqueue_copy(sinks_dev, sinks_host)
    var sinks_lt = LayoutTensor[mut=False, dtype, sink_layout](
        sinks_dev.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[sink_layout].row_major(IndexList[1](num_q_heads)),
    )

    var test_out_size = total_length * num_q_heads * head_size

    # ============ Run 1: WS FA4 (BM=32) over the full cache ============
    var test_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var test_out_lt = LayoutTensor[dtype, output_layout](
        test_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    comptime if use_sink:
        flash_attention[ragged=True, sink=True](
            test_out_lt,
            q_lt,
            k_cache,
            v_cache,
            mask,
            input_row_offsets_lt,
            scale,
            ctx,
            sink_weights=sinks_lt,
        )
    else:
        flash_attention[ragged=True](
            test_out_lt,
            q_lt,
            k_cache,
            v_cache,
            mask,
            input_row_offsets_lt,
            scale,
            ctx,
        )

    # ============ Run 2: naive over the full key range [0, num_keys) ======
    var ref_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var ref_out_lt = LayoutTensor[dtype, output_layout](
        ref_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    comptime if use_sink:
        mha_gpu_naive[ragged=True, sink=True](
            q_lt,
            k_cache,
            v_cache,
            mask,
            ref_out_lt,
            input_row_offsets_lt,
            scale,
            batch_size,
            valid_length,
            num_keys,
            num_q_heads,
            head_size,
            group,
            ctx,
            sinks_lt,
        )
    else:
        mha_gpu_naive[ragged=True](
            q_lt,
            k_cache,
            v_cache,
            mask,
            ref_out_lt,
            input_row_offsets_lt,
            scale,
            batch_size,
            valid_length,
            num_keys,
            num_q_heads,
            head_size,
            group,
            ctx,
        )

    ctx.synchronize()

    var test_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    var ref_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    ctx.enqueue_copy(test_out_host, test_out_dev)
    ctx.enqueue_copy(ref_out_host, ref_out_dev)
    ctx.synchronize()

    # Track the worst error globally AND per depth block (one 64-wide combine
    # column) to catch a packed-TMEM column swap that a global atol could miss.
    comptime BLK = 64
    var max_abs_diff: Float32 = 0.0
    var argmax_idx = 0
    var blk0_max: Float32 = 0.0  # depth [0, 64)
    var blk1_max: Float32 = 0.0  # depth [64, 128), only at depth 128
    # `NaN > x` is False in IEEE, so the atol check below would silently pass a
    # NaN output -- the exact all-empty-warpgroup signature. Detect it here.
    var nan_count = 0
    var first_nan_idx = -1
    for i in range(total_length):
        for h in range(num_q_heads):
            for d in range(head_size):
                var flat = (i * num_q_heads + h) * head_size + d
                var a = test_out_host[flat].cast[DType.float32]()
                var expected = ref_out_host[flat].cast[DType.float32]()
                if isnan(a):
                    nan_count += 1
                    if first_nan_idx < 0:
                        first_nan_idx = flat
                var diff = abs(a - expected)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                    argmax_idx = flat
                if d < BLK:
                    blk0_max = max(blk0_max, diff)
                else:
                    blk1_max = max(blk1_max, diff)
    print(
        "  max-abs diff(WS O, full naive) =",
        max_abs_diff,
        " at flat idx",
        argmax_idx,
        " block0=",
        blk0_max,
        " block1=",
        blk1_max,
        " nan_count=",
        nan_count,
    )

    _ = q_dev^
    _ = kv_block_dev^
    _ = paged_lut_dev^
    _ = input_row_offsets_dev^
    _ = cache_lengths_dev^
    _ = sinks_dev^
    _ = test_out_dev^
    _ = ref_out_dev^

    comptime atol = Float32(0.04)
    var fail_reason = String("")

    # NaN output = the all-empty-warpgroup signature (Level-1 `exp2(-inf - -inf)`
    # when a WG's whole owned-tile span is FULL_MASK-skipped). Checked explicitly
    # because the atol comparison below cannot catch NaN.
    if nan_count > 0:
        fail_reason = (
            "NaN x"
            + String(nan_count)
            + " (first idx "
            + String(first_nan_idx)
            + ") -- all-empty-warpgroup combine bug"
        )
    elif max_abs_diff > atol:
        fail_reason = (
            "max-abs diff " + String(max_abs_diff) + " > atol " + String(atol)
        )
    else:
        # Per-block structural ("where") check, depth 128 only: a correct combine
        # scatters bf16 noise across both depth blocks; a packed-TMEM column
        # clobber/swap pins large error to one block. Fire only when the worse
        # block is well above the bf16 floor (> half atol) and >5x the better.
        comptime if head_size > BLK:
            var worse = max(blk0_max, blk1_max)
            var better = min(blk0_max, blk1_max)
            if worse > Float32(0.5) * atol and worse > Float32(5.0) * better:
                fail_reason = (
                    "combine error concentrated in one depth block (block0 "
                    + String(blk0_max)
                    + " vs block1 "
                    + String(blk1_max)
                    + ") -- structural combine bug (column clobber/swap)"
                )

    if fail_reason.byte_length() > 0:
        var msg = (
            "test_mha_sm100_ws_bm32: FAILED mask="
            + String(mask_t.name())
            + " depth="
            + String(head_size)
            + " num_keys="
            + String(num_keys)
            + ": "
            + fail_reason
        )
        comptime if raise_on_fail:
            raise Error(msg)
        else:
            print(msg)
    else:
        print("test_mha_sm100_ws_bm32: PASSED")


def main() raises:
    # `cache_length` drives T = ceil((cache_length + valid_length) / 256). Every
    # shape below routes to WS (max_prompt_len <= 32; the interim T<=3 cache cap
    # was removed after the T>=4 correction-SMEM fix, 2026-07-15). T ranges over
    # {1,2,3,5}, including the T==5 shape at the end that first exercised the
    # shared-ring main loop.
    with DeviceContext() as ctx:
        # ---- Cross-CTA cluster split-K (P pinned by FA4_WS_SPLITK_FORCE) -------
        # The `-D FA4_WS_SPLITK_FORCE=P` targets route the WS BM=32 config through
        # a P-CTA cluster: each CTA owns 1/P of the KV, runs the 8-way intra-CTA
        # split over its slice, then DSMEM-combines its OWN depth band (the THIRD
        # split-K level). These cells exit early so the default (force-unset)
        # matrix below stays single-CTA WS. All depth-128, vl=8, compared vs
        # mha_gpu_naive over the full [0, num_keys) range at the bf16 floor.
        # FSK is clamped to a power of two <= P_MAX (4) by dispatch, so `>= 4`
        # selects the P=4 cells and `2..3` the P=2 cells.
        comptime FSK = get_defined_int["FA4_WS_SPLITK_FORCE", 0]()
        comptime if FSK >= 2:
            comptime if FSK >= 4:
                # ===== P >= 4 (cross-CTA cluster split-K, m_pack == 4 owning
                # bands) ===== The force knob pins P in {4,6,8,10,16}; for P > 4
                # only the first `m_pack` (== 4) partitions own a depth band, while
                # partitions >= 4 still partition the KV, stage a normalized
                # partial, and are reduced into the owning bands (many empty at
                # high P: cache=1024 -> T=5 leaves P-5 empty for P>5; cache=640 ->
                # T=3 leaves P-3 empty). All compared vs mha_gpu_naive over the
                # full [0, num_keys) range at the bf16 floor.
                # (a) all-live-at-P4 / mostly-empty-at-high-P: cache=1024 -> T=5.
                execute_ws_bm32_test[NullMask, 128](
                    NullMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                execute_ws_bm32_test[CausalMask, 128](
                    CausalMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                # (b) window x split composition: the per-partition splitk_window
                #     offset composes with the per-quarter mask column offset.
                #     cache=640 -> T=3; the window/chunk spans the partition
                #     boundary so live partitions see visible keys.
                execute_ws_bm32_test[ChunkedMask[384], 128](
                    ChunkedMask[384](),
                    cache_length=640,
                    valid_length=8,
                    ctx=ctx,
                )
                execute_ws_bm32_test[SlidingWindowCausalMask[512], 128](
                    SlidingWindowCausalMask[512](),
                    cache_length=640,
                    valid_length=8,
                    ctx=ctx,
                )
                # (c) EMPTY trailing partition (L3 neutral element): cache=640 ->
                #     num_keys=648 -> T=3, T < P so trailing partitions get 0 tiles
                #     and must take the neutral element (M=-inf, L=0, finite-0 O ->
                #     w_p=0) yet still publish + write their band.
                execute_ws_bm32_test[NullMask, 128](
                    NullMask(), cache_length=640, valid_length=8, ctx=ctx
                )
                execute_ws_bm32_test[CausalMask, 128](
                    CausalMask(), cache_length=640, valid_length=8, ctx=ctx
                )
                # (d) depth-64 (num_d_tiles=1, own_cols=16): confirm no depth-128
                #     hardcode in the L3 band math -- Null/Causal (T=5) plus both
                #     window masks (T=3 window x split on the narrow depth).
                execute_ws_bm32_test[NullMask, 64](
                    NullMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                execute_ws_bm32_test[CausalMask, 64](
                    CausalMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                execute_ws_bm32_test[ChunkedMask[384], 64](
                    ChunkedMask[384](),
                    cache_length=640,
                    valid_length=8,
                    ctx=ctx,
                )
                execute_ws_bm32_test[SlidingWindowCausalMask[512], 64](
                    SlidingWindowCausalMask[512](),
                    cache_length=640,
                    valid_length=8,
                    ctx=ctx,
                )
            else:
                # ===== P=2 (re-banding 2:1: two warp-bands per partition) =====
                # Null + Causal: cache=1024 -> num_keys=1032 -> T=5,
                # splitk_window(5,2) = {3,2} tiles (both non-empty).
                execute_ws_bm32_test[NullMask, 128](
                    NullMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                execute_ws_bm32_test[CausalMask, 128](
                    CausalMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                # (b) window x split composition: the per-partition splitk_window
                # offset composes with the per-quarter mask column offset. cache
                # =640 -> num_keys=648; the window/chunk spans the partition
                # boundary (both partitions see visible keys).
                execute_ws_bm32_test[ChunkedMask[384], 128](
                    ChunkedMask[384](),
                    cache_length=640,
                    valid_length=8,
                    ctx=ctx,
                )
                execute_ws_bm32_test[SlidingWindowCausalMask[512], 128](
                    SlidingWindowCausalMask[512](),
                    cache_length=640,
                    valid_length=8,
                    ctx=ctx,
                )
                # (d) depth-64 P=2 (num_d_tiles=1, own_cols=16): re-banding 2:1 on
                #     the narrow depth. cache=1024 -> T=5, both partitions non-empty.
                execute_ws_bm32_test[NullMask, 64](
                    NullMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                execute_ws_bm32_test[CausalMask, 64](
                    CausalMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                # (e) sink, P=2: the sink mass must be folded EXACTLY ONCE across
                #     the P partitions -- the `fold_sink` partition-0 refinement
                #     (splitk_partition_idx==0) composes with the WS WG0-quarter-0
                #     gate (warp_idx==0) so ONLY partition 0's WG0-q0 folds it. A
                #     residual per-partition (Px) or per-quarter (4x) over-count
                #     inflates the denominator far above the floor. Moderate spread
                #     [-2,6) straddles the scores so the denominator is sensitive.
                execute_ws_bm32_test[CausalMask, 128, use_sink=True](
                    CausalMask(), cache_length=1024, valid_length=8, ctx=ctx
                )
                execute_ws_bm32_test[
                    SlidingWindowCausalMask[512], 128, use_sink=True
                ](
                    SlidingWindowCausalMask[512](),
                    cache_length=1024,
                    valid_length=8,
                    ctx=ctx,
                )
                # large-sink (~+10) clamp-stress x Causal (Phase B Risk 5 under the
                # finer split): Gmax is pinned to the sink, so every quarter /
                # partition must rescale to ~0 and only WG0-q0 may carry the sink.
                execute_ws_bm32_test[
                    CausalMask, 128, use_sink=True, large_sink=True
                ](CausalMask(), cache_length=1024, valid_length=8, ctx=ctx)
            return

        # ---- Null + Causal x depth{64,128} x T{1,3} -------------------------
        comptime for depth in [128, 64]:
            execute_ws_bm32_test[NullMask, depth](
                NullMask(), cache_length=192, valid_length=8, ctx=ctx
            )
            execute_ws_bm32_test[NullMask, depth](
                NullMask(), cache_length=640, valid_length=8, ctx=ctx
            )
            execute_ws_bm32_test[CausalMask, depth](
                CausalMask(), cache_length=192, valid_length=8, ctx=ctx
            )
            execute_ws_bm32_test[CausalMask, depth](
                CausalMask(), cache_length=640, valid_length=8, ctx=ctx
            )

        # ---- Windowed both-live (T==3) x depth{64,128} + GQA + page_size ------
        comptime for depth in [128, 64]:
            execute_ws_bm32_test[SlidingWindowCausalMask[512], depth](
                SlidingWindowCausalMask[512](),
                cache_length=640,
                valid_length=8,
                ctx=ctx,
            )
            execute_ws_bm32_test[ChunkedMask[384], depth](
                ChunkedMask[384](), cache_length=640, valid_length=8, ctx=ctx
            )
        execute_ws_bm32_test[CausalMask, 128, num_q_heads=32](
            CausalMask(), cache_length=640, valid_length=8, ctx=ctx
        )
        execute_ws_bm32_test[CausalMask, 128, page_size=64](
            CausalMask(), cache_length=640, valid_length=8, ctx=ctx
        )

        # ---- A3-i regression: single-WG-fully-masked (all-empty-WG probe) -----
        # The two-level WS combine's Level-1 self-normalize (exp2((m_g - m_wg) *
        # log2e)) would divide by an all-`-inf` warpgroup max -- exp2(-inf - -inf)
        # = NaN -- IF a whole WG saw zero visible keys for a row. That NaN is
        # UNREACHABLE here, for two independent reasons:
        #   (1) Masked scores are the FINITE MASK_VALUE (-10000), not -inf
        #       (attention_utils.mojo `.select(s, MASK_VALUE)`, with the -inf alt
        #       commented out at :2908). So a fully-masked-but-processed tile
        #       yields a finite m_wg that the combine's exp2((m_wg - M)*log2e)
        #       scale drives to ~0 -- the WG drops out cleanly, no NaN.
        #   (2) At T>=2 the softmax parity split gives each WG >=1 tile, so no WG
        #       ever processes zero tiles (the only other route to m_wg=-inf).
        # These cases DELIBERATELY drive one WG's entire tile span to fully-masked
        # for EVERY query row and assert nan_count==0 (raise_on_fail=True). They
        # are the regression that catches a reintroduction of the NaN -- e.g. if
        # masking is switched to -inf (the commented alt) or the parity split
        # changes. No kernel-side Level-1 guard is added (the NaN can't occur).
        #
        # SlidingWindow[96]: each row's window sits entirely inside ONE 256-key
        # tile, so the OTHER warpgroup's tile(s) are fully masked for all rows.
        #   cache_length=384: rows 384..391, window in [256,512)=tile1 (odd/WG1)
        #     => WG0 (tile0, even) fully masked for every row. num_keys=392, T=2.
        execute_ws_bm32_test[SlidingWindowCausalMask[96], 128](
            SlidingWindowCausalMask[96](),
            cache_length=384,
            valid_length=8,
            ctx=ctx,
        )
        #   cache_length=640: rows 640..647, window in [512,768)=tile2 (even/WG0)
        #     => WG1 (tile1, odd) fully masked for every row. num_keys=648, T=3.
        execute_ws_bm32_test[SlidingWindowCausalMask[96], 128](
            SlidingWindowCausalMask[96](),
            cache_length=640,
            valid_length=8,
            ctx=ctx,
        )
        # Wider-window + Chunked corroboration (window/chunk spans multiple
        # quarters of one tile; one WG still fully masked for the windowed rows):
        execute_ws_bm32_test[SlidingWindowCausalMask[128], 128](
            SlidingWindowCausalMask[128](),
            cache_length=504,
            valid_length=8,
            ctx=ctx,
        )
        execute_ws_bm32_test[SlidingWindowCausalMask[128], 128](
            SlidingWindowCausalMask[128](),
            cache_length=640,
            valid_length=8,
            ctx=ctx,
        )
        execute_ws_bm32_test[ChunkedMask[256], 128](
            ChunkedMask[256](), cache_length=504, valid_length=8, ctx=ctx
        )

        # ---- Phase B: attention sink -----------------------------------------
        # The WS 8-way split folds the sink mass ONCE (WG0 quarter 0,
        # softmax_warp.mojo `fold_sink and warp_idx == 0`); a per-quarter fold
        # would 4x-count it. Compared vs mha_gpu_naive[sink=True] over the same
        # per-head sinks. The [-2,6) per-head spread straddles the ~3-magnitude
        # scores, so max_abs_diff (worst over 64 heads) samples the WG0-q0 clamp
        # band (`sink >< per-quarter local max`, Risk 5).
        comptime for depth in [128, 64]:
            execute_ws_bm32_test[CausalMask, depth, use_sink=True](
                CausalMask(), cache_length=192, valid_length=8, ctx=ctx
            )  # T=1
            execute_ws_bm32_test[CausalMask, depth, use_sink=True](
                CausalMask(), cache_length=640, valid_length=8, ctx=ctx
            )  # T=3
            execute_ws_bm32_test[
                SlidingWindowCausalMask[512], depth, use_sink=True
            ](
                SlidingWindowCausalMask[512](),
                cache_length=640,
                valid_length=8,
                ctx=ctx,
            )  # T=3
        # Sink carried by an otherwise-fully-masked WG (SlidingWindow[96]: one
        # WG's whole tile span is masked for every row; the sink is still folded
        # in WG0 quarter 0, so its clamp fires vs the MASK_VALUE local max):
        execute_ws_bm32_test[SlidingWindowCausalMask[96], 128, use_sink=True](
            SlidingWindowCausalMask[96](),
            cache_length=384,
            valid_length=8,
            ctx=ctx,
        )  # WG0 masked, T=2
        execute_ws_bm32_test[SlidingWindowCausalMask[96], 128, use_sink=True](
            SlidingWindowCausalMask[96](),
            cache_length=640,
            valid_length=8,
            ctx=ctx,
        )  # WG1 masked, T=3

        # ---- T==5 (num_keys 1280): routes to WS (cap removed). This is the shape
        # that first exercised the shared-ring MAIN LOOP (main_iters>=1) and
        # surfaced the correction-SMEM OOB write (undersized correction region ->
        # fixed 2026-07-15 by sizing it 2*WARPGROUP_SIZE in smem.mojo). It wraps
        # the 6-slot KV ring >3x; green vs naive at the bf16 floor confirms WS
        # steady state.
        execute_ws_bm32_test[CausalMask, 128](
            CausalMask(), cache_length=1272, valid_length=8, ctx=ctx
        )

        # ---- Phase C2: multi-tile prompts (vl > 32), forced onto WS -----------
        # The auto route only sends max_prompt_len <= 32 to WS today; the widened
        # rule (C4) admits larger prompts. These cells force WS via
        # `-D FA4_FORCE_CONFIG=1` (the multi-tile variant target) and drive
        # prompts that span many BM=32 query tiles (PairBM_eff = 32//group = 4 at
        # group=8, so vl=64 -> 16 tiles, vl=128 -> 32, vl=200 -> 50) via each
        # CTA's `prompt_offset`, vs mha_gpu_naive at the bf16 floor. This proves
        # the widened envelope is CORRECT before the route admits it. Guarded
        # `== 1` so the default (auto) target is byte-identical.
        comptime if get_defined_int["FA4_FORCE_CONFIG", 0]() == 1:
            comptime for depth in [128, 64]:
                # vl=64 (16 query tiles): T=1 (num_keys=256) and T=3 (704).
                execute_ws_bm32_test[CausalMask, depth](
                    CausalMask(), cache_length=192, valid_length=64, ctx=ctx
                )
                execute_ws_bm32_test[CausalMask, depth](
                    CausalMask(), cache_length=640, valid_length=64, ctx=ctx
                )
                # vl=128 (32 query tiles), windowed, T=3 (num_keys=768).
                execute_ws_bm32_test[SlidingWindowCausalMask[512], depth](
                    SlidingWindowCausalMask[512](),
                    cache_length=640,
                    valid_length=128,
                    ctx=ctx,
                )
            # vl=200 (50 tiles) + T=4 (num_keys=840): multi-tile prompt AND the
            # shared-ring main loop (main_iters>=1) together.
            execute_ws_bm32_test[CausalMask, 128](
                CausalMask(), cache_length=640, valid_length=200, ctx=ctx
            )
