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

"""M4 verification for SM100 FA4 1Q split-K: writer-combines-all DSMEM O combine.

Target hardware family: NVIDIA SM100 (B200) — clusters + DSMEM are a B200
feature.

## What this guards

The in-cluster DSMEM O combine is unconditional for split-K. The
writer (partition 0) emits the FULL combined O over the whole `[0, num_keys)`
range, so the oracle is plain `mha_gpu_naive` over the entire cache with the
SAME mask object — no windowing, no per-partition weight. The split kernel
(Run 1) and the naive reference (Run 2) share the same mask, so the comparison
is mask- and depth-agnostic.

This file is parametrized by compile-time defines so one source drives a
`mask × depth × P` sweep (see BUILD):
  * `-D FA4_1Q_SPLITK_CACHE=N`  sets the KV cache length, which STEERS the
        dispatch's auto-selected partition count P. The production dispatch has
        no compile-time P knob (it auto-picks from occupancy + KV length); with
        this file's tiny prompt the heuristic's `by_cache` term dominates, so
        `P == floor_pow2(N // 512)`: 1024->2, 2048->4, 4096->8 (512->1). The
        test replicates the same heuristic to know the realized P.
  * `-D FA4_1Q_SPLITK_MASK=M`   selects the mask:
        0=Null, 1=Causal, 2=Chunked, 3=SlidingWindowCausal,
        4=SlidingWindowNonCausal.
  * `-D FA4_1Q_SPLITK_DEPTH=D`  selects `head_size` (64 or 128).

Null and Causal have `start_column == 0` and exact `total_iters`, so static
split-K partitioning is exact (committed coverage). The windowed masks
(Chunked / SlidingWindow*) have `start_column > 0`; they are a discovery sweep —
a pass confirms the per-partition slicing handles a nonzero lower bound, a
failure (or an empty partition at large P) points at later milestones (M5/M6).

## Tolerance

`atol = 0.04`. The kernel computes scores in bf16-MMA + f32-softmax while the
host reference casts the SAME bf16 Q/K to f32 and dots in f32; the exp2/LSE math
matches the kernel ops. The dominant error is bf16 input rounding amplified
through the softmax weights, plus the bf16 output store. Empirically < 0.04.
"""

from std.collections import Set
from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed
from std.sys import get_defined_int, get_defined_bool

from std.gpu.host import DeviceContext, DeviceAttribute
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import (
    MHAMask,
    NullMask,
    CausalMask,
    ChunkedMask,
    SlidingWindowCausalMask,
    SlidingWindowNonCausalMask,
)

from std.utils import IndexList


comptime _LUT_TAIL_PAD = 16

# Window/chunk sizes for the discovery masks. With score_row ~ 1024 and BN=128,
# SW_WINDOW=896 puts start_column at tile 1 (active keys ~ [129, 1024], ~8 active
# tiles) — a genuine start_column > 0 while keeping P=2/4 non-empty. P=8 on a
# windowed mask may yield an empty partition (active count < 8), which is an M6
# discovery, not a combine bug. CHUNK is a representative ChunkedMask window.
comptime SW_WINDOW = 896
comptime CHUNK = 512


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def execute_combine_test[
    mask_t: MHAMask, head_size: Int
](mask: mask_t, ctx: DeviceContext) raises:
    # Which partition's diagnostic window to print (W=0 = the combine writer).
    # P itself is no longer forced -- the dispatch auto-selects it; we steer it
    # via the cache length and recompute it below to know the realized value.
    comptime W = get_defined_int["FA4_1Q_SPLITK_WRITER", 0]()
    # `-D FA4_1Q_SPLITK_SINK=True` enables attention sinks on BOTH the split-K
    # run and the naive oracle. With a windowed mask + large P some partitions
    # (incl. trailing ones) are empty; the sink mass must still be folded into
    # the cluster-global denominator EXACTLY ONCE (by the always-non-empty
    # rank-0 writer), never by an empty partition. This exercises that path.
    comptime USE_SINK = get_defined_bool["FA4_1Q_SPLITK_SINK", False]()
    comptime sink_layout = Layout.row_major(UNKNOWN_VALUE)

    # gpt-oss-20b-like shape: 64 q-heads, 8 kv-heads (group 8).
    comptime num_q_heads = 64
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=head_size)
    comptime dtype = DType.bfloat16
    comptime page_size = 128
    comptime kv_heads = kv_params.num_heads
    comptime group = num_q_heads // kv_params.num_heads
    comptime num_layers = 1
    comptime layer_idx = 0
    # head_size in {64,128} / page_size=128 -> FA4 1Q BN == 128 == page_size:
    # each combined tile is exactly one cache page.
    comptime BN = 128

    var valid_length = 2
    # Shape-driven split-K: the dispatch auto-selects P from occupancy + KV
    # length (no compile-time P knob). We steer P via the cache length -- with
    # this tiny prompt the heuristic's `by_cache = max_cache_valid_length // 512`
    # term dominates, so P == floor_pow2(cache_length // 512):
    #   512 -> 1, 1024 -> 2, 2048 -> 4, 4096 -> 8.
    var cache_length = get_defined_int["FA4_1Q_SPLITK_CACHE", 1024]()
    var num_keys = cache_length + valid_length
    var total_length = valid_length
    var batch_size = 1

    # Replicate the dispatch's auto-P heuristic (sm100/dispatch.mojo) so the test
    # knows the realized cluster size. The prompt is tiny (valid_length = 2 << any
    # 2Q tile), so the 2Q grid is exactly one prompt tile and (with fuse_gqa
    # folding the 64 q-heads into 8 kv-heads) raw_grid_2q = num_kv_heads *
    # batch_size. On B200 `by_grid` is then large, so `by_cache` selects P.
    # Same inputs + same formula as the dispatch => this P IS the launched P.
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var grid_threshold = UInt32(sm_count // 2)
    var raw_grid_2q = UInt32(kv_heads * batch_size)
    var by_grid = grid_threshold // max(raw_grid_2q, UInt32(1))
    var by_cache = UInt32(cache_length) // UInt32(512)
    var np = min(min(by_grid, by_cache), UInt32(8))
    var P = Int(8 if np >= 8 else (4 if np >= 4 else (2 if np >= 2 else 1)))

    # Front-loaded balanced split (matches `splitk_window`); diagnostic only
    # here -- the combine writer emits the FULL O, so the oracle is full-range.
    var T = ceildiv(num_keys, BN)
    var q_tiles = T // P
    var r_tiles = T % P
    var cb = W * q_tiles + min(W, r_tiles)
    var ce = (W + 1) * q_tiles + min(W + 1, r_tiles)
    var win_begin = cb * BN
    var win_end = min(ce * BN, num_keys)
    var window_len = win_end - win_begin

    print(
        "test_mha_sm100_1q_splitk_combine: mask=",
        mask_t.name(),
        " depth=",
        head_size,
        " P=",
        P,
        " num_keys=",
        num_keys,
        " T=",
        T,
        " sink=",
        USE_SINK,
        sep="",
    )
    if not (T >= P):
        raise Error("test misconfigured: need T >= P (no empty partitions)")
    if not (ce > cb):
        raise Error("partition window is empty (T < P?)")

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

    # ============ Run 1: split FA4 over the FULL cache ============
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
        # `flash_attention`/`mha_gpu_naive` read both the `k` and `v` cache
        # views, which are disjoint kv_idx halves of one `blocks` buffer sharing
        # its origin, so the nested-origin exclusivity check rejects passing
        # both. Declare the block origin as UnsafeAnyOrigin to opt out of
        # exclusivity checking (mirrors `test_mha_sm100_1q_sink.mojo`).
        kv_block_paged_lt.as_unsafe_any_origin(),
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(valid_length),
        UInt32(num_keys),
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    var test_out_size = total_length * num_q_heads * head_size
    var test_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var test_out_lt = LayoutTensor[dtype, output_layout](
        test_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )

    # --- Per-head sink weights. `num_q_heads` entries (sinks are indexed by the
    # query head). Zero-filled when sinks are off so the tensor is always valid
    # to construct; the `sink=True` parameter is what actually folds them in. ---
    var sinks_host = ctx.enqueue_create_host_buffer[dtype](num_q_heads)
    for h in range(num_q_heads):
        comptime if USE_SINK:
            # Spread per head to ~[-1, 2.5): a material shift to the softmax
            # denominator that also exercises the head-indexed sink lookup.
            sinks_host[h] = Scalar[dtype](
                Float32(h % 8) * Float32(0.5) - Float32(1.0)
            )
        else:
            sinks_host[h] = Scalar[dtype](0)
    var sinks_dev = ctx.enqueue_create_buffer[dtype](num_q_heads)
    ctx.enqueue_copy(sinks_dev, sinks_host)
    var sinks_lt = LayoutTensor[mut=False, dtype, sink_layout](
        sinks_dev.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[sink_layout].row_major(IndexList[1](num_q_heads)),
    )

    comptime if USE_SINK:
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

    # ============ Run 2: naive over the FULL key range [0, num_keys) =====
    # M4 writer-combines-all: partition 0 writes the FULL combined O, so the
    # oracle is plain `mha_gpu_naive` over the entire cache with the same mask
    # (no windowing, no per-partition weight). Reuses Run 1's paged collection.
    var ref_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var ref_out_lt = LayoutTensor[dtype, output_layout](
        ref_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    comptime if USE_SINK:
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

    # M4: the writer (partition 0) emits the FULL combined O; compare it
    # directly to the full-range naive (no per-partition weight). Track the worst
    # error PER DEPTH BLOCK (one block = one combine column = 64 bf16 elems under
    # the 128B output swizzle), not just the global max: a structural combine
    # fault (e.g. the depth-128 `o_final` single-column clobber, #36, which copied
    # depth [64,128) over [0,64)) CONFINES large error to one block, while bf16
    # rounding scatters at the ~2^-8 floor. The per-block balance assert below
    # then catches a swap whose magnitude would slip UNDER the global atol.
    comptime BLK = 64
    var max_abs_diff: Float32 = 0.0
    var argmax_idx = 0
    var blk0_max: Float32 = 0.0  # depth [0, 64)
    var blk1_max: Float32 = 0.0  # depth [64, 128), only populated at depth 128
    for i in range(total_length):
        for h in range(num_q_heads):
            for d in range(head_size):
                var flat = (i * num_q_heads + h) * head_size + d
                var a = test_out_host[flat].cast[DType.float32]()
                var expected = ref_out_host[flat].cast[DType.float32]()
                var diff = abs(a - expected)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                    argmax_idx = flat
                if d < BLK:
                    blk0_max = max(blk0_max, diff)
                else:
                    blk1_max = max(blk1_max, diff)
    print(
        "  max-abs diff(split combined O, full naive) =",
        max_abs_diff,
        " at flat idx",
        argmax_idx,
        " block0=",
        blk0_max,
        " block1=",
        blk1_max,
    )

    _ = q_dev^
    _ = kv_block_dev^
    _ = paged_lut_dev^
    _ = input_row_offsets_dev^
    _ = cache_lengths_dev^
    _ = test_out_dev^
    _ = ref_out_dev^
    _ = sinks_dev^

    comptime atol = Float32(0.04)
    if max_abs_diff > atol:
        raise Error(
            "split-K P="
            + String(P)
            + " mask="
            + String(mask_t.name())
            + " depth="
            + String(head_size)
            + " combined O diverged from full naive: max-abs diff "
            + String(max_abs_diff)
            + " > atol "
            + String(atol)
        )

    # Per-block structural ("where") check, depth 128 only (one block at depth
    # 64). A correct combine produces bf16 noise scattered across both depth
    # blocks at the ~2^-8 floor; a column clobber/swap pins large error to one
    # block. Fire only when the worse block is well above noise (> half atol) so
    # the floor never flakes, and require a >5x imbalance (the #36 bug was
    # 10-200x; chunked block0 0.866 vs block1 ~0.0039). This catches a swap whose
    # global magnitude would slip under atol -- "check WHERE the worst error sits".
    comptime if head_size > BLK:
        var worse = max(blk0_max, blk1_max)
        var better = min(blk0_max, blk1_max)
        if worse > Float32(0.5) * atol and worse > Float32(5.0) * better:
            raise Error(
                "split-K P="
                + String(P)
                + " mask="
                + String(mask_t.name())
                + " depth="
                + String(head_size)
                + " combine error concentrated in one depth block (block0 "
                + String(blk0_max)
                + " vs block1 "
                + String(blk1_max)
                + ") -- structural combine bug (column clobber/swap), not bf16"
                " noise"
            )

    print("test_mha_sm100_1q_splitk_combine: PASSED")


def main() raises:
    comptime MASK = get_defined_int["FA4_1Q_SPLITK_MASK", 0]()
    comptime DEPTH = get_defined_int["FA4_1Q_SPLITK_DEPTH", 64]()

    with DeviceContext() as ctx:
        comptime if MASK == 0:
            execute_combine_test[NullMask, DEPTH](NullMask(), ctx)
        elif MASK == 1:
            execute_combine_test[CausalMask, DEPTH](CausalMask(), ctx)
        elif MASK == 2:
            execute_combine_test[ChunkedMask[CHUNK], DEPTH](
                ChunkedMask[CHUNK](), ctx
            )
        elif MASK == 3:
            execute_combine_test[SlidingWindowCausalMask[SW_WINDOW], DEPTH](
                SlidingWindowCausalMask[SW_WINDOW](), ctx
            )
        else:
            execute_combine_test[SlidingWindowNonCausalMask[SW_WINDOW], DEPTH](
                SlidingWindowNonCausalMask[SW_WINDOW](), ctx
            )
