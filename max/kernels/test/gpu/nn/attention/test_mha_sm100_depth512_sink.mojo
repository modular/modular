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

"""Attention-sink coverage for the SM100 depth=256/512 pair-CTA prefill.

Target hardware family: NVIDIA SM100 (B200).

Each cell runs paged `flash_attention[sink=True]` and compares it with
`mha_gpu_naive[sink=True]` over the same KV blocks and sink weights. Prompts
with `max_prompt_len * group > 32` route to the depth512 kernel; shorter ones
and decode run FA4. Shapes:

* depth=256 (Q/K/V padded from 192), 64 Q / 8 KV heads,
  `SlidingWindowCausalMask[128]`, in bf16 and fp8.
* depth=512, 32 Q / 4 KV heads, `CausalMask`, in bf16 and fp8.
* depth=128, 64 Q / 8 KV heads, `SlidingWindowNonCausalMask[1024]`, bf16.
  This never reaches depth512 and covers the FA4 2Q sink route.

fp8 cells quantize Q, K, V and the sinks to e4m3, then run the reference on
a bf16 copy of the quantized values.
"""

from std.collections import Set
from std.math import align_up, ceildiv, inf, isfinite, rsqrt
from std.random import rand, random_ui64, seed

from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    SlidingWindowCausalMask,
    SlidingWindowNonCausalMask,
)

from std.utils import IndexList


# Mirrors `padded_lut_cols` in `kv_cache_test_utils.mojo`, which is in another
# bazel package.
comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    return align_up(cols, 8) + _LUT_TAIL_PAD


# ===-----------------------------------------------------------------------===#
# Core test
# ===-----------------------------------------------------------------------===#


def execute_sink_test[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    mask_t: MHAMask,
    dtype: DType = DType.bfloat16,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    use_sink: Bool,
    cell_name: String,
    mask: mask_t,
    scale: Float32,
    ctx: DeviceContext,
    num_partitions: Optional[Int] = None,
) raises -> Float32:
    """Runs paged `flash_attention` and `mha_gpu_naive` on the same inputs.

    Parameters:
        num_q_heads: Number of query heads.
        kv_params: KV cache head count and head size.
        mask_t: Attention mask type.
        dtype: Q/K/V and sink dtype of the kernel under test. The reference
            always runs in bfloat16.

    Args:
        valid_lengths: Per-sequence prompt lengths.
        cache_lengths: Per-sequence cached token counts.
        use_sink: Whether to apply per-head sink weights.
        cell_name: Label printed with the result.
        mask: Attention mask.
        scale: Softmax scale. Passed explicitly because production uses
            `192 ** -0.5` with Q/K padded to 256.
        ctx: Device context.
        num_partitions: Optional split-K partition override.

    Returns:
        The max-abs output difference, or infinity if any output is not
        finite.
    """
    comptime ref_dtype = DType.bfloat16
    comptime page_size = 128
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime head_size = kv_params.head_size
    comptime group = num_q_heads // kv_params.num_heads

    var batch_size = len(valid_lengths)

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    print(
        "  cell=",
        cell_name,
        " dtype=",
        dtype,
        " sink=",
        use_sink,
        " valid=",
        valid_lengths[0],
        " cache=",
        cache_lengths[0],
        " max_prompt_len=",
        max_prompt_length,
        " num_keys=",
        max_full_context_length,
        sep="",
    )

    # --- Layouts ---
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

    # --- Host metadata: row offsets + cache lengths ---
    var input_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets[batch_size] = running_offset

    var input_row_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(input_row_offsets_dev, input_row_offsets)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)

    # --- Q (ragged: [total_length, num_q_heads, head_size]) ---
    var q_size = total_length * num_q_heads * head_size
    var q_ref_host = ctx.enqueue_create_host_buffer[ref_dtype](q_size)
    var q_ref_host_tt = LayoutTensor[ref_dtype, q_ragged_layout](
        q_ref_host.unsafe_ptr(),
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    # Zero-mean Q keeps the scores near zero, so the sink stays significant
    # even against a few hundred keys.
    random(q_ref_host_tt, min=-1, max=1)
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    for i in range(q_size):
        q_host[i] = q_ref_host[i].cast[dtype]()
        q_ref_host[i] = q_host[i].cast[ref_dtype]()
    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    var q_ref_dev = ctx.enqueue_create_buffer[ref_dtype](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(q_ref_dev, q_ref_host)

    # --- Paged KV blocks ---
    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size + 4
    )
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
    var kv_block_ref_host = ctx.enqueue_create_host_buffer[ref_dtype](
        kv_block_size
    )
    var kv_block_ref_host_tt = LayoutTensor[ref_dtype, kv_block_6d_layout](
        kv_block_ref_host.unsafe_ptr(),
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
    )
    random(kv_block_ref_host_tt)
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](kv_block_size)
    for i in range(kv_block_size):
        kv_block_host[i] = kv_block_ref_host[i].cast[dtype]()
        kv_block_ref_host[i] = kv_block_host[i].cast[ref_dtype]()
    var kv_block_dev = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var kv_block_ref_dev = ctx.enqueue_create_buffer[ref_dtype](kv_block_size)
    ctx.enqueue_copy(kv_block_dev, kv_block_host)
    ctx.enqueue_copy(kv_block_ref_dev, kv_block_ref_host)

    # --- Paged lookup table (unique random physical blocks per (bs, blk)) ---
    var lut_cols = padded_lut_cols(ceildiv(max_full_context_length, page_size))
    var paged_lut_shape = IndexList[2](batch_size, lut_cols)
    var paged_lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * lut_cols
    )
    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            paged_lut_set.add(randval)
            paged_lut_host[bs * lut_cols + block_idx] = UInt32(randval)
    var paged_lut_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size * lut_cols
    )
    ctx.enqueue_copy(paged_lut_dev, paged_lut_host)

    # --- Per-head sink weights ---
    # Uniform in [-2, 6), wide enough that a dropped or double-counted sink
    # moves the output well past tolerance.
    var sinks_host = ctx.enqueue_create_host_buffer[dtype](num_q_heads)
    var sinks_ref_host = ctx.enqueue_create_host_buffer[ref_dtype](num_q_heads)
    if use_sink:
        rand(sinks_ref_host.as_span())
        for h in range(num_q_heads):
            sinks_host[h] = (
                sinks_ref_host[h].cast[.float32]() * Float32(8.0) - Float32(2.0)
            ).cast[dtype]()
            sinks_ref_host[h] = sinks_host[h].cast[ref_dtype]()
    else:
        sinks_host.as_span().fill(Scalar[dtype](0))
        sinks_ref_host.as_span().fill(Scalar[ref_dtype](0))
    var sinks_dev = ctx.enqueue_create_buffer[dtype](num_q_heads)
    var sinks_ref_dev = ctx.enqueue_create_buffer[ref_dtype](num_q_heads)
    ctx.enqueue_copy(sinks_dev, sinks_host)
    ctx.enqueue_copy(sinks_ref_dev, sinks_ref_host)

    var input_row_offsets_lt = LayoutTensor[
        mut=False, .uint32, row_offsets_layout
    ](
        input_row_offsets_dev,
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
    )
    var cache_lengths_lt = LayoutTensor[
        mut=False, .uint32, cache_lengths_layout
    ](
        cache_lengths_dev,
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )
    var paged_lut_lt = LayoutTensor[mut=False, .uint32, paged_lut_layout](
        paged_lut_dev,
        RuntimeLayout[paged_lut_layout].row_major(paged_lut_shape),
    )
    var q_runtime_layout = RuntimeLayout[q_ragged_layout].row_major(
        IndexList[3](total_length, num_q_heads, head_size)
    )
    var q_lt = LayoutTensor[mut=False, dtype, q_ragged_layout](
        q_dev, q_runtime_layout
    )
    var q_ref_lt = LayoutTensor[mut=False, ref_dtype, q_ragged_layout](
        q_ref_dev, q_runtime_layout
    )
    var sink_runtime_layout = RuntimeLayout[sink_layout].row_major(
        IndexList[1](num_q_heads)
    )
    var sinks_lt = LayoutTensor[mut=False, dtype, sink_layout](
        sinks_dev.unsafe_ptr().as_unsafe_any_origin(), sink_runtime_layout
    )
    var sinks_ref_lt = LayoutTensor[mut=False, ref_dtype, sink_layout](
        sinks_ref_dev.unsafe_ptr().as_unsafe_any_origin(), sink_runtime_layout
    )

    var kv_runtime_layout = RuntimeLayout[kv_block_6d_layout].row_major(
        kv_block_paged_shape
    )
    var kv_block_lt = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_dev, kv_runtime_layout
    )
    var kv_block_ref_lt = LayoutTensor[ref_dtype, kv_block_6d_layout](
        kv_block_ref_dev, kv_runtime_layout
    )
    # The K and V views are disjoint halves of one `blocks` buffer, which the
    # origin exclusivity check would reject, so opt out with an unsafe origin.
    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        kv_block_lt.as_unsafe_any_origin(),
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )
    var kv_ref_collection = PagedKVCacheCollection[
        ref_dtype, kv_params, page_size
    ](
        kv_block_ref_lt.as_unsafe_any_origin(),
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    # --- Kernel under test ---
    var out_size = total_length * num_q_heads * head_size
    var out_runtime_layout = RuntimeLayout[output_layout].row_major(
        IndexList[3](total_length, num_q_heads, head_size)
    )
    var test_out_dev = ctx.enqueue_create_buffer[ref_dtype](out_size)
    var test_out_lt = LayoutTensor[ref_dtype, output_layout](
        test_out_dev.unsafe_ptr(), out_runtime_layout
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)
    if use_sink:
        flash_attention[ragged=True, sink=True](
            test_out_lt,
            q_lt,
            k_cache,
            v_cache,
            mask,
            input_row_offsets_lt,
            scale,
            ctx,
            num_partitions=num_partitions,
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
            num_partitions=num_partitions,
        )

    # --- Reference ---
    var ref_out_dev = ctx.enqueue_create_buffer[ref_dtype](out_size)
    var ref_out_lt = LayoutTensor[ref_dtype, output_layout](
        ref_out_dev.unsafe_ptr(), out_runtime_layout
    )
    var k_ref_cache = kv_ref_collection.get_key_cache(layer_idx)
    var v_ref_cache = kv_ref_collection.get_value_cache(layer_idx)
    if use_sink:
        mha_gpu_naive[ragged=True, sink=True](
            q_ref_lt,
            k_ref_cache,
            v_ref_cache,
            mask,
            ref_out_lt,
            input_row_offsets_lt,
            scale,
            batch_size,
            max_prompt_length,
            max_full_context_length,
            num_q_heads,
            head_size,
            group,
            ctx,
            sinks_ref_lt,
        )
    else:
        mha_gpu_naive[ragged=True](
            q_ref_lt,
            k_ref_cache,
            v_ref_cache,
            mask,
            ref_out_lt,
            input_row_offsets_lt,
            scale,
            batch_size,
            max_prompt_length,
            max_full_context_length,
            num_q_heads,
            head_size,
            group,
            ctx,
        )

    var test_out_host = ctx.enqueue_create_host_buffer[ref_dtype](out_size)
    var ref_out_host = ctx.enqueue_create_host_buffer[ref_dtype](out_size)
    ctx.enqueue_copy(test_out_host, test_out_dev)
    ctx.enqueue_copy(ref_out_host, ref_out_dev)
    ctx.synchronize()

    var max_abs_diff: Float32 = 0.0
    var argmax_idx = 0
    for i in range(out_size):
        var a = test_out_host[i].cast[.float32]()
        var b = ref_out_host[i].cast[.float32]()
        var d = abs(a - b)
        # `d > max_abs_diff` is false for NaN.
        if not isfinite(d):
            max_abs_diff = inf[DType.float32]()
            argmax_idx = i
            break
        if d > max_abs_diff:
            max_abs_diff = d
            argmax_idx = i
    print(
        "    max-abs diff vs naive =", max_abs_diff, " at flat idx", argmax_idx
    )

    _ = q_dev^
    _ = q_ref_dev^
    _ = kv_block_dev^
    _ = kv_block_ref_dev^
    _ = paged_lut_dev^
    _ = input_row_offsets_dev^
    _ = cache_lengths_dev^
    _ = sinks_dev^
    _ = sinks_ref_dev^
    _ = test_out_dev^
    _ = ref_out_dev^

    return max_abs_diff


# ===-----------------------------------------------------------------------===#
# Entry point
# ===-----------------------------------------------------------------------===#


def main() raises:
    comptime bf16_atol = Float32(1e-2)
    comptime fp8_atol = Float32(4e-2)
    comptime fp8 = DType.float8_e4m3fn
    var n_fail = 0
    var result_names = List[String]()
    var result_diffs = List[Float32]()
    var result_atols = List[Float32]()

    @inline(.always)
    def check(
        name: String, diff: Float32, atol: Float32
    ) {mut n_fail, mut result_names, mut result_diffs, mut result_atols}:
        result_names.append(name)
        result_diffs.append(diff)
        result_atols.append(atol)
        if diff > atol:
            n_fail += 1

    with DeviceContext() as ctx:
        seed(0x51A6)

        print("test_mha_sm100_depth512_sink: d256<-192, 64q/8kv, window=128")
        comptime swa_num_q_heads = 64
        comptime swa_kv_params = KVCacheStaticParams(num_heads=8, head_size=256)
        var swa_mask = SlidingWindowCausalMask[128]()
        var swa_scale = rsqrt(Float32(192))

        # q_len=5 is the smallest prompt that routes to depth512 at group=8;
        # q_len=1 stays on FA4.
        for q_len in [1, 5, 8, 64, 2048]:
            var name = "swa_" + String(q_len)
            check(
                name,
                execute_sink_test[swa_num_q_heads, swa_kv_params](
                    [q_len], [0], True, name, swa_mask, swa_scale, ctx
                ),
                bf16_atol,
            )

        # Speculative-decode verify: 8 tokens against a 200-token cache.
        check(
            "verify_8",
            execute_sink_test[swa_num_q_heads, swa_kv_params](
                [8], [200], True, "verify_8", swa_mask, swa_scale, ctx
            ),
            bf16_atol,
        )

        # The batch's max prompt length picks the route, so the decode row
        # runs in depth512 alongside the 64-token prefill row.
        check(
            "mixed_batch_decode_prefill",
            execute_sink_test[swa_num_q_heads, swa_kv_params](
                [1, 64],
                [8192, 0],
                True,
                "mixed_batch_decode_prefill",
                swa_mask,
                swa_scale,
                ctx,
            ),
            bf16_atol,
        )

        # FA4 split-K decode, which must fold the sink into one partition
        # only. Causal rather than SWA-128, which would leave 3 of the 4
        # partitions with no visible keys.
        check(
            "decode_splitk_sink_p4",
            execute_sink_test[swa_num_q_heads, swa_kv_params](
                [1],
                [4096],
                True,
                "decode_splitk_sink_p4",
                CausalMask(),
                swa_scale,
                ctx,
                num_partitions=4,
            ),
            bf16_atol,
        )

        # The 128-key window spans one page, so these cross a window and a
        # page boundary together.
        for q_len in [127, 128, 129, 1023, 1024, 1025]:
            var name = "boundary_" + String(q_len)
            check(
                name,
                execute_sink_test[swa_num_q_heads, swa_kv_params](
                    [q_len], [0], True, name, swa_mask, swa_scale, ctx
                ),
                bf16_atol,
            )

        for q_len in [1, 64]:
            var name = "swa_" + String(q_len) + "_sink_off"
            check(
                name,
                execute_sink_test[swa_num_q_heads, swa_kv_params](
                    [q_len], [0], False, name, swa_mask, swa_scale, ctx
                ),
                bf16_atol,
            )

        # (q_len, cache_len, use_sink) cells for the fp8 and depth=512 shapes.
        var cell_q_lens = [64, 1024, 8, 64]
        var cell_cache_lens = [0, 0, 200, 0]
        var cell_sinks = [True, True, True, False]

        print("test_mha_sm100_depth512_sink: d256<-192 fp8, 64q/8kv")
        for i in range(len(cell_q_lens)):
            var name = (
                "fp8_swa_c"
                + String(cell_cache_lens[i])
                + "_q"
                + String(cell_q_lens[i])
                + ("" if cell_sinks[i] else "_sink_off")
            )
            check(
                name,
                execute_sink_test[swa_num_q_heads, swa_kv_params, dtype=fp8](
                    [cell_q_lens[i]],
                    [cell_cache_lens[i]],
                    cell_sinks[i],
                    name,
                    swa_mask,
                    swa_scale,
                    ctx,
                ),
                fp8_atol,
            )

        # depth=512 is the split_o configuration, where a thread pair shares
        # each row's max and sum.
        print("test_mha_sm100_depth512_sink: d512, 32q/4kv, causal")
        comptime d512_num_q_heads = 32
        comptime d512_kv_params = KVCacheStaticParams(
            num_heads=4, head_size=512
        )
        var d512_scale = rsqrt(Float32(512))
        for i in range(len(cell_q_lens)):
            var suffix = (
                "_c"
                + String(cell_cache_lens[i])
                + "_q"
                + String(cell_q_lens[i])
                + ("" if cell_sinks[i] else "_sink_off")
            )
            check(
                "d512" + suffix,
                execute_sink_test[d512_num_q_heads, d512_kv_params](
                    [cell_q_lens[i]],
                    [cell_cache_lens[i]],
                    cell_sinks[i],
                    "d512" + suffix,
                    CausalMask(),
                    d512_scale,
                    ctx,
                ),
                bf16_atol,
            )
            check(
                "fp8_d512" + suffix,
                execute_sink_test[d512_num_q_heads, d512_kv_params, dtype=fp8](
                    [cell_q_lens[i]],
                    [cell_cache_lens[i]],
                    cell_sinks[i],
                    "fp8_d512" + suffix,
                    CausalMask(),
                    d512_scale,
                    ctx,
                ),
                fp8_atol,
            )

        print(
            "test_mha_sm100_depth512_sink: FA4 2Q route (d128, 64q/8kv,"
            " non-causal window=1024)"
        )
        comptime drafter_num_q_heads = 64
        comptime drafter_kv_params = KVCacheStaticParams(
            num_heads=8, head_size=128
        )
        var drafter_mask = SlidingWindowNonCausalMask[1024]()
        var drafter_scale = rsqrt(Float32(128))

        # Contexts empty, shorter than, and at the window, and a query as
        # long as the window.
        var drafter_cache_lens = [0, 500, 1024, 0]
        var drafter_q_lens = [8, 8, 8, 1024]
        for i in range(len(drafter_cache_lens)):
            var cache_len = drafter_cache_lens[i]
            var q_len = drafter_q_lens[i]
            var name = "drafter_c" + String(cache_len) + "_q" + String(q_len)
            check(
                name,
                execute_sink_test[drafter_num_q_heads, drafter_kv_params](
                    [q_len],
                    [cache_len],
                    True,
                    name,
                    drafter_mask,
                    drafter_scale,
                    ctx,
                ),
                bf16_atol,
            )

        print("=== max-abs diff vs naive ===")
        for i in range(len(result_names)):
            var diff = result_diffs[i]
            var atol = result_atols[i]
            print(
                "  ",
                result_names[i],
                ": ",
                diff,
                " (atol=",
                atol,
                ") -> ",
                "PASS" if diff <= atol else "FAIL",
                sep="",
            )

        if n_fail > 0:
            raise Error(String(n_fail) + " cell(s) exceeded tolerance vs naive")

        print("test_mha_sm100_depth512_sink: ALL PASSED")
