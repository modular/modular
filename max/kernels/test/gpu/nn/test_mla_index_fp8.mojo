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
"""Tests for mla_indexer_ragged_float8_paged."""

from std.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mla_index_fp8 import mla_indexer_ragged_float8_paged
from nn.attention.mha_mask import MaskName
from std.random import rand, random_ui64
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.utils.index import IndexList
from std.testing import assert_true
from std.collections import Set


def test_mla_index_fp8_paged_variable_lengths[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    mask_name: StaticString = MaskName.NULL.name,
    strict_complete: Bool = False,
](seq_lens: List[Int], cache_lens: List[Int], ctx: DeviceContext,) raises:
    """Test mla_indexer_ragged_float8_paged with variable-length sequences.

    Parameters:
        num_heads: Number of attention heads.
        depth: Head dimension.
        page_size: Page size for paged KV cache.
        top_k: Number of top indices to return.
        mask_name: Mask type name (NULL or CAUSAL).
        strict_complete: When True, additionally assert that every token
            selects its *complete* set of valid keys (exactly `num_keys`
            distinct indices covering `[0, num_keys)`).  Only valid in the
            dense regime where `top_k >= num_keys` for every token, so the
            indexer is expected to return all valid keys (no real sparsity).
            This is the strong invariant that the lenient default check
            (which permits -1 at any position) does NOT enforce; it is what
            catches the topk_gpu out_vals/out_idxs row-stride desync bug,
            where higher query rows collapsed to all -1 (see the regression
            cases in main()).

    Args:
        seq_lens: Length of each sequence (new tokens) per batch item.
        cache_lens: Length of cached tokens per batch item.
        ctx: Device context.
    """
    comptime use_causal_mask = mask_name != MaskName.NULL.name
    var batch_size = len(seq_lens)
    assert (
        len(cache_lens) == batch_size
    ), "cache_lens must have same length as seq_lens"

    # Compute totals and max lengths
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    for i in range(batch_size):
        total_seq_len += seq_lens[i]
        max_seq_len = max(max_seq_len, seq_lens[i])
        max_cache_len = max(max_cache_len, cache_lens[i])

    print(
        "test_mla_index_fp8_paged_variable_lengths with params:",
        "num_heads:",
        num_heads,
        "depth:",
        depth,
        "page_size:",
        page_size,
        "mask:",
        mask_name,
        "batch_size:",
        batch_size,
        "total_seq_len:",
        total_seq_len,
        "max_seq_len:",
        max_seq_len,
        "max_cache_len:",
        max_cache_len,
        "top_k:",
        top_k,
    )

    comptime kv_params = KVCacheStaticParams(
        num_heads=1,  # MLA uses single head for K
        head_size=depth,
        is_mla=True,
    )
    comptime num_layers = 1

    # Calculate number of pages needed (based on max sequence)
    var total_num_keys_max = max_cache_len + max_seq_len
    var pages_per_seq = (total_num_keys_max + page_size - 1) // page_size
    var num_blocks = batch_size * pages_per_seq + 10  # Extra blocks

    # Q tensor: [total_seq_len, num_heads, depth]
    var q_size = total_seq_len * num_heads * depth
    var q_ptr = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](q_size)
    rand(q_ptr.as_span())
    var q_device = ctx.enqueue_create_buffer[DType.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_ptr)

    # Q scales: [total_seq_len, num_heads]
    var qs_size = total_seq_len * num_heads
    var qs_ptr = ctx.enqueue_create_host_buffer[DType.float32](qs_size)
    rand(qs_ptr.as_span())
    var qs_device = ctx.enqueue_create_buffer[DType.float32](qs_size)
    ctx.enqueue_copy(qs_device, qs_ptr)

    # Input row offsets: [batch_size + 1] for ragged indexing (variable lengths)
    var input_row_offsets_ptr = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    input_row_offsets_ptr[0] = UInt32(0)
    for i in range(batch_size):
        input_row_offsets_ptr[i + 1] = input_row_offsets_ptr[i] + UInt32(
            seq_lens[i]
        )
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_ptr)

    # Cache lengths: [batch_size] - variable cached tokens per sequence
    var cache_lengths_ptr = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for i in range(batch_size):
        cache_lengths_ptr[i] = UInt32(cache_lens[i])
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_ptr)

    # K blocks: [num_blocks, 1, num_layers, page_size, num_heads, head_size]
    var k_shape = IndexList[6](
        num_blocks,
        1,  # MLA uses single kv
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime k_block_layout = Layout.row_major[6]()
    var k_block_runtime_layout = RuntimeLayout[k_block_layout].row_major(
        k_shape
    )
    var k_block_device = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.as_span())

    # K scale blocks
    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_dim_granularity,
    )
    var ks_block_device = ctx.enqueue_create_buffer[DType.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.as_span())

    # Page lookup tables
    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var k_lut_device = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_shape.flattened_length()
    )

    var paged_lut_set = Set[Int]()
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(pages_per_seq):
                var block_idx = Int(random_ui64(0, UInt64(num_blocks - 1)))
                while block_idx in paged_lut_set:
                    block_idx = Int(random_ui64(0, UInt64(num_blocks - 1)))
                paged_lut_set.add(block_idx)
                k_lut_host[bs * pages_per_seq + page_idx] = UInt32(block_idx)

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    comptime ks_block_layout = Layout.row_major[6]()
    var ks_block_runtime_layout = RuntimeLayout[ks_block_layout].row_major(
        ks_shape
    )
    var k_collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ](
        LayoutTensor[DType.float8_e4m3fn, k_block_layout](
            k_block_device,
            k_block_runtime_layout,
        ),
        LayoutTensor[mut=False, DType.uint32, cache_lengths_layout](
            cache_lengths_device,
            cache_lengths_runtime_layout,
        ),
        LayoutTensor[mut=False, DType.uint32, paged_lut_layout](
            k_lut_device,
            paged_lut_runtime_layout,
        ),
        UInt32(max_seq_len),  # max_seq_length (new tokens)
        UInt32(max_cache_len),  # max_cache_length (cached tokens)
        LayoutTensor[DType.float32, ks_block_layout](
            ks_block_device,
            ks_block_runtime_layout,
        ),
    )

    # Dense output: [total_seq_len, top_k]
    var total_output_size = total_seq_len * top_k

    var o_ptr = ctx.enqueue_create_host_buffer[DType.int32](total_output_size)
    var o_device = ctx.enqueue_create_buffer[DType.int32](total_output_size)

    var q_tile = TileTensor(
        q_device,
        row_major(total_seq_len, num_heads, depth),
    )

    var qs_tile = TileTensor(
        qs_device,
        row_major(total_seq_len, num_heads),
    )

    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device,
        row_major(
            batch_size + 1,
        ),
    )

    var o_tile = TileTensor(
        o_device,
        row_major(total_seq_len, top_k),
    )

    mla_indexer_ragged_float8_paged[
        DType.float8_e4m3fn,
        type_of(k_collection),
        num_heads,
        depth,
        top_k,
        mask_name,
    ](
        o_tile,
        q_tile,
        qs_tile,
        input_row_offsets_tile,
        k_collection,
        UInt32(0),  # layer_idx
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(o_ptr, o_device)
    ctx.synchronize()

    # Build a mapping from global token index to its valid key range
    # With causal mask: num_keys = cache_len + local_seq_idx + 1
    # Without mask (NULL): num_keys = cache_len + seq_len
    var token_to_num_keys = List[Int]()
    for batch_idx in range(batch_size):
        var cache_len = cache_lens[batch_idx]
        var seq_len = seq_lens[batch_idx]

        comptime if use_causal_mask:
            for local_seq_idx in range(seq_len):
                var num_keys = cache_len + local_seq_idx + 1
                token_to_num_keys.append(num_keys)
        else:
            var num_keys = cache_len + seq_len
            for _ in range(seq_len):
                token_to_num_keys.append(num_keys)

    # Verify output:
    # - For k_idx < num_keys: index must be valid [0, num_keys)
    # - For k_idx >= num_keys: index must be -1 (invalid/padded)
    var global_token_idx = 0
    for batch_idx in range(batch_size):
        for _ in range(seq_lens[batch_idx]):
            var num_keys = token_to_num_keys[global_token_idx]
            var valid_count = 0
            for k_idx in range(top_k):
                var output_idx = global_token_idx * top_k + k_idx
                var idx_int = Int(o_ptr[output_idx])

                if idx_int >= 0:
                    valid_count += 1

                if k_idx < num_keys:
                    # Valid position: index should be in range or -1 if masked
                    assert_true(
                        idx_int == -1 or (idx_int >= 0 and idx_int < num_keys),
                        "Invalid index "
                        + String(idx_int)
                        + " at k_idx "
                        + String(k_idx)
                        + " for token "
                        + String(global_token_idx)
                        + " with num_keys "
                        + String(num_keys),
                    )
                else:
                    # Beyond valid range: must be -1
                    assert_true(
                        idx_int == -1,
                        "Expected -1 at k_idx "
                        + String(k_idx)
                        + " >= num_keys "
                        + String(num_keys)
                        + " for token "
                        + String(global_token_idx)
                        + ", got "
                        + String(idx_int),
                    )

            comptime if strict_complete:
                # Dense regime (top_k >= num_keys): the indexer must select
                # ALL num_keys valid keys, never drop or collapse any.  The
                # topk_gpu row-stride desync bug corrupted this for higher
                # query rows (valid_count fell below num_keys, reaching 0 for
                # the last prefill tokens), so this exact-count check is the
                # regression guard.  The default (non-strict) check above
                # permits -1 anywhere and would NOT catch that collapse.
                assert_true(
                    valid_count == num_keys,
                    "Incomplete top-k for token "
                    + String(global_token_idx)
                    + ": selected "
                    + String(valid_count)
                    + " valid keys but expected "
                    + String(num_keys)
                    + " (dense causal set). Indicates dropped/collapsed keys"
                    + " (e.g. topk out_vals/out_idxs row-stride desync).",
                )
            global_token_idx += 1

    print("  Test passed!")

    # Cleanup
    _ = k_block_device
    _ = k_lut_device
    _ = cache_lengths_device
    _ = ks_block_device
    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = o_device


def main() raises:
    with DeviceContext() as ctx:
        print("Testing mla_indexer_ragged_float8_paged...")

        # ===== Tests with NULL mask (no causal masking) =====
        print("\n--- NULL mask tests ---")

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=16,
            mask_name=MaskName.NULL.name,
        ](
            seq_lens=[16, 32, 8, 64],
            cache_lens=[64, 128, 32, 96],
            ctx=ctx,
        )

        # Test with very short sequences (edge case: some num_keys < top_k)
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=64,
            page_size=32,
            top_k=32,
            mask_name=MaskName.NULL.name,
        ](
            seq_lens=[4, 8, 2],
            cache_lens=[4, 8, 2],
            ctx=ctx,
        )

        # ===== Tests with CAUSAL mask =====
        print("\n--- CAUSAL mask tests ---")

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=16,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[16, 32, 8, 64],
            cache_lens=[64, 128, 32, 96],
            ctx=ctx,
        )

        # Test with mixed prefill/decode (some seq_len=1, some larger)
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=16,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[1, 1, 32, 1],  # Mix of decode (1) and prefill
            cache_lens=[100, 50, 0, 200],  # Varied cache sizes
            ctx=ctx,
        )

        # Test causal mask with very short sequences
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=64,
            page_size=32,
            top_k=32,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[4, 8, 2],
            cache_lens=[4, 8, 2],
            ctx=ctx,
        )

        # ===== Regression: large top_k (2048) + long context =====
        # These cover two bugs that only appear at production scale:
        #   (A) topk_gpu stage-2 dynamic shared memory exceeded the device
        #       per-block limit once max_k = min(top_k, ctx) reached ~2000,
        #       crashing the launch with CUDA_ERROR_INVALID_VALUE.
        #   (B) fill_invalid_topk_kernel only covered the first 1024 output
        #       columns, leaving columns [1024, top_k) as garbage when
        #       top_k > 1024.
        # Each case mixes a long sequence (drives max_num_keys past the old
        # smem cliff -> exercises A) with a short sequence whose token needs
        # -1 padding spanning columns >1024 (-> exercises B).
        print("\n--- regression: top_k=2048, long context ---")

        # Decode, causal: long seq (cache 2100) + short seq (cache 50, so its
        # token needs -1 across columns [51, 2048), including the >1024 range).
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[1, 1, 1, 1],
            cache_lens=[2100, 1990, 1500, 50],
            ctx=ctx,
        )

        # Decode, NULL mask: long + short seq.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.NULL.name,
        ](
            seq_lens=[1, 1],
            cache_lens=[2100, 100],
            ctx=ctx,
        )

        # Prefill, causal: 200 new tokens over a 1900-token cache
        # (max_num_keys=2100, past the old cliff; early tokens need -1 padding).
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[200],
            cache_lens=[1900],
            ctx=ctx,
        )

        # ===== Regression: topk out_vals/out_idxs row-stride desync =====
        # GLM-5.1 / DSv3.2 prefix-cached prefill: a multi-token chunk on top of
        # a cached prefix where max_num_keys < top_k, so effective_k =
        # min(top_k, max_num_keys) < top_k.  topk_gpu indexes both of its
        # outputs by effective_k, so out_vals (effective_k stride) and out_idxs
        # MUST share that stride; aliasing out_idxs onto the top_k-strided
        # output desynced them, scattering each query row's indices r*(top_k -
        # effective_k) elements off -> higher rows collapsed to all -1.  Row 0
        # always looked fine (offset 0), so the lenient check missed it; the
        # earlier top_k=2048 cases had max_num_keys>=2048 (effective_k==top_k)
        # so they never triggered it.  These cases force max_num_keys < top_k
        # AND use strict_complete to require every token's full causal set.
        print("\n--- regression: topk stride desync (max_num_keys < top_k) ---")

        # GLM geometry: 64 heads, depth 128, top_k=2048, ~900-token cached
        # prefix + 179 fresh tokens => max_num_keys=1075 < 2048.  Last fresh
        # token must still select all 1075 causal keys.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[179],
            cache_lens=[896],
            ctx=ctx,
        )

        # Multi-batch mix of cached prefixes + multi-token chunks, all with
        # max_num_keys < top_k.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[64, 200, 32],
            cache_lens=[300, 500, 100],
            ctx=ctx,
        )

        print("\nAll tests passed!")
