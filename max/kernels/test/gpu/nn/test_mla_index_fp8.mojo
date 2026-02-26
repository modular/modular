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

from gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.mla_index_fp8 import mla_indexer_ragged_float8_paged
from nn.mha_mask import MaskName
from random import rand, random_ui64
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout.layout_tensor import LayoutTensor
from utils.index import Index, IndexList
from testing import assert_true
from collections import Set

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


fn test_mla_index_fp8_paged_variable_lengths[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    mask_name: StaticString = MaskName.NULL.name,
](seq_lens: List[Int], cache_lens: List[Int], ctx: DeviceContext,) raises:
    """Test mla_indexer_ragged_float8_paged with variable-length sequences.

    Parameters:
        num_heads: Number of attention heads.
        depth: Head dimension.
        page_size: Page size for paged KV cache.
        top_k: Number of top indices to return.
        mask_name: Mask type name (NULL or CAUSAL).

    Args:
        seq_lens: Length of each sequence (new tokens) per batch item.
        cache_lens: Length of cached tokens per batch item.
        ctx: Device context.
    """
    comptime use_causal_mask = mask_name != MaskName.NULL.name
    var batch_size = len(seq_lens)
    debug_assert(
        len(cache_lens) == batch_size,
        "cache_lens must have same length as seq_lens",
    )

    # Compute totals and max lengths
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    for i in range(batch_size):
        total_seq_len += seq_lens[i]
        max_seq_len = max(max_seq_len, seq_lens[i])
        max_cache_len = max(max_cache_len, cache_lens[i])

    # max_num_keys uses static max values to match kernel's calculation
    # (kernel uses k_cache.max_context_length() + max_prompt_length())
    var max_num_keys = max_cache_len + max_seq_len

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
        head_size=UInt(depth),
        is_mla=True,
    )
    comptime num_layers = 1

    # Calculate number of pages needed (based on max sequence)
    var total_num_keys_max = max_cache_len + max_seq_len
    var pages_per_seq = (total_num_keys_max + page_size - 1) // page_size
    var num_blocks = batch_size * pages_per_seq + 10  # Extra blocks

    # Q tensor: [total_seq_len, num_heads, depth]
    var q_size = total_seq_len * num_heads * depth
    var q_ptr = UnsafePointer[Scalar[DType.float8_e4m3fn]].alloc(q_size)
    rand(q_ptr, q_size)
    var q_device = ctx.enqueue_create_buffer[DType.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_ptr)

    # Q scales: [total_seq_len, num_heads]
    var qs_size = total_seq_len * num_heads
    var qs_ptr = UnsafePointer[Scalar[DType.float32]].alloc(qs_size)
    rand(qs_ptr, qs_size)
    var qs_device = ctx.enqueue_create_buffer[DType.float32](qs_size)
    ctx.enqueue_copy(qs_device, qs_ptr)

    # Input row offsets: [batch_size + 1] for ragged indexing (variable lengths)
    var input_row_offsets_ptr = UnsafePointer[UInt32].alloc(batch_size + 1)
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
    var cache_lengths_ptr = UnsafePointer[UInt32].alloc(batch_size)
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
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    comptime k_block_layout = Layout.row_major[6]()
    var k_block_runtime_layout = RuntimeLayout[k_block_layout].row_major(
        k_shape
    )
    var k_block_device = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.unsafe_ptr(), k_shape.flattened_length())

    # K scale blocks
    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        head_dim_granularity,
    )
    var ks_block_device = ctx.enqueue_create_buffer[DType.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.unsafe_ptr(), ks_shape.flattened_length())

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
        DType.float8_e4m3fn, kv_params, page_size, DType.float32, 128
    ](
        LayoutTensor[DType.float8_e4m3fn, k_block_layout, MutAnyOrigin](
            k_block_device.unsafe_ptr(), k_block_runtime_layout
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_device.unsafe_ptr(), cache_lengths_runtime_layout
        ),
        LayoutTensor[DType.uint32, paged_lut_layout, ImmutAnyOrigin](
            k_lut_device.unsafe_ptr(), paged_lut_runtime_layout
        ),
        UInt32(max_seq_len),  # max_seq_length (new tokens)
        UInt32(max_cache_len),  # max_cache_length (cached tokens)
        LayoutTensor[DType.float32, ks_block_layout, MutAnyOrigin](
            ks_block_device.unsafe_ptr(), ks_block_runtime_layout
        ),
    )

    # Dense output: [total_seq_len, top_k]
    var total_output_size = total_seq_len * top_k

    var o_ptr = UnsafePointer[Scalar[DType.int32]].alloc(total_output_size)
    var o_device = ctx.enqueue_create_buffer[DType.int32](total_output_size)

    comptime q_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, depth)
    var q_tensor = LayoutTensor[DType.float8_e4m3fn, q_layout](
        q_device.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(total_seq_len, num_heads, depth)
        ),
    )

    comptime qs_layout = Layout.row_major(UNKNOWN_VALUE, num_heads)
    var qs_tensor = LayoutTensor[DType.float32, qs_layout](
        qs_device.unsafe_ptr(),
        RuntimeLayout[qs_layout].row_major(Index(total_seq_len, num_heads)),
    )

    comptime input_row_offsets_layout = Layout.row_major(UNKNOWN_VALUE)
    var input_row_offsets_tensor = LayoutTensor[
        DType.uint32, input_row_offsets_layout
    ](
        input_row_offsets_device.unsafe_ptr(),
        RuntimeLayout[input_row_offsets_layout].row_major(
            Index(batch_size + 1)
        ),
    )

    comptime o_layout = Layout.row_major(UNKNOWN_VALUE, top_k)
    var o_tensor = LayoutTensor[DType.int32, o_layout](
        o_device.unsafe_ptr(),
        RuntimeLayout[o_layout].row_major(Index(total_seq_len, top_k)),
    )

    mla_indexer_ragged_float8_paged[
        DType.float8_e4m3fn,
        q_layout,
        qs_layout,
        o_layout,
        type_of(k_collection),
        num_heads,
        depth,
        top_k,
        mask_name,
    ](
        o_tensor,
        q_tensor,
        qs_tensor,
        input_row_offsets_tensor,
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

        @parameter
        if use_causal_mask:
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
            for k_idx in range(top_k):
                var output_idx = global_token_idx * top_k + k_idx
                var idx_int = Int(o_ptr[output_idx])

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

    q_ptr.free()
    qs_ptr.free()
    input_row_offsets_ptr.free()
    cache_lengths_ptr.free()
    o_ptr.free()


def main():
    with DeviceContext() as ctx:
        print("Testing mla_indexer_ragged_float8_paged...")

        # ===== Tests with NULL mask (no causal masking) =====
        print("\n--- NULL mask tests ---")

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=16,
            mask_name = MaskName.NULL.name,
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
            mask_name = MaskName.NULL.name,
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
            mask_name = MaskName.CAUSAL.name,
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
            mask_name = MaskName.CAUSAL.name,
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
            mask_name = MaskName.CAUSAL.name,
        ](
            seq_lens=[4, 8, 2],
            cache_lens=[4, 8, 2],
            ctx=ctx,
        )

        print("\nAll tests passed!")
