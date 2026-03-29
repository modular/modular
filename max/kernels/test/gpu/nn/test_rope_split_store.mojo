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

"""Correctness test for fused rope+split+store kernel.

Fills a random QKV buffer, runs rope_split_store_ragged, and compares
Q output and KV cache contents against the unfused reference path.
"""

from std.collections import Set
from std.memory import alloc, memcpy
from std.math import ceildiv
from std.random import rand, random_ui64, seed

from std.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.testing import assert_almost_equal

from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.kv_cache_ragged import kv_cache_store_ragged
from nn.rope_split_store import _rope_split_store_ragged

from std.utils import Index, IndexList


def execute_test[
    interleaved: Bool,
    num_q_heads: Int = 32,
    num_kv_heads: Int = 8,
    head_size: Int = 128,
](ctx: DeviceContext) raises:
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_size)
    )
    comptime dtype = DType.bfloat16
    comptime head_dim = Int(kv_params.head_size)
    comptime num_paged_blocks = 64
    comptime page_size = 128
    var num_layers = 1
    var layer_idx = 0

    comptime max_seq_len = 1024
    comptime hidden_size = num_q_heads * head_dim
    comptime combined_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    comptime q_dim_val = num_q_heads * head_dim
    comptime k_dim_val = num_kv_heads * head_dim

    var prompt_lens = [4, 8, 16, 1]
    var cache_lens = [10, 20, 5, 100]
    var batch_size = len(prompt_lens)

    var total_length = 0
    var max_cache_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        total_length += prompt_lens[i]
        max_cache_length = max(max_cache_length, cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

    # --- Host allocations ---
    var qkv_elems = total_length * combined_dim
    var qkv_host = alloc[Scalar[dtype]](qkv_elems)
    rand[dtype](qkv_host, qkv_elems)

    var fused_output_elems = total_length * hidden_size
    var fused_output_host = alloc[Scalar[dtype]](fused_output_elems)

    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_elems = kv_block_shape.flattened_length()
    var fused_kv_host = alloc[Scalar[dtype]](kv_block_elems)
    rand[dtype](fused_kv_host, kv_block_elems)

    var unfused_kv_host = alloc[Scalar[dtype]](kv_block_elems)
    for i in range(kv_block_elems):
        unfused_kv_host[i] = fused_kv_host[i]

    var row_offsets_host = alloc[UInt32](batch_size + 1)
    var offset = 0
    for i in range(batch_size):
        row_offsets_host[i] = UInt32(offset)
        offset += prompt_lens[i]
    row_offsets_host[batch_size] = UInt32(offset)

    var cache_lengths_host = alloc[UInt32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lens[i])

    var paged_lut_cols = ceildiv(max_full_context_length, page_size)
    var paged_lut_elems = batch_size * paged_lut_cols
    var paged_lut_host = alloc[UInt32](paged_lut_elems)
    var block_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lens[bs] + prompt_lens[bs]
        for block_idx in range(ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in block_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))
            block_set.add(randval)
            paged_lut_host[bs * paged_lut_cols + block_idx] = UInt32(randval)

    var freqs_elems = max_seq_len * head_dim
    var freqs_host = alloc[Scalar[dtype]](freqs_elems)
    rand[dtype](freqs_host, freqs_elems)

    # --- Device allocations and copies ---
    var qkv_device = ctx.enqueue_create_buffer[dtype](qkv_elems)
    ctx.enqueue_copy(qkv_device, qkv_host)

    var fused_output_device = ctx.enqueue_create_buffer[dtype](
        fused_output_elems
    )

    var fused_kv_device = ctx.enqueue_create_buffer[dtype](kv_block_elems)
    ctx.enqueue_copy(fused_kv_device, fused_kv_host)

    var unfused_kv_device = ctx.enqueue_create_buffer[dtype](kv_block_elems)
    ctx.enqueue_copy(unfused_kv_device, unfused_kv_host)

    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var paged_lut_device = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_elems
    )
    ctx.enqueue_copy(paged_lut_device, paged_lut_host)

    var freqs_device = ctx.enqueue_create_buffer[dtype](freqs_elems)
    ctx.enqueue_copy(freqs_device, freqs_host)

    ctx.synchronize()

    # --- Build KV collections (requires LayoutTensor wrappers) ---
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[
        DType.uint32, cl_layout, ImmutAnyOrigin
    ](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )

    comptime paged_lut_lt_layout = Layout.row_major[2]()
    var paged_lut_lt = LayoutTensor[
        DType.uint32, paged_lut_lt_layout, ImmutAnyOrigin
    ](
        paged_lut_device.unsafe_ptr(),
        RuntimeLayout[paged_lut_lt_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    comptime kv_lt_layout = Layout.row_major[6]()
    var fused_kv_lt = LayoutTensor[dtype, kv_lt_layout, MutAnyOrigin](
        fused_kv_device.unsafe_ptr(),
        RuntimeLayout[kv_lt_layout].row_major(kv_block_shape),
    )
    var unfused_kv_lt = LayoutTensor[dtype, kv_lt_layout, MutAnyOrigin](
        unfused_kv_device.unsafe_ptr(),
        RuntimeLayout[kv_lt_layout].row_major(kv_block_shape),
    )

    var fused_kv_collection = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        fused_kv_lt,
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var unfused_kv_collection = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        unfused_kv_lt,
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    # =====================================================================
    # Run fused rope_split_store
    # =====================================================================
    var fused_k_cache = fused_kv_collection.get_key_cache(layer_idx)
    var fused_v_cache = fused_kv_collection.get_value_cache(layer_idx)

    comptime freqs_tile_layout = row_major[max_seq_len, head_dim]()
    var freqs_tensor = TileTensor(freqs_device.unsafe_ptr(), freqs_tile_layout)

    var qkv_tile = TileTensor(
        qkv_device.unsafe_ptr(),
        row_major((Idx(total_length), Idx[combined_dim]())),
    )
    var row_offsets_tile = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )
    var fused_out_tile = TileTensor(
        fused_output_device.unsafe_ptr(),
        row_major((Idx(total_length), Idx[hidden_size]())),
    )

    _rope_split_store_ragged[target="gpu", interleaved=interleaved](
        qkv_tile,
        row_offsets_tile,
        freqs_tensor,
        fused_k_cache,
        fused_v_cache,
        fused_out_tile,
        ctx,
    )

    # =====================================================================
    # Run UNFUSED reference: store K/V to cache, then rope Q + K-in-cache
    # =====================================================================
    var qkv_ptr = qkv_device.unsafe_ptr()
    var row_offsets_tile_ref = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )

    # Store raw K and V to unfused cache
    var k_ptr = qkv_ptr + q_dim_val
    var v_ptr = qkv_ptr + q_dim_val + k_dim_val
    var k_stride0 = combined_dim
    var v_stride0 = combined_dim
    var unfused_k_cache = unfused_kv_collection.get_key_cache(layer_idx)
    var unfused_v_cache = unfused_kv_collection.get_value_cache(layer_idx)

    # Build a LayoutTensor for row_offsets (kv_cache_store_ragged requires it)
    comptime row_offsets_lt_layout = Layout(UNKNOWN_VALUE)
    var row_offsets_lt = LayoutTensor[
        DType.uint32, row_offsets_lt_layout, MutAnyOrigin
    ](
        row_offsets_device.unsafe_ptr(),
        RuntimeLayout[row_offsets_lt_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    @parameter
    @__copy_capture(k_ptr, k_stride0)
    def k_load_fn[
        width: Int, alignment: Int = 1
    ](idx: IndexList[3]) -> SIMD[dtype, width]:
        var flat = idx[0] * k_stride0 + idx[1] * head_dim + idx[2]
        return (k_ptr + flat).load[width=width]()

    @parameter
    @__copy_capture(v_ptr, v_stride0)
    def v_load_fn[
        width: Int, alignment: Int = 1
    ](idx: IndexList[3]) -> SIMD[dtype, width]:
        var flat = idx[0] * v_stride0 + idx[1] * head_dim + idx[2]
        return (v_ptr + flat).load[width=width]()

    kv_cache_store_ragged[target="gpu", input_fn=k_load_fn](
        unfused_k_cache,
        IndexList[3](total_length, num_kv_heads, head_dim),
        row_offsets_lt,
        ctx,
    )
    kv_cache_store_ragged[target="gpu", input_fn=v_load_fn](
        unfused_v_cache,
        IndexList[3](total_length, num_kv_heads, head_dim),
        row_offsets_lt,
        ctx,
    )

    # Apply rope to Q and K-in-cache.
    # Q lives in the flat QKV buffer with stride combined_dim per token.
    # Extract Q into a contiguous [total_length, num_q_heads, head_dim]
    # buffer on the host, then upload to device.
    var q_contig_elems = total_length * q_dim_val
    var q_contig_host = alloc[Scalar[dtype]](q_contig_elems)
    for t in range(total_length):
        memcpy(
            dest=q_contig_host + t * q_dim_val,
            src=qkv_host + t * combined_dim,
            count=q_dim_val,
        )

    var q_contig_device = ctx.enqueue_create_buffer[dtype](q_contig_elems)
    ctx.enqueue_copy(q_contig_device, q_contig_host)
    ctx.synchronize()

    var q_tile = TileTensor(
        q_contig_device.unsafe_ptr(),
        row_major((Idx(total_length), Idx[num_q_heads](), Idx[head_dim]())),
    )

    var rope_q_out_elems = total_length * q_dim_val
    var rope_q_out_device = ctx.enqueue_create_buffer[dtype](rope_q_out_elems)
    var rope_q_out_tile = TileTensor(
        rope_q_out_device.unsafe_ptr(),
        row_major((Idx(total_length), Idx[num_q_heads](), Idx[head_dim]())),
    )

    fused_qk_rope_ragged[
        unfused_kv_collection.CacheType,
        interleaved=interleaved,
        target="gpu",
    ](
        q_proj=q_tile,
        input_row_offsets=row_offsets_tile_ref,
        kv_collection=unfused_kv_collection,
        freqs_cis=freqs_tensor,
        position_ids=None,
        layer_idx=UInt32(layer_idx),
        output=rope_q_out_tile,
        context=ctx,
    )
    ctx.synchronize()

    # =====================================================================
    # Compare Q outputs
    # Fused kernel outputs [total_seq_len, q_dim] (rank 2) while the
    # reference outputs [total_seq_len, num_q_heads, head_dim] (rank 3).
    # Flat-pointer comparison works because both use contiguous row_major
    # layout and q_dim == num_q_heads * head_dim.
    # =====================================================================
    print("Comparing Q outputs...")
    var fused_q_result = ctx.enqueue_create_host_buffer[dtype](
        fused_output_elems
    )
    ctx.enqueue_copy(fused_q_result, fused_output_device)
    var unfused_q_result = ctx.enqueue_create_host_buffer[dtype](
        rope_q_out_elems
    )
    ctx.enqueue_copy(unfused_q_result, rope_q_out_device)
    ctx.synchronize()

    for i in range(total_length * hidden_size):
        assert_almost_equal(
            fused_q_result.unsafe_ptr()[i],
            unfused_q_result.unsafe_ptr()[i],
            atol=1e-2,
            rtol=1e-1,
            msg="Q output mismatch at flat index " + String(i),
        )
    print("Q outputs match!")

    # =====================================================================
    # Compare KV block buffers
    # =====================================================================
    print("Comparing KV block buffers...")
    var fused_kv_result = ctx.enqueue_create_host_buffer[dtype](kv_block_elems)
    ctx.enqueue_copy(fused_kv_result, fused_kv_device)
    var unfused_kv_result = ctx.enqueue_create_host_buffer[dtype](
        kv_block_elems
    )
    ctx.enqueue_copy(unfused_kv_result, unfused_kv_device)
    ctx.synchronize()

    var fused_kv_ptr = fused_kv_result.unsafe_ptr()
    var unfused_kv_ptr = unfused_kv_result.unsafe_ptr()
    var nl = num_layers
    var ps = page_size
    var nh = Int(kv_params.num_heads)
    var hd = Int(kv_params.head_size)
    var inner = nl * ps * nh * hd
    var v_mismatches = 0
    for block in range(num_paged_blocks):
        var block_offset = block * 2 * inner
        var v_offset = block_offset + inner
        for i in range(inner):
            if fused_kv_ptr[v_offset + i] != unfused_kv_ptr[v_offset + i]:
                v_mismatches += 1

    print("V mismatches:", v_mismatches)
    if v_mismatches > 0:
        raise Error("V cache should be identical (no rope applied)")

    # Compare K cache (first half of each block). K has rope applied by
    # both paths in the same order, so exact comparison is valid.
    var k_mismatches = 0
    for block in range(num_paged_blocks):
        var block_offset = block * 2 * inner
        for i in range(inner):
            if (
                fused_kv_ptr[block_offset + i]
                != unfused_kv_ptr[block_offset + i]
            ):
                k_mismatches += 1

    print("K mismatches:", k_mismatches)
    if k_mismatches > 0:
        raise Error("K cache mismatch (rope applied incorrectly)")

    print("All checks passed!")

    # --- Free host allocations ---
    qkv_host.free()
    fused_output_host.free()
    fused_kv_host.free()
    unfused_kv_host.free()
    row_offsets_host.free()
    cache_lengths_host.free()
    paged_lut_host.free()
    freqs_host.free()
    q_contig_host.free()


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        # Default: GQA (32 Q heads, 8 KV heads, head_size=128)
        print("=== GQA interleaved=True ===")
        execute_test[interleaved=True](ctx)
        print("\n=== GQA interleaved=False ===")
        execute_test[interleaved=False](ctx)

        # MHA: num_kv_heads == num_q_heads
        print("\n=== MHA (32/32) interleaved=True ===")
        execute_test[interleaved=True, num_q_heads=32, num_kv_heads=32](ctx)

        # MQA: num_kv_heads == 1
        print("\n=== MQA (32/1) interleaved=True ===")
        execute_test[interleaved=True, num_q_heads=32, num_kv_heads=1](ctx)

        # Smaller head_size (64)
        print("\n=== head_size=64 interleaved=True ===")
        execute_test[interleaved=True, head_size=64](ctx)
        print("\n=== head_size=64 interleaved=False ===")
        execute_test[interleaved=False, head_size=64](ctx)
