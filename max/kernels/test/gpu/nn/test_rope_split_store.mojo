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
from std.memory import memcpy
from std.math import ceildiv
from std.random import random_ui64, seed

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
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
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

    # --- Layouts ---
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        UNKNOWN_VALUE,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    comptime paged_lut_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime qkv_layout = Layout.row_major(UNKNOWN_VALUE, combined_dim)
    comptime output_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_dim
    )
    comptime freqs_tile_layout = row_major[max_seq_len, head_dim]()
    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)

    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )

    # --- Managed tensors ---
    var qkv = ManagedLayoutTensor[dtype, qkv_layout](
        RuntimeLayout[qkv_layout].row_major(
            IndexList[2](total_length, combined_dim)
        ),
        ctx,
    )
    random(qkv.tensor[update=False]())

    var fused_output = ManagedLayoutTensor[dtype, output_layout](
        RuntimeLayout[output_layout].row_major(
            IndexList[2](total_length, hidden_size)
        ),
        ctx,
    )

    var fused_kv_block = ManagedLayoutTensor[dtype, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape), ctx
    )
    random(fused_kv_block.tensor[update=False]())

    var unfused_kv_block = ManagedLayoutTensor[dtype, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape), ctx
    )
    var fused_kv_host = fused_kv_block.tensor()
    var unfused_kv_host = unfused_kv_block.tensor[update=False]()
    for i in range(kv_block_shape.flattened_length()):
        unfused_kv_host.ptr[i] = fused_kv_host.ptr[i]

    var row_offsets = ManagedLayoutTensor[DType.uint32, row_offsets_layout](
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
        ctx,
    )
    var row_offsets_host = row_offsets.tensor[update=False]()
    var offset = 0
    for i in range(batch_size):
        row_offsets_host[i] = UInt32(offset)
        offset += prompt_lens[i]
    row_offsets_host[batch_size] = UInt32(offset)

    var cache_lengths = ManagedLayoutTensor[DType.uint32, cache_lengths_layout](
        RuntimeLayout[cache_lengths_layout].row_major(Index(batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lens[i])

    var paged_lut = ManagedLayoutTensor[DType.uint32, paged_lut_layout](
        RuntimeLayout[paged_lut_layout].row_major(paged_lut_shape), ctx
    )
    var paged_lut_host = paged_lut.tensor[update=False]()
    var block_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lens[bs] + prompt_lens[bs]
        for block_idx in range(ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in block_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))
            block_set.add(randval)
            paged_lut_host[bs, block_idx] = UInt32(randval)

    comptime freqs_lt_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var freqs = ManagedLayoutTensor[dtype, freqs_lt_layout](
        RuntimeLayout[freqs_lt_layout].row_major(
            IndexList[2](max_seq_len, head_dim)
        ),
        ctx,
    )
    random(freqs.tensor[update=False]())

    var freqs_device = freqs.device_tensor()
    var freqs_tensor = TileTensor(freqs_device.ptr, freqs_tile_layout)

    # --- Build KV collections ---
    var cache_lengths_immut = LayoutTensor[
        DType.uint32, cache_lengths_layout, ImmutAnyOrigin
    ](
        cache_lengths.device_tensor().ptr,
        cache_lengths.device_tensor().runtime_layout,
    )
    comptime paged_lut_kv_layout = Layout.row_major[2]()
    var paged_lut_dev = paged_lut.device_tensor()
    var paged_lut_immut = LayoutTensor[
        DType.uint32, paged_lut_kv_layout, ImmutAnyOrigin
    ](
        paged_lut_dev.ptr,
        RuntimeLayout[paged_lut_kv_layout].row_major(paged_lut_shape),
    )

    # Upload both KV blocks to device once (avoids redundant copies and
    # ensures unfused_kv_block's device buffer matches its host data).
    var fused_kv_dev = fused_kv_block.device_tensor()
    var unfused_kv_dev = unfused_kv_block.device_tensor()

    var fused_kv_collection = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            fused_kv_dev.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                fused_kv_dev.runtime_layout.shape.value.canonicalize(),
                fused_kv_dev.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_immut,
        paged_lut_immut,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var unfused_kv_collection = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            unfused_kv_dev.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                unfused_kv_dev.runtime_layout.shape.value.canonicalize(),
                unfused_kv_dev.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_immut,
        paged_lut_immut,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    # =====================================================================
    # Run fused rope_split_store
    # =====================================================================
    var fused_k_cache = fused_kv_collection.get_key_cache(layer_idx)
    var fused_v_cache = fused_kv_collection.get_value_cache(layer_idx)

    var qkv_dev = qkv.device_tensor()
    var qkv_tile = TileTensor(
        qkv_dev.ptr,
        row_major((Idx(total_length), Idx[combined_dim]())),
    )
    var row_offsets_tile = TileTensor(
        row_offsets.device_tensor().ptr,
        row_major(Idx(batch_size + 1)),
    )
    var fused_out_dev = fused_output.device_tensor[update=False]()
    var fused_out_tile = TileTensor(
        fused_out_dev.ptr,
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
    var qkv_dev_ref = qkv.device_tensor[update=False]()
    var row_offsets_tile_ref = TileTensor(
        row_offsets.device_tensor[update=False]().ptr,
        row_major(Idx(batch_size + 1)),
    )

    # Store raw K and V to unfused cache
    var k_ptr = qkv_dev_ref.ptr + q_dim_val
    var v_ptr = qkv_dev_ref.ptr + q_dim_val + k_dim_val
    var k_stride0 = combined_dim
    var v_stride0 = combined_dim
    var unfused_k_cache = unfused_kv_collection.get_key_cache(layer_idx)
    var unfused_v_cache = unfused_kv_collection.get_value_cache(layer_idx)
    var row_offsets_lt = row_offsets.device_tensor[update=False]()

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
    var q_contig = ManagedLayoutTensor[dtype, q_ragged_layout](
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_dim)
        ),
        ctx,
    )
    var q_contig_host = q_contig.tensor[update=False]()
    var qkv_host = qkv.tensor()
    for t in range(total_length):
        memcpy(
            dest=q_contig_host.ptr + t * q_dim_val,
            src=qkv_host.ptr + t * combined_dim,
            count=q_dim_val,
        )

    var q_tile = TileTensor(
        q_contig.device_tensor().ptr,
        row_major((Idx(total_length), Idx[num_q_heads](), Idx[head_dim]())),
    )
    var rope_q_out = ManagedLayoutTensor[dtype, q_ragged_layout](
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_dim)
        ),
        ctx,
    )
    var rope_q_out_tile = TileTensor(
        rope_q_out.device_tensor[update=False]().ptr,
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
    var fused_q_host = fused_output.tensor()
    var unfused_q_host = rope_q_out.tensor()
    for i in range(total_length * hidden_size):
        assert_almost_equal(
            fused_q_host.ptr[i],
            unfused_q_host.ptr[i],
            atol=1e-2,
            rtol=1e-1,
            msg="Q output mismatch at flat index " + String(i),
        )
    print("Q outputs match!")

    # =====================================================================
    # Compare KV block buffers
    # =====================================================================
    print("Comparing KV block buffers...")
    var fused_kv_result = fused_kv_block.tensor()
    var unfused_kv_result = unfused_kv_block.tensor()
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
            if (
                fused_kv_result.ptr[v_offset + i]
                != unfused_kv_result.ptr[v_offset + i]
            ):
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
                fused_kv_result.ptr[block_offset + i]
                != unfused_kv_result.ptr[block_offset + i]
            ):
                k_mismatches += 1

    print("K mismatches:", k_mismatches)
    if k_mismatches > 0:
        raise Error("K cache mismatch (rope applied incorrectly)")

    print("All checks passed!")


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
