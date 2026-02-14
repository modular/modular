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

"""Test MLA decode with paged KV cache and variable sequence lengths.

This test exercises the splitK ragged layout path that was fixed:
  Bug A: o_accum_split layout mismatch (ragged vs padded)
  Bug B: LSE stride mismatch (per-batch seq_len vs q_max_seq_len)
  Bug C: Combine kernel output indexing (padded vs ragged)

The key stress scenarios:
  - Variable KV cache lengths per batch (30 to 32768 tokens), causing
    different num_partitions per batch and many empty splits
  - Extreme disparity in cache lengths within the same batch
"""

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from collections import Optional, OptionalReg
from math import ceildiv, exp
from random import randn, seed
from sys import argv, has_nvidia_gpu_accelerator, size_of

from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import alloc
from nn.mha import mha_gpu_naive
from nn.mha_mask import NullMask
from nn.mha_operand import KVCacheMHAOperand
from nn.mha_score_mod import IdentityScoreMod
from nn.mla import flare_mla_decoding
from testing import assert_almost_equal, assert_true
from gpu.host.info import B200
from utils.index import Index, IndexList


# ===-----------------------------------------------------------------------===#
# Test parameters matching DeepSeek-V2-Lite
# ===-----------------------------------------------------------------------===#

comptime DEPTH = 576  # Q/K head dimension
comptime V_DEPTH = 512  # Output head dimension (depth - 64)
comptime PAGE_SIZE = 128  # Standard page size
comptime NUM_LAYERS = 1  # Single layer for testing
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# ===-----------------------------------------------------------------------===#
# Core test: paged KV cache with variable lengths + numerical verification
# ===-----------------------------------------------------------------------===#


fn run_test_paged_variable[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    var batch_size = len(cache_lengths)
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " q_type:",
        q_type,
        " kv_type:",
        kv_type,
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    # All seq_lens = 1 (standard decode)
    comptime q_max_seq_len = 1
    var total_q_tokens = batch_size  # Each batch has 1 query token

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i], PAGE_SIZE)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache on host
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )

    var blocks_host = UnsafePointer[Scalar[kv_type]].alloc(block_elems)

    # Generate random data in bf16 then cast to kv_type.  This avoids
    # issues with randn producing poorly-distributed values for float8
    # types.  For bf16 kv_type the cast is a no-op.
    # Use std=0.5 to keep QK dot products moderate and softmax
    # numerically stable across all cache lengths.
    var blocks_bf16 = UnsafePointer[Scalar[q_type]].alloc(block_elems)
    randn[q_type](blocks_bf16, block_elems, mean=0.0, standard_deviation=0.5)
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()
    blocks_bf16.free()

    # Zero out the unused tail slots in each page so that if the kernel
    # accidentally reads beyond a batch's cache_len, the attention weights
    # on those tokens are negligible (exp of Q*0 is small relative to real data).
    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var _tok_stride = Int(kv_params.num_heads) * Int(kv_params.head_size)

    var cache_lengths_host = UnsafePointer[UInt32].alloc(batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = ceildiv(max_cache_len, PAGE_SIZE)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = UnsafePointer[UInt32].alloc(lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i], PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in each page (tokens beyond cache_len).
    var cur_page = 0
    for bi in range(batch_size):
        var cl = cache_lengths[bi]
        var num_pages_i = ceildiv(cl, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = cl - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            # Zero out tokens [valid_toks, PAGE_SIZE) in this page
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # -----------------------------------------------------------------------
    # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, DEPTH])
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = UnsafePointer[Scalar[q_type]].alloc(q_size)
    randn[q_type](q_host, q_size, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets (batch_size + 1 elements)
    # -----------------------------------------------------------------------
    var row_offsets_host = UnsafePointer[UInt32].alloc(batch_size + 1)
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # -----------------------------------------------------------------------
    # Step 4: Output tensor
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = UnsafePointer[Scalar[q_type]].alloc(out_size)

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Step 6: Build LayoutTensors and PagedKVCacheCollection on device
    # -----------------------------------------------------------------------

    # First create mutable LayoutTensors from device buffers
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[DType.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )

    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[DType.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    # Build PagedKVCacheCollection using the pattern from existing tests:
    # wrap .ptr into LayoutTensors with appropriate origins
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, cl_layout, ImmutAnyOrigin](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, lt_layout_2d, ImmutAnyOrigin](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    comptime q_layout_3d = Layout.row_major(
        Index(UNKNOWN_VALUE, num_heads, DEPTH)
    )
    var q_lt = LayoutTensor[q_type, q_layout_3d](
        q_device.unsafe_ptr(),
        RuntimeLayout[q_layout_3d].row_major(
            Index(total_q_tokens, num_heads, DEPTH)
        ),
    )

    comptime out_layout_3d = Layout.row_major(
        Index(UNKNOWN_VALUE, num_heads, V_DEPTH)
    )
    var out_lt = LayoutTensor[q_type, out_layout_3d](
        out_device.unsafe_ptr(),
        RuntimeLayout[out_layout_3d].row_major(
            Index(total_q_tokens, num_heads, V_DEPTH)
        ),
    )

    comptime ro_layout = Layout.row_major(UNKNOWN_VALUE)
    var row_offsets_lt = LayoutTensor[DType.uint32, ro_layout](
        row_offsets_device.unsafe_ptr(),
        RuntimeLayout[ro_layout].row_major(Index(batch_size + 1)),
    )

    # -----------------------------------------------------------------------
    # Step 7: Call the kernel
    # -----------------------------------------------------------------------
    print("  Launching MLA decode kernel...")

    flare_mla_decoding[rank=3, ragged=True](
        out_lt,
        q_lt,
        kv_cache,
        NullMask(),
        IdentityScoreMod(),
        row_offsets_lt,
        scale,
        ctx,
        q_max_seq_len,
    )

    ctx.synchronize()
    print("  Kernel completed successfully (no crash).")

    # -----------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive reference
    # -----------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch and comparing...")

    # Element-wise comparison: same tolerances as test_mla_decode_kv_fp8.
    var rtol = 5e-2  # 0.05
    var atol = 3e-1  # 0.3
    var total_checked = 0
    var max_abs_err = Float64(0)

    # Run mha_gpu_naive per batch with correct num_keys to avoid
    # zero-padding effects from shorter caches.
    for b in range(batch_size):
        var cache_len = cache_lengths[b]

        # Extract contiguous K for this batch from paged blocks
        var k_b_size = cache_len * KV_NUM_HEADS * DEPTH
        var k_b_host = UnsafePointer[Scalar[kv_type]].alloc(k_b_size)

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi], PAGE_SIZE)

        for tok in range(cache_len):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            var src_offset = (
                physical_page
                * kv_dim2
                * NUM_LAYERS
                * PAGE_SIZE
                * Int(kv_params.num_heads)
                * Int(kv_params.head_size)
                + tok_in_page
                * Int(kv_params.num_heads)
                * Int(kv_params.head_size)
            )
            var dst_offset = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                k_b_host[dst_offset + d] = blocks_host[src_offset + d]

        # Q for this batch: [1, 1, num_heads, depth]
        var q_b_size = 1 * num_heads * DEPTH
        var q_b_host = UnsafePointer[Scalar[q_type]].alloc(q_b_size)
        for i in range(q_b_size):
            q_b_host[i] = q_host[b * num_heads * DEPTH + i]

        # Reference output: [1, 1, num_heads, depth] (full depth)
        var ref_b_size = 1 * num_heads * DEPTH
        var ref_b_host = UnsafePointer[Scalar[q_type]].alloc(ref_b_size)

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
        ctx.synchronize()

        # Build 4D LayoutTensors: batch_size=1
        comptime layout_4d = Layout.row_major[4]()
        var q_b_lt = LayoutTensor[q_type, layout_4d](
            q_b_device.unsafe_ptr(),
            RuntimeLayout[layout_4d].row_major(Index(1, 1, num_heads, DEPTH)),
        )
        var k_b_lt = LayoutTensor[kv_type, layout_4d](
            k_b_device.unsafe_ptr(),
            RuntimeLayout[layout_4d].row_major(
                Index(1, cache_len, KV_NUM_HEADS, DEPTH)
            ),
        )
        var ref_b_lt = LayoutTensor[q_type, layout_4d](
            ref_b_device.unsafe_ptr(),
            RuntimeLayout[layout_4d].row_major(Index(1, 1, num_heads, DEPTH)),
        )

        # Run mha_gpu_naive: batch_size=1, num_keys=cache_len
        # K passed as both K and V (MLA: V = K[:,:,:512])
        mha_gpu_naive(
            q_b_lt,
            k_b_lt,
            k_b_lt,
            NullMask(),
            ref_b_lt,
            scale,
            1,  # batch_size
            1,  # seq_len
            cache_len,  # num_keys (exact per batch)
            num_heads,
            DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare first V_DEPTH=512 dims (depth-64) per head
        # ref layout: [1, 1, num_heads, depth]
        # actual layout: [total_tokens, num_heads, V_DEPTH]
        var out_offset = b * num_heads * V_DEPTH
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var expect = ref_b_host.load(d + DEPTH * h).cast[
                    DType.float64
                ]()
                var actual = out_host.load(out_offset + V_DEPTH * h + d).cast[
                    DType.float64
                ]()
                var abs_err = abs(actual - expect)
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if abs_err > 1e-1:
                    print(b, h, d, actual, expect)
                assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

        total_checked += num_heads * V_DEPTH

        # Cleanup per-batch buffers
        k_b_host.free()
        q_b_host.free()
        ref_b_host.free()
        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    # Cleanup
    q_host.free()
    out_host.free()
    blocks_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    row_offsets_host.free()

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


# ===-----------------------------------------------------------------------===#
# Benchmark: paged KV cache with variable lengths (no numerical verification)
# ===-----------------------------------------------------------------------===#


fn run_bench_paged_variable[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    var batch_size = len(cache_lengths)
    print(
        "bench:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " q_type:",
        q_type,
        " kv_type:",
        kv_type,
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    comptime q_max_seq_len = 1
    var total_q_tokens = batch_size

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i], PAGE_SIZE)

    comptime scale = Float32(0.125)

    # Step 1: Paged KV cache on host
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )

    var blocks_host = UnsafePointer[Scalar[kv_type]].alloc(block_elems)
    var blocks_bf16 = UnsafePointer[Scalar[q_type]].alloc(block_elems)
    randn[q_type](blocks_bf16, block_elems, mean=0.0, standard_deviation=0.5)
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()
    blocks_bf16.free()

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var _tok_stride = Int(kv_params.num_heads) * Int(kv_params.head_size)

    var cache_lengths_host = UnsafePointer[UInt32].alloc(batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = ceildiv(max_cache_len, PAGE_SIZE)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = UnsafePointer[UInt32].alloc(lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i], PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in each page
    var cur_page = 0
    for bi in range(batch_size):
        var cl = cache_lengths[bi]
        var num_pages_i = ceildiv(cl, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = cl - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # Step 2: Q tensor
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = UnsafePointer[Scalar[q_type]].alloc(q_size)
    randn[q_type](q_host, q_size, mean=0.0, standard_deviation=0.5)

    # Step 3: input_row_offsets
    var row_offsets_host = UnsafePointer[UInt32].alloc(batch_size + 1)
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # Step 4: Output tensor
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = UnsafePointer[Scalar[q_type]].alloc(out_size)

    # Step 5: Copy to device
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    ctx.synchronize()

    # Step 6: Build LayoutTensors and PagedKVCacheCollection
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[DType.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )

    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[DType.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, cl_layout, ImmutAnyOrigin](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, lt_layout_2d, ImmutAnyOrigin](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    comptime q_layout_3d = Layout.row_major(
        Index(UNKNOWN_VALUE, num_heads, DEPTH)
    )
    var q_lt = LayoutTensor[q_type, q_layout_3d](
        q_device.unsafe_ptr(),
        RuntimeLayout[q_layout_3d].row_major(
            Index(total_q_tokens, num_heads, DEPTH)
        ),
    )

    comptime out_layout_3d = Layout.row_major(
        Index(UNKNOWN_VALUE, num_heads, V_DEPTH)
    )
    var out_lt = LayoutTensor[q_type, out_layout_3d](
        out_device.unsafe_ptr(),
        RuntimeLayout[out_layout_3d].row_major(
            Index(total_q_tokens, num_heads, V_DEPTH)
        ),
    )

    comptime ro_layout = Layout.row_major(UNKNOWN_VALUE)
    var row_offsets_lt = LayoutTensor[DType.uint32, ro_layout](
        row_offsets_device.unsafe_ptr(),
        RuntimeLayout[ro_layout].row_major(Index(batch_size + 1)),
    )

    # Step 7: Benchmark - warmup + timed iterations
    @parameter
    @always_inline
    @__copy_capture(out_lt, q_lt, kv_cache, row_offsets_lt)
    fn kernel_launch(ctx: DeviceContext) raises:
        flare_mla_decoding[rank=3, ragged=True](
            out_lt,
            q_lt,
            kv_cache,
            NullMask(),
            IdentityScoreMod(),
            row_offsets_lt,
            scale,
            ctx,
            q_max_seq_len,
        )

    comptime nrun = 200

    # Warmup
    kernel_launch(ctx)

    var nstime = Float64(ctx.execution_time[kernel_launch](nrun)) / Float64(
        nrun
    )
    var mstime = nstime / 1000000
    print("  ", nrun, "runs avg", mstime, "ms")
    print()

    # Cleanup
    q_host.free()
    out_host.free()
    blocks_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    row_offsets_host.free()

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


# ===-----------------------------------------------------------------------===#
# Helper functions to reduce duplication in main()
# ===-----------------------------------------------------------------------===#


fn make_uniform(count: Int, value: Int) -> List[Int]:
    """Create a list of `count` identical cache lengths."""
    var result = List[Int]()
    for _ in range(count):
        result.append(value)
    return result^


fn run_both_kv_types[
    num_heads: Int
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext) raises:
    """Run correctness test for both bf16 and fp8 KV types."""
    run_test_paged_variable[DType.bfloat16, DType.bfloat16, num_heads](
        name + "_bf16", cache_lengths, ctx
    )
    run_test_paged_variable[DType.bfloat16, DType.float8_e4m3fn, num_heads](
        name + "_fp8", cache_lengths, ctx
    )


fn run_uniform_both[
    num_heads: Int
](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext) raises:
    """Run correctness test with uniform cache lengths for both KV types."""
    run_both_kv_types[num_heads](name, make_uniform(count, value), ctx)


fn run_bench_both_kv_types[
    num_heads: Int
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext) raises:
    """Run benchmark for both bf16 and fp8 KV types."""
    run_bench_paged_variable[DType.bfloat16, DType.bfloat16, num_heads](
        name + "_bf16", cache_lengths, ctx
    )
    run_bench_paged_variable[DType.bfloat16, DType.float8_e4m3fn, num_heads](
        name + "_fp8", cache_lengths, ctx
    )


fn run_bench_uniform_both[
    num_heads: Int
](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext) raises:
    """Run benchmark with uniform cache lengths for both KV types."""
    run_bench_both_kv_types[num_heads](name, make_uniform(count, value), ctx)


# ===-----------------------------------------------------------------------===#
# Entry point
# ===-----------------------------------------------------------------------===#


def main():
    seed(42)

    with DeviceContext() as ctx:

        @parameter
        if has_nvidia_gpu_accelerator() and ctx.default_device_info == B200:
            if is_benchmark():
                # -----------------------------------------------------------
                # Benchmark mode: time kernel execution, no verification
                # DeepSeek-V2/R1 full: 128 heads, depth=576
                # -----------------------------------------------------------
                print("=" * 72)
                print("MLA Decode Paged Variable-Length BENCHMARK (B200)")
                print("=" * 72)
                print()

                # Batch size 1: single long context
                run_bench_uniform_both[128]("bs1_32k", 1, 32768, ctx)

                # Batch size 1: medium context
                run_bench_uniform_both[128]("bs1_4k", 1, 4096, ctx)

                # Batch size 2: variable lengths
                var b2 = List[Int]()
                b2.append(4096)
                b2.append(32768)
                run_bench_both_kv_types[128]("bs2_4k_32k", b2, ctx)

                # Batch size 4: mixed lengths
                var b4 = List[Int]()
                b4.append(1024)
                b4.append(4096)
                b4.append(16384)
                b4.append(32768)
                run_bench_both_kv_types[128]("bs4_mixed", b4, ctx)

                # Batch size 8: mixed lengths (production-like)
                var b8 = List[Int]()
                b8.append(128)
                b8.append(512)
                b8.append(1024)
                b8.append(4096)
                b8.append(8192)
                b8.append(16384)
                b8.append(24576)
                b8.append(32768)
                run_bench_both_kv_types[128]("bs8_mixed", b8, ctx)

                # Batch size 8: all long (worst case)
                run_bench_uniform_both[128]("bs8_all32k", 8, 32768, ctx)

                print("=" * 72)
                print("BENCHMARK COMPLETE")
                print("=" * 72)

            else:
                # -----------------------------------------------------------
                # Correctness mode: numerical verification against reference
                # -----------------------------------------------------------
                print("=" * 72)
                print("MLA Decode Paged Variable-Length Test (B200)")
                print("=" * 72)
                print()

                # -----------------------------------------------------------
                # Group 1: Small caches with reference check (16 heads)
                # Both bf16 and fp8 KV types are tested to exercise both
                # mla_decode_sm100_kv_bf16 and mla_decode_sm100_kv_fp8
                # kernels.
                # -----------------------------------------------------------
                print(
                    "--- Group 1: Small caches with reference check"
                    " (16 heads) ---"
                )

                var cl1 = List[Int]()
                cl1.append(30)
                cl1.append(50)
                cl1.append(80)
                cl1.append(100)
                run_both_kv_types[16]("short_uniform_q1", cl1, ctx)

                var cl2 = List[Int]()
                cl2.append(30)
                cl2.append(256)
                cl2.append(640)
                cl2.append(1024)
                run_both_kv_types[16]("variable_cache_q1", cl2, ctx)

                var cl3 = List[Int]()
                cl3.append(30)
                cl3.append(128)
                cl3.append(512)
                cl3.append(2048)
                run_both_kv_types[16]("moderate_disparity_q1", cl3, ctx)

                # -----------------------------------------------------------
                # Group 2: Large caches with reference check (16 heads)
                # -----------------------------------------------------------
                print(
                    "--- Group 2: Large caches with reference check"
                    " (16 heads) ---"
                )

                var cl4 = List[Int]()
                cl4.append(30)
                cl4.append(1024)
                cl4.append(8192)
                cl4.append(32768)
                run_both_kv_types[16]("extreme_disparity_q1", cl4, ctx)

                var cl5 = List[Int]()
                cl5.append(16384)
                cl5.append(20000)
                cl5.append(24576)
                cl5.append(32768)
                run_both_kv_types[16]("all_long_q1", cl5, ctx)

                var cl6 = List[Int]()
                cl6.append(30)
                cl6.append(128)
                cl6.append(256)
                cl6.append(512)
                cl6.append(1024)
                cl6.append(4096)
                cl6.append(16384)
                cl6.append(32768)
                run_both_kv_types[16]("mixed_8batch_q1", cl6, ctx)

                # -----------------------------------------------------------
                # Group 3: Higher head counts (64, 128) with verification
                # -----------------------------------------------------------
                print(
                    "--- Group 3: Higher head counts with reference check ---"
                )

                var cl7 = List[Int]()
                cl7.append(30)
                cl7.append(512)
                cl7.append(4096)
                cl7.append(16384)
                run_both_kv_types[64]("64heads_disparity_q1", cl7, ctx)

                var cl8 = List[Int]()
                cl8.append(30)
                cl8.append(1024)
                cl8.append(8192)
                cl8.append(32768)
                run_both_kv_types[128]("128heads_extreme_q1", cl8, ctx)

                # -----------------------------------------------------------
                # Group 4: Latency sensitive configs (uniform cache lengths)
                # Matches bench_mla_decode.yaml latency sensitive section
                # -----------------------------------------------------------
                print("--- Group 4: Latency sensitive configs (16 heads) ---")

                run_uniform_both[16]("lat_bs1_32k", 1, 32768, ctx)
                run_uniform_both[16]("lat_bs1_64k", 1, 65536, ctx)
                run_uniform_both[16]("lat_bs2_32k", 2, 32768, ctx)
                run_uniform_both[16]("lat_bs2_64k", 2, 65536, ctx)
                run_uniform_both[16]("lat_bs4_32k", 4, 32768, ctx)
                run_uniform_both[16]("lat_bs4_64k", 4, 65536, ctx)
                run_uniform_both[16]("lat_bs8_32k", 8, 32768, ctx)
                run_uniform_both[16]("lat_bs8_64k", 8, 65536, ctx)

                # -----------------------------------------------------------
                # Group 5: Production-representative configs
                # SGLang/Ant Group: 12-48 per GPU, vLLM TP8: up to 128
                # KV cache 4K-8K is the common production range
                # -----------------------------------------------------------
                print(
                    "--- Group 5: Production-representative configs"
                    " (16 heads) ---"
                )

                run_uniform_both[16]("prod_bs16_4k", 16, 4096, ctx)
                run_uniform_both[16]("prod_bs16_8k", 16, 8192, ctx)
                run_uniform_both[16]("prod_bs32_4k", 32, 4096, ctx)
                run_uniform_both[16]("prod_bs32_8k", 32, 8192, ctx)
                run_uniform_both[16]("prod_bs64_2k", 64, 2048, ctx)
                run_uniform_both[16]("prod_bs128_1k", 128, 1024, ctx)

                # -----------------------------------------------------------
                # Group 6: Extra-large batch stress test configs
                # -----------------------------------------------------------
                print(
                    "--- Group 6: Extra-large batch stress test configs"
                    " (16 heads) ---"
                )

                run_uniform_both[16]("stress_bs256_2k", 256, 2048, ctx)
                run_uniform_both[16]("stress_bs256_4k", 256, 4096, ctx)
                run_uniform_both[16]("stress_bs512_2k", 512, 2048, ctx)

                # Batch size 512, cache 4096 -- skipped: exceeds test GPU
                # memory budget (~2.4GB). Covered by bench_mla_decode.yaml.

                # -----------------------------------------------------------
                # Group 7: Very small cache_len (<1K)
                # Tests the kernel with very few KV pages -- potential
                # underutilization. Covers prefill-like decode scenarios
                # with tiny cache and high batch.
                # -----------------------------------------------------------
                print(
                    "--- Group 7: Very small cache_len configs (16 heads) ---"
                )

                run_uniform_both[16]("tiny_bs128_64", 128, 64, ctx)
                run_uniform_both[16]("tiny_bs128_128", 128, 128, ctx)
                run_uniform_both[16]("tiny_bs64_256", 64, 256, ctx)
                run_uniform_both[16]("tiny_bs64_512", 64, 512, ctx)

                # -----------------------------------------------------------
                # Group 8: Mid-range configs (fills 8K->32K gap with 16K)
                # -----------------------------------------------------------
                print("--- Group 8: Mid-range configs (16 heads) ---")

                run_uniform_both[16]("mid_bs8_16k", 8, 16384, ctx)
                run_uniform_both[16]("mid_bs4_16k", 4, 16384, ctx)

                # -----------------------------------------------------------
                # Group 9: Large cache_len (>64K up to 163K max context)
                # Tests deep into long-context territory for DeepSeek.
                # -----------------------------------------------------------
                print("--- Group 9: Large cache_len configs (16 heads) ---")

                run_uniform_both[16]("large_bs1_98k", 1, 98304, ctx)
                run_uniform_both[16]("large_bs1_131k", 1, 131072, ctx)
                run_uniform_both[16]("large_bs1_163k", 1, 163840, ctx)

                # -----------------------------------------------------------
                # Group 10: Non-power-of-2 cache_len configs
                # Validates that kernel dispatch, page-level partitioning,
                # and split-k reduction handle arbitrary cache lengths.
                # -----------------------------------------------------------
                print(
                    "--- Group 10: Non-power-of-2 cache_len configs"
                    " (16 heads) ---"
                )

                run_uniform_both[16]("npo2_bs32_3000", 32, 3000, ctx)
                run_uniform_both[16]("npo2_bs16_5000", 16, 5000, ctx)
                run_uniform_both[16]("npo2_bs16_7777", 16, 7777, ctx)
                run_uniform_both[16]("npo2_bs8_10000", 8, 10000, ctx)
                run_uniform_both[16]("npo2_bs2_50000", 2, 50000, ctx)

                # -----------------------------------------------------------
                # Group 11: Variable cache_len across batch (log-spaced)
                # Tests the kernel with heterogeneous cache lengths in a
                # single batch, sampled in log space from 128 to 163840
                # (DeepSeek max context). This exercises the variable-length
                # dispatch path, page-level partitioning with different
                # split counts per sequence, and the combine kernel with
                # mixed workloads.
                # -----------------------------------------------------------
                print(
                    "--- Group 11: Variable cache_len across batch"
                    " (log-spaced, 16 heads) ---"
                )

                # 29 sequences with log-spaced cache lengths from 128 to
                # 163840, rounded to multiples of 128 (page_size).
                var variable_logspace = List[Int]()
                variable_logspace.append(128)
                variable_logspace.append(128)
                variable_logspace.append(256)
                variable_logspace.append(256)
                variable_logspace.append(384)
                variable_logspace.append(512)
                variable_logspace.append(640)
                variable_logspace.append(896)
                variable_logspace.append(1152)
                variable_logspace.append(1536)
                variable_logspace.append(2048)
                variable_logspace.append(2688)
                variable_logspace.append(3456)
                variable_logspace.append(4480)
                variable_logspace.append(5760)
                variable_logspace.append(7552)
                variable_logspace.append(9728)
                variable_logspace.append(12544)
                variable_logspace.append(16128)
                variable_logspace.append(20864)
                variable_logspace.append(26880)
                variable_logspace.append(34560)
                variable_logspace.append(44544)
                variable_logspace.append(57344)
                variable_logspace.append(73856)
                variable_logspace.append(95104)
                variable_logspace.append(122496)
                variable_logspace.append(157696)
                variable_logspace.append(163840)
                run_both_kv_types[16](
                    "variable_logspace", variable_logspace, ctx
                )

                print("=" * 72)
                print("ALL TESTS PASSED")
                print("=" * 72)
        else:
            print("Skipping: requires B200 GPU")
