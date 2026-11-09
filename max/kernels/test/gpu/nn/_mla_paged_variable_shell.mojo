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

from std.math import align_up, ceildiv
from std.random import randn, seed
from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask, MHAMask, NullMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from std.testing import assert_almost_equal
from _mla_test_vendors import MLATestVendor


# ===-----------------------------------------------------------------------===#
# Test parameters matching DeepSeek-V2-Lite
# ===-----------------------------------------------------------------------===#

comptime DEPTH = 576  # Q/K head dimension
comptime V_DEPTH = 512  # Output head dimension (depth - 64)
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime NUM_LAYERS = 1  # Single layer for testing
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# ===-----------------------------------------------------------------------===#
# Core test: paged KV cache with variable lengths + numerical verification
# ===-----------------------------------------------------------------------===#


def make_uniform(count: Int, value: Int) -> List[Int]:
    """Create a list of `count` identical cache lengths."""
    var result = List[Int]()
    for _ in range(count):
        result.append(value)
    return result^


struct PagedVariableTests[V: MLATestVendor]:
    """The paged variable-length MLA decode tests, run on `V`'s kernels."""

    @staticmethod
    def run_test_paged_variable[
        q_type: DType,
        kv_type: DType,
        num_heads: Int,
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        ctx: DeviceContext[],
        num_partitions: Optional[Int] = None,
    ) raises:
        var batch_size = len(cache_lengths)
        print(
            t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
            t" q_type: {q_type}  kv_type: {kv_type}"
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
            # Match production: allocate pages for cache_len + new token
            total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

        comptime scale = Float32(0.125)
        comptime group = num_heads

        # -----------------------------------------------------------------------
        # Step 1: Create the paged KV cache on host
        # -----------------------------------------------------------------------
        comptime kv_params = KVCacheStaticParams(
            num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
        )
        comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

        var block_elems = (
            total_pages
            * kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )

        var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

        # Generate random data in bf16 then cast to kv_type.  This avoids
        # issues with randn producing poorly-distributed values for float8
        # types.  For bf16 kv_type the cast is a no-op.
        # Use std=0.5 to keep QK dot products moderate and softmax
        # numerically stable across all cache lengths.
        var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
        randn(
            blocks_bf16,
            mean=0.0,
            standard_deviation=0.5,
        )
        for i in range(block_elems):
            blocks_host[i] = blocks_bf16[i].cast[kv_type]()

        # Zero out the unused tail slots in each page so that if the kernel
        # accidentally reads beyond a batch's cache_len, the attention weights
        # on those tokens are negligible (exp of Q*0 is small relative to real data).
        var _page_stride = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        var _tok_stride = kv_params.num_heads * kv_params.head_size

        var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

        # Match production: pages cover cache_len + new token
        # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
        # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
        var max_pages_per_batch = align_up(
            ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
        )
        var lut_size = batch_size * max_pages_per_batch
        var lookup_table_host = List(length=lut_size, fill=UInt32(0))

        var page_offset = 0
        for i in range(batch_size):
            var num_pages_i = ceildiv(
                cache_lengths[i] + q_max_seq_len, PAGE_SIZE
            )
            for p in range(num_pages_i):
                lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                    page_offset + p
                )
            page_offset += num_pages_i

        # Zero out tail slots in each page (tokens beyond num_keys).
        var cur_page = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + q_max_seq_len
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for pg in range(num_pages_i):
                var valid_toks = num_keys_i - pg * PAGE_SIZE
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
        var q_host = List(length=q_size, fill=Scalar[q_type](0))
        randn(q_host, mean=0.0, standard_deviation=0.5)

        # -----------------------------------------------------------------------
        # Step 3: input_row_offsets (batch_size + 1 elements)
        # -----------------------------------------------------------------------
        var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
        row_offsets_host[0] = UInt32(0)
        for i in range(batch_size):
            row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

        # -----------------------------------------------------------------------
        # Step 4: Output tensor
        # -----------------------------------------------------------------------
        var out_size = total_q_tokens * num_heads * V_DEPTH
        var out_host = List(length=out_size, fill=Scalar[q_type](0))

        # -----------------------------------------------------------------------
        # Step 5: Copy everything to device
        # -----------------------------------------------------------------------
        var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
        ctx.enqueue_copy(blocks_device, blocks_host)

        var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
            batch_size
        )
        ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

        var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
        ctx.enqueue_copy(lookup_table_device, lookup_table_host)

        var q_device = ctx.enqueue_create_buffer[q_type](q_size)
        ctx.enqueue_copy(q_device, q_host)

        var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
            batch_size + 1
        )
        ctx.enqueue_copy(row_offsets_device, row_offsets_host)

        var out_device = ctx.enqueue_create_buffer[q_type](out_size)

        ctx.synchronize()

        # -----------------------------------------------------------------------
        # Step 6: Build TileTensors and PagedKVCacheCollection on device
        # -----------------------------------------------------------------------

        var blocks_tt = TileTensor(
            blocks_device,
            row_major(
                Int64(total_pages),
                Idx[kv_dim2],
                Idx[NUM_LAYERS],
                Idx[PAGE_SIZE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        )
        var cache_lengths_tt = TileTensor(
            cache_lengths_device, row_major(Int64(batch_size))
        )
        var lookup_table_tt = TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(max_pages_per_batch)),
        )
        var kv_collection = PagedKVCacheCollection[
            kv_type,
            kv_params,
            PAGE_SIZE,
            MutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            MutAnyOrigin,
        ](
            blocks_tt.as_unsafe_any_origin(),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
        )

        var kv_cache = kv_collection.get_key_cache(0)

        var q_tt = TileTensor(
            q_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
        )

        var out_tt = TileTensor(
            out_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        )

        var row_offsets_tt = TileTensor(
            row_offsets_device,
            row_major(batch_size + 1),
        )

        # -----------------------------------------------------------------------
        # Step 7: Pre-compute scalar args and call the kernel
        # -----------------------------------------------------------------------
        var mla_args = Self.V.scalar_args[
            num_heads, is_fp8_kv=(kv_type == .float8_e4m3fn)
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        var scalar_args_buf_tt = TileTensor(
            mla_args.unsafe_ptr(), row_major[3]()
        )

        print("  Launching MLA decode kernel...")

        flare_mla_decoding[
            kernels=Self.V.Kernels,
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            q_max_seq_len=q_max_seq_len,
            num_partitions=num_partitions,
        )

        ctx.synchronize()
        print("  Kernel completed successfully (no crash).")

        _ = mla_args^

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
            # num_keys includes the new token (production: cache_len + seq_len)
            var ref_num_keys = cache_len + q_max_seq_len

            # Extract contiguous K for this batch from paged blocks
            var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
            var k_b_host = List(length=k_b_size, fill=Scalar[kv_type](0))

            var page_base_b = 0
            for bi in range(b):
                page_base_b += ceildiv(
                    cache_lengths[bi] + q_max_seq_len, PAGE_SIZE
                )

            for tok in range(ref_num_keys):
                var page_idx, tok_in_page = divmod(tok, PAGE_SIZE)
                var physical_page = page_base_b + page_idx

                var src_offset = (
                    physical_page
                    * kv_dim2
                    * NUM_LAYERS
                    * PAGE_SIZE
                    * kv_params.num_heads
                    * kv_params.head_size
                    + tok_in_page * kv_params.num_heads * kv_params.head_size
                )
                var dst_offset = tok * KV_NUM_HEADS * DEPTH
                for d in range(KV_NUM_HEADS * DEPTH):
                    k_b_host[dst_offset + d] = blocks_host[src_offset + d]

            # Q for this batch: [1, 1, num_heads, depth]
            var q_b_size = 1 * num_heads * DEPTH
            var q_b_host = List(length=q_b_size, fill=Scalar[q_type](0))
            for i in range(q_b_size):
                q_b_host[i] = q_host[b * num_heads * DEPTH + i]

            # Reference output: [1, 1, num_heads, depth] (full depth)
            var ref_b_size = 1 * num_heads * DEPTH
            var ref_b_host = List(length=ref_b_size, fill=Scalar[q_type](0))

            # Copy to device
            var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
            ctx.enqueue_copy(k_b_device, k_b_host)

            var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
            ctx.enqueue_copy(q_b_device, q_b_host)

            var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
            ctx.synchronize()

            # Build 4D TileTensors for mha_gpu_naive reference
            var q_b_tt = TileTensor(
                q_b_device,
                row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
            )
            var k_b_tt = TileTensor(
                k_b_device,
                row_major(
                    (
                        Idx[1],
                        ref_num_keys,
                        Idx[KV_NUM_HEADS],
                        Idx[DEPTH],
                    )
                ),
            )
            var ref_b_tt = TileTensor(
                ref_b_device,
                row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
            )

            # Run mha_gpu_naive: batch_size=1, num_keys=ref_num_keys
            # K passed as both K and V (MLA: V = K[:,:,:512])
            mha_gpu_naive(
                q_b_tt,
                k_b_tt,
                k_b_tt,
                NullMask(),
                ref_b_tt,
                scale,
                1,  # batch_size
                1,  # seq_len
                ref_num_keys,  # num_keys (cache_len + new token)
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
                    var expect = ref_b_host[d + DEPTH * h].cast[.float64]()
                    var actual = out_host[out_offset + V_DEPTH * h + d].cast[
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
            _ = k_b_device
            _ = q_b_device
            _ = ref_b_device
            _ = ref_b_host^
            _ = q_b_host^
            _ = k_b_host^

        print(
            "  Verified:",
            total_checked,
            "elements, max_abs_err:",
            max_abs_err,
        )

        print("  PASS:", name, "\n")

        # Cleanup

        _ = blocks_device
        _ = cache_lengths_device
        _ = lookup_table_device
        _ = q_device
        _ = row_offsets_device
        _ = out_device
        _ = row_offsets_host^
        _ = lookup_table_host^
        _ = cache_lengths_host^
        _ = blocks_host^
        _ = out_host^
        _ = q_host^
        _ = blocks_bf16^

    # ===-----------------------------------------------------------------------===#
    # Core test: paged KV cache with variable lengths AND q_max_seq_len > 1
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    def run_test_paged_variable_multiq[
        q_type: DType,
        kv_type: DType,
        num_heads: Int,
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        q_max_seq_len: Int,
        ctx: DeviceContext[],
    ) raises:
        """Test MLA decode with paged KV cache, variable cache lengths, AND
        q_max_seq_len > 1 (multiple query tokens per batch entry).

        This exercises the ragged layout path where the grid iterates over
        query tokens within a batch entry, testing that:
        - Q row offsets are computed correctly for multi-token decode
        - Output row offsets include the seq_len dimension
        - The combine kernel's o_accum_split layout handles the padded seq dimension
        - FP8 blockwise scale loading is independent of Q seq_len
        """
        var batch_size = len(cache_lengths)
        print(
            t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
            t" q_type: {q_type}  kv_type: {kv_type}  q_max_seq_len:"
            t" {q_max_seq_len}"
        )
        for i in range(batch_size):
            print("  batch", i, ": cache_len=", cache_lengths[i])

        var total_q_tokens = batch_size * q_max_seq_len

        var max_cache_len = 0
        var total_pages = 0
        for i in range(batch_size):
            if cache_lengths[i] > max_cache_len:
                max_cache_len = cache_lengths[i]
            # Match production: allocate pages for cache_len + new tokens
            total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

        comptime scale = Float32(0.125)
        comptime group = num_heads

        # -----------------------------------------------------------------------
        # Step 1: Create the paged KV cache on host
        # -----------------------------------------------------------------------
        comptime kv_params = KVCacheStaticParams(
            num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
        )
        comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

        var block_elems = (
            total_pages
            * kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )

        var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

        var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
        randn(
            blocks_bf16,
            mean=0.0,
            standard_deviation=0.5,
        )
        for i in range(block_elems):
            blocks_host[i] = blocks_bf16[i].cast[kv_type]()

        var _page_stride = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        var _tok_stride = kv_params.num_heads * kv_params.head_size

        var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

        # Match production: pages cover cache_len + new tokens
        # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
        # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
        var max_pages_per_batch = align_up(
            ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
        )
        var lut_size = batch_size * max_pages_per_batch
        var lookup_table_host = List(length=lut_size, fill=UInt32(0))

        var page_offset = 0
        for i in range(batch_size):
            var num_pages_i = ceildiv(
                cache_lengths[i] + q_max_seq_len, PAGE_SIZE
            )
            for p in range(num_pages_i):
                lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                    page_offset + p
                )
            page_offset += num_pages_i

        # Zero out tail slots in each page (tokens beyond num_keys).
        var cur_page = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + q_max_seq_len
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for pg in range(num_pages_i):
                var valid_toks = num_keys_i - pg * PAGE_SIZE
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
        var q_host = List(length=q_size, fill=Scalar[q_type](0))
        randn(q_host, mean=0.0, standard_deviation=0.5)

        # -----------------------------------------------------------------------
        # Step 3: input_row_offsets (batch_size + 1 elements)
        # Each batch has q_max_seq_len tokens
        # -----------------------------------------------------------------------
        var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
        row_offsets_host[0] = UInt32(0)
        for i in range(batch_size):
            row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(
                q_max_seq_len
            )

        # -----------------------------------------------------------------------
        # Step 4: Output tensor
        # -----------------------------------------------------------------------
        var out_size = total_q_tokens * num_heads * V_DEPTH
        var out_host = List(length=out_size, fill=Scalar[q_type](0))

        # -----------------------------------------------------------------------
        # Step 5: Copy everything to device
        # -----------------------------------------------------------------------
        var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
        ctx.enqueue_copy(blocks_device, blocks_host)

        var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
            batch_size
        )
        ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

        var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
        ctx.enqueue_copy(lookup_table_device, lookup_table_host)

        var q_device = ctx.enqueue_create_buffer[q_type](q_size)
        ctx.enqueue_copy(q_device, q_host)

        var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
            batch_size + 1
        )
        ctx.enqueue_copy(row_offsets_device, row_offsets_host)

        var out_device = ctx.enqueue_create_buffer[q_type](out_size)

        ctx.synchronize()

        # -----------------------------------------------------------------------
        # Step 6: Build TileTensors and PagedKVCacheCollection on device
        # -----------------------------------------------------------------------

        var blocks_tt = TileTensor(
            blocks_device,
            row_major(
                Int64(total_pages),
                Idx[kv_dim2],
                Idx[NUM_LAYERS],
                Idx[PAGE_SIZE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        )
        var cache_lengths_tt = TileTensor(
            cache_lengths_device, row_major(Int64(batch_size))
        )
        var lookup_table_tt = TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(max_pages_per_batch)),
        )
        var kv_collection = PagedKVCacheCollection[
            kv_type,
            kv_params,
            PAGE_SIZE,
            MutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            MutAnyOrigin,
        ](
            blocks_tt.as_unsafe_any_origin(),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
        )

        var kv_cache = kv_collection.get_key_cache(0)

        var q_tt = TileTensor(
            q_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
        )

        var out_tt = TileTensor(
            out_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        )

        var row_offsets_tt = TileTensor(
            row_offsets_device,
            row_major(batch_size + 1),
        )

        # -----------------------------------------------------------------------
        # Step 7: Pre-compute scalar args and call the kernel
        # -----------------------------------------------------------------------
        var mla_args = Self.V.scalar_args[
            num_heads, is_fp8_kv=(kv_type == .float8_e4m3fn)
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        var scalar_args_buf_tt = TileTensor(
            mla_args.unsafe_ptr(), row_major[3]()
        )

        print("  Launching MLA decode kernel...")

        flare_mla_decoding[
            kernels=Self.V.Kernels,
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            q_max_seq_len=q_max_seq_len,
        )

        ctx.synchronize()
        print("  Kernel completed successfully (no crash).")

        # -----------------------------------------------------------------------
        # Step 8: Numerical verification using mha_gpu_naive reference
        # -----------------------------------------------------------------------
        ctx.enqueue_copy(out_host, out_device)
        ctx.synchronize()

        print("  Computing GPU naive reference per batch and comparing...")

        var rtol = 5e-2  # 0.05
        var atol = 3e-1  # 0.3
        var total_checked = 0
        var max_abs_err = Float64(0)

        for b in range(batch_size):
            var cache_len = cache_lengths[b]
            var ref_num_keys = cache_len + q_max_seq_len

            # Extract contiguous K for this batch from paged blocks
            var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
            var k_b_host = List(length=k_b_size, fill=Scalar[kv_type](0))

            var page_base_b = 0
            for bi in range(b):
                page_base_b += ceildiv(
                    cache_lengths[bi] + q_max_seq_len, PAGE_SIZE
                )

            for tok in range(ref_num_keys):
                var page_idx, tok_in_page = divmod(tok, PAGE_SIZE)
                var physical_page = page_base_b + page_idx

                var src_offset = (
                    physical_page
                    * kv_dim2
                    * NUM_LAYERS
                    * PAGE_SIZE
                    * kv_params.num_heads
                    * kv_params.head_size
                    + tok_in_page * kv_params.num_heads * kv_params.head_size
                )
                var dst_offset = tok * KV_NUM_HEADS * DEPTH
                for d in range(KV_NUM_HEADS * DEPTH):
                    k_b_host[dst_offset + d] = blocks_host[src_offset + d]

            # Q for this batch: [1, q_max_seq_len, num_heads, depth]
            var q_b_size = q_max_seq_len * num_heads * DEPTH
            var q_b_host = List(length=q_b_size, fill=Scalar[q_type](0))
            var q_batch_offset = b * q_max_seq_len * num_heads * DEPTH
            for i in range(q_b_size):
                q_b_host[i] = q_host[q_batch_offset + i]

            # Reference output: [1, q_max_seq_len, num_heads, depth] (full depth)
            var ref_b_size = q_max_seq_len * num_heads * DEPTH
            var ref_b_host = List(length=ref_b_size, fill=Scalar[q_type](0))

            # Copy to device
            var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
            ctx.enqueue_copy(k_b_device, k_b_host)

            var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
            ctx.enqueue_copy(q_b_device, q_b_host)

            var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
            ctx.synchronize()

            # Build 4D TileTensors for mha_gpu_naive reference
            var q_b_tt = TileTensor(
                q_b_device,
                row_major((Idx[1], q_max_seq_len, Idx[num_heads], Idx[DEPTH])),
            )
            var k_b_tt = TileTensor(
                k_b_device,
                row_major(
                    (
                        Idx[1],
                        ref_num_keys,
                        Idx[KV_NUM_HEADS],
                        Idx[DEPTH],
                    )
                ),
            )
            var ref_b_tt = TileTensor(
                ref_b_device,
                row_major((Idx[1], q_max_seq_len, Idx[num_heads], Idx[DEPTH])),
            )

            # Run mha_gpu_naive: batch_size=1, seq_len=q_max_seq_len
            # K passed as both K and V (MLA: V = K[:,:,:512])
            mha_gpu_naive(
                q_b_tt,
                k_b_tt,
                k_b_tt,
                NullMask(),
                ref_b_tt,
                scale,
                1,  # batch_size
                q_max_seq_len,  # seq_len
                ref_num_keys,  # num_keys (cache_len + new tokens)
                num_heads,
                DEPTH,
                group,
                ctx,
            )

            ctx.synchronize()
            ctx.enqueue_copy(ref_b_host, ref_b_device)
            ctx.synchronize()

            # Compare first V_DEPTH=512 dims (depth-64) per head, per query token
            # ref layout: [1, q_max_seq_len, num_heads, depth]
            # actual layout: [total_tokens, num_heads, V_DEPTH]
            for s in range(q_max_seq_len):
                var out_offset = (b * q_max_seq_len + s) * num_heads * V_DEPTH
                var ref_s_offset = s * num_heads * DEPTH
                for h in range(num_heads):
                    for d in range(V_DEPTH):
                        var expect = ref_b_host[
                            ref_s_offset + d + DEPTH * h
                        ].cast[DType.float64]()
                        var actual = out_host[
                            out_offset + V_DEPTH * h + d
                        ].cast[DType.float64]()
                        var abs_err = abs(actual - expect)
                        if abs_err > max_abs_err:
                            max_abs_err = abs_err
                        if abs_err > 1e-1:
                            print(b, s, h, d, actual, expect)
                        assert_almost_equal(
                            actual, expect, atol=atol, rtol=rtol
                        )

                total_checked += num_heads * V_DEPTH

            # Cleanup per-batch buffers
            _ = k_b_device
            _ = q_b_device
            _ = ref_b_device
            _ = ref_b_host^
            _ = q_b_host^
            _ = k_b_host^

        print(
            "  Verified:",
            total_checked,
            "elements, max_abs_err:",
            max_abs_err,
        )

        print("  PASS:", name, "\n")

        # Cleanup

        _ = blocks_device
        _ = cache_lengths_device
        _ = lookup_table_device
        _ = q_device
        _ = row_offsets_device
        _ = out_device
        _ = row_offsets_host^
        _ = lookup_table_host^
        _ = cache_lengths_host^
        _ = blocks_host^
        _ = out_host^
        _ = q_host^
        _ = blocks_bf16^

    # ===-----------------------------------------------------------------------===#
    # Core test: truly ragged Q — each batch has a DIFFERENT number of Q tokens
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    def run_test_paged_variable_ragged_q[
        q_type: DType,
        kv_type: DType,
        num_heads: Int,
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        seq_lens: List[Int],
        ctx: DeviceContext[],
    ) raises:
        """Test MLA decode with paged KV cache AND truly variable per-batch Q
        sequence lengths (ragged Q).

        Unlike run_test_paged_variable_multiq (where all batches have the same
        q_max_seq_len), this test gives each batch a DIFFERENT number of query
        tokens.  For example:
            batch 0: 1 Q token
            batch 1: 3 Q tokens
            batch 2: 2 Q tokens
            batch 3: 4 Q tokens
            q_max_seq_len = 4  (the max)
            row_offsets = [0, 1, 4, 6, 10]

        The kernel grid launches q_max_seq_len CTAs per batch and head tile.
        CTAs whose query token >= per-batch seq_len early-exit via the ragged
        guard in OffsetPosition.

        This verifies:
        - OffsetPosition.seq_len is correctly derived from row_offsets per batch
        - The early-exit path writes -inf LSE and zeroed o_accum_split for the
          combine kernel
        - num_keys = cache_length + per-batch seq_len (NOT + q_max_seq_len)
        - Page allocation covers cache_len + seq_lens[i] per batch
        - FP8 blockwise scaling is independent of variable Q lengths
        """
        var batch_size = len(cache_lengths)
        assert (
            len(seq_lens) == batch_size
        ), "cache_lengths and seq_lens must have same length"
        var q_max_seq_len = 0
        var total_q_tokens = 0
        for i in range(batch_size):
            if seq_lens[i] > q_max_seq_len:
                q_max_seq_len = seq_lens[i]
            total_q_tokens += seq_lens[i]

        print(
            t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
            t" q_type: {q_type}  kv_type: {kv_type}  q_max_seq_len:"
            t" {q_max_seq_len}  total_q_tokens: {total_q_tokens}"
        )
        for i in range(batch_size):
            print(
                t"  batch {i} : cache_len= {cache_lengths[i]}  q_seq_len="
                t" {seq_lens[i]}"
            )

        var max_cache_len = 0
        var total_pages = 0
        for i in range(batch_size):
            if cache_lengths[i] > max_cache_len:
                max_cache_len = cache_lengths[i]
            # Each batch needs pages for cache_len + its own seq_len
            total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

        comptime scale = Float32(0.125)
        comptime group = num_heads

        # -----------------------------------------------------------------------
        # Step 1: Create the paged KV cache on host
        # -----------------------------------------------------------------------
        comptime kv_params = KVCacheStaticParams(
            num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
        )
        comptime kv_dim2 = 1

        var block_elems = (
            total_pages
            * kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )

        var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

        var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
        randn(
            blocks_bf16,
            mean=0.0,
            standard_deviation=0.5,
        )
        for i in range(block_elems):
            blocks_host[i] = blocks_bf16[i].cast[kv_type]()

        var _page_stride = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        var _tok_stride = kv_params.num_heads * kv_params.head_size

        var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

        # max_pages_per_batch uses cache_len + seq_len for the max batch
        # We need the lookup table to be wide enough for the batch with the
        # most pages.
        var max_num_keys_any_batch = 0
        for i in range(batch_size):
            var nk = cache_lengths[i] + seq_lens[i]
            if nk > max_num_keys_any_batch:
                max_num_keys_any_batch = nk
        # Pad to multiple of 8 for LUT row stride alignment (SIMD populate).
        var max_pages_per_batch = align_up(
            ceildiv(max_num_keys_any_batch, PAGE_SIZE), 8
        )
        var lut_size = batch_size * max_pages_per_batch
        var lookup_table_host = List(length=lut_size, fill=UInt32(0))

        var page_offset = 0
        for i in range(batch_size):
            var num_pages_i = ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)
            for p in range(num_pages_i):
                lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                    page_offset + p
                )
            page_offset += num_pages_i

        # Zero out tail slots in each page (tokens beyond num_keys).
        var cur_page = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + seq_lens[bi]
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for pg in range(num_pages_i):
                var valid_toks = num_keys_i - pg * PAGE_SIZE
                if valid_toks > PAGE_SIZE:
                    valid_toks = PAGE_SIZE
                var base = cur_page * _page_stride + valid_toks * _tok_stride
                var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
                for z in range(zero_count):
                    blocks_host[base + z] = 0
                cur_page += 1

        # -----------------------------------------------------------------------
        # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, DEPTH])
        # -----------------------------------------------------------------------
        var q_size = total_q_tokens * num_heads * DEPTH
        var q_host = List(length=q_size, fill=Scalar[q_type](0))
        randn(q_host, mean=0.0, standard_deviation=0.5)

        # -----------------------------------------------------------------------
        # Step 3: input_row_offsets — truly ragged per batch
        # -----------------------------------------------------------------------
        var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
        row_offsets_host[0] = UInt32(0)
        for i in range(batch_size):
            row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

        # -----------------------------------------------------------------------
        # Step 4: Output tensor
        # -----------------------------------------------------------------------
        var out_size = total_q_tokens * num_heads * V_DEPTH
        var out_host = List(length=out_size, fill=Scalar[q_type](0))

        # -----------------------------------------------------------------------
        # Step 5: Copy everything to device
        # -----------------------------------------------------------------------
        var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
        ctx.enqueue_copy(blocks_device, blocks_host)

        var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
            batch_size
        )
        ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

        var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
        ctx.enqueue_copy(lookup_table_device, lookup_table_host)

        var q_device = ctx.enqueue_create_buffer[q_type](q_size)
        ctx.enqueue_copy(q_device, q_host)

        var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
            batch_size + 1
        )
        ctx.enqueue_copy(row_offsets_device, row_offsets_host)

        var out_device = ctx.enqueue_create_buffer[q_type](out_size)

        ctx.synchronize()

        # -----------------------------------------------------------------------
        # Step 6: Build TileTensors and PagedKVCacheCollection on device
        # -----------------------------------------------------------------------

        var blocks_tt = TileTensor(
            blocks_device,
            row_major(
                Int64(total_pages),
                Idx[kv_dim2],
                Idx[NUM_LAYERS],
                Idx[PAGE_SIZE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        )
        var cache_lengths_tt = TileTensor(
            cache_lengths_device, row_major(Int64(batch_size))
        )
        var lookup_table_tt = TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(max_pages_per_batch)),
        )
        var kv_collection = PagedKVCacheCollection[
            kv_type,
            kv_params,
            PAGE_SIZE,
            MutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            MutAnyOrigin,
        ](
            blocks_tt.as_unsafe_any_origin(),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
        )

        var kv_cache = kv_collection.get_key_cache(0)

        var q_tt = TileTensor(
            q_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
        )

        var out_tt = TileTensor(
            out_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        )

        var row_offsets_tt = TileTensor(
            row_offsets_device,
            row_major(batch_size + 1),
        )

        # -----------------------------------------------------------------------
        # Step 7: Pre-compute scalar args and call the kernel
        # -----------------------------------------------------------------------
        var mla_args = Self.V.scalar_args[
            num_heads, is_fp8_kv=(kv_type == .float8_e4m3fn)
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        var scalar_args_buf_tt = TileTensor(
            mla_args.unsafe_ptr(), row_major[3]()
        )

        print("  Launching MLA decode kernel...")

        flare_mla_decoding[
            kernels=Self.V.Kernels,
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            q_max_seq_len=q_max_seq_len,
        )

        ctx.synchronize()
        print("  Kernel completed successfully (no crash).")

        # -----------------------------------------------------------------------
        # Step 8: Numerical verification using mha_gpu_naive reference
        # -----------------------------------------------------------------------
        ctx.enqueue_copy(out_host, out_device)
        ctx.synchronize()

        print("  Computing GPU naive reference per batch and comparing...")

        var rtol = 5e-2
        var atol = 3e-1
        var total_checked = 0
        var max_abs_err = Float64(0)

        # Track where each batch's Q tokens start in the ragged output
        var q_token_offset = 0

        for b in range(batch_size):
            var cache_len = cache_lengths[b]
            var b_seq_len = seq_lens[b]
            # num_keys for this batch = cache_len + this batch's seq_len
            var ref_num_keys = cache_len + b_seq_len

            # Extract contiguous K for this batch from paged blocks
            var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
            var k_b_host = List(length=k_b_size, fill=Scalar[kv_type](0))

            var page_base_b = 0
            for bi in range(b):
                page_base_b += ceildiv(
                    cache_lengths[bi] + seq_lens[bi], PAGE_SIZE
                )

            for tok in range(ref_num_keys):
                var page_idx, tok_in_page = divmod(tok, PAGE_SIZE)
                var physical_page = page_base_b + page_idx

                var src_offset = (
                    physical_page
                    * kv_dim2
                    * NUM_LAYERS
                    * PAGE_SIZE
                    * kv_params.num_heads
                    * kv_params.head_size
                    + tok_in_page * kv_params.num_heads * kv_params.head_size
                )
                var dst_offset = tok * KV_NUM_HEADS * DEPTH
                for d in range(KV_NUM_HEADS * DEPTH):
                    k_b_host[dst_offset + d] = blocks_host[src_offset + d]

            # Q for this batch: [1, b_seq_len, num_heads, depth]
            var q_b_size = b_seq_len * num_heads * DEPTH
            var q_b_host = List(length=q_b_size, fill=Scalar[q_type](0))
            var q_batch_start = q_token_offset * num_heads * DEPTH
            for i in range(q_b_size):
                q_b_host[i] = q_host[q_batch_start + i]

            # Reference output: [1, b_seq_len, num_heads, depth] (full depth)
            var ref_b_size = b_seq_len * num_heads * DEPTH
            var ref_b_host = List(length=ref_b_size, fill=Scalar[q_type](0))

            # Copy to device
            var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
            ctx.enqueue_copy(k_b_device, k_b_host)

            var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
            ctx.enqueue_copy(q_b_device, q_b_host)

            var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
            ctx.synchronize()

            # Build 4D TileTensors for mha_gpu_naive reference
            var q_b_tt = TileTensor(
                q_b_device,
                row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
            )
            var k_b_tt = TileTensor(
                k_b_device,
                row_major(
                    (
                        Idx[1],
                        ref_num_keys,
                        Idx[KV_NUM_HEADS],
                        Idx[DEPTH],
                    )
                ),
            )
            var ref_b_tt = TileTensor(
                ref_b_device,
                row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
            )

            # Run mha_gpu_naive: batch_size=1, seq_len=b_seq_len
            mha_gpu_naive(
                q_b_tt,
                k_b_tt,
                k_b_tt,
                NullMask(),
                ref_b_tt,
                scale,
                1,  # batch_size
                b_seq_len,  # this batch's seq_len
                ref_num_keys,  # cache_len + this batch's seq_len
                num_heads,
                DEPTH,
                group,
                ctx,
            )

            ctx.synchronize()
            ctx.enqueue_copy(ref_b_host, ref_b_device)
            ctx.synchronize()

            # Compare first V_DEPTH dims per head, per query token
            # ref layout: [1, b_seq_len, num_heads, depth]
            # actual layout: [total_tokens, num_heads, V_DEPTH]  (ragged)
            for s in range(b_seq_len):
                var out_offset = (q_token_offset + s) * num_heads * V_DEPTH
                var ref_s_offset = s * num_heads * DEPTH
                for h in range(num_heads):
                    for d in range(V_DEPTH):
                        var expect = ref_b_host[
                            ref_s_offset + d + DEPTH * h
                        ].cast[DType.float64]()
                        var actual = out_host[
                            out_offset + V_DEPTH * h + d
                        ].cast[DType.float64]()
                        var abs_err = abs(actual - expect)
                        if abs_err > max_abs_err:
                            max_abs_err = abs_err
                        if abs_err > 1e-1:
                            print(
                                t"batch {b} tok {s} head {h} dim {d} actual"
                                t" {actual} expect {expect}"
                            )
                        assert_almost_equal(
                            actual, expect, atol=atol, rtol=rtol
                        )

                total_checked += num_heads * V_DEPTH

            q_token_offset += b_seq_len

            # Cleanup per-batch buffers
            _ = k_b_device
            _ = q_b_device
            _ = ref_b_device
            _ = ref_b_host^
            _ = q_b_host^
            _ = k_b_host^

        print(
            "  Verified:",
            total_checked,
            "elements, max_abs_err:",
            max_abs_err,
        )

        print("  PASS:", name, "\n")

        # Cleanup

        _ = blocks_device
        _ = cache_lengths_device
        _ = lookup_table_device
        _ = q_device
        _ = row_offsets_device
        _ = out_device
        _ = row_offsets_host^
        _ = lookup_table_host^
        _ = cache_lengths_host^
        _ = blocks_host^
        _ = out_host^
        _ = q_host^
        _ = blocks_bf16^

    # ===-----------------------------------------------------------------------===#
    # Benchmark: paged KV cache with variable lengths (no numerical verification)
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    def run_bench_paged_variable[
        q_type: DType,
        kv_type: DType,
        num_heads: Int,
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        ctx: DeviceContext[],
    ) raises:
        var batch_size = len(cache_lengths)
        print(
            t"bench: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
            t" q_type: {q_type}  kv_type: {kv_type}"
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
            # Match production: allocate pages for cache_len + new token
            total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

        comptime scale = Float32(0.125)

        # Step 1: Paged KV cache on host
        comptime kv_params = KVCacheStaticParams(
            num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
        )
        comptime kv_dim2 = 1

        var block_elems = (
            total_pages
            * kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )

        var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
        var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
        randn(
            blocks_bf16,
            mean=0.0,
            standard_deviation=0.5,
        )
        for i in range(block_elems):
            blocks_host[i] = blocks_bf16[i].cast[kv_type]()

        var _page_stride = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        var _tok_stride = kv_params.num_heads * kv_params.head_size

        var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

        # Match production: pages cover cache_len + new token
        # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
        # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
        var max_pages_per_batch = align_up(
            ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
        )
        var lut_size = batch_size * max_pages_per_batch
        var lookup_table_host = List(length=lut_size, fill=UInt32(0))

        var page_offset = 0
        for i in range(batch_size):
            var num_pages_i = ceildiv(
                cache_lengths[i] + q_max_seq_len, PAGE_SIZE
            )
            for p in range(num_pages_i):
                lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                    page_offset + p
                )
            page_offset += num_pages_i

        # Zero out tail slots in each page (tokens beyond num_keys)
        var cur_page = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + q_max_seq_len
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for pg in range(num_pages_i):
                var valid_toks = num_keys_i - pg * PAGE_SIZE
                if valid_toks > PAGE_SIZE:
                    valid_toks = PAGE_SIZE
                var base = cur_page * _page_stride + valid_toks * _tok_stride
                var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
                for z in range(zero_count):
                    blocks_host[base + z] = 0
                cur_page += 1

        # Step 2: Q tensor
        var q_size = total_q_tokens * num_heads * DEPTH
        var q_host = List(length=q_size, fill=Scalar[q_type](0))
        randn(q_host, mean=0.0, standard_deviation=0.5)

        # Step 3: input_row_offsets
        var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
        row_offsets_host[0] = UInt32(0)
        for i in range(batch_size):
            row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

        # Step 4: Output tensor
        var out_size = total_q_tokens * num_heads * V_DEPTH
        var out_host = List(length=out_size, fill=Scalar[q_type](0))

        # Step 5: Copy to device
        var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
        ctx.enqueue_copy(blocks_device, blocks_host)

        var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
            batch_size
        )
        ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

        var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
        ctx.enqueue_copy(lookup_table_device, lookup_table_host)

        var q_device = ctx.enqueue_create_buffer[q_type](q_size)
        ctx.enqueue_copy(q_device, q_host)

        var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
            batch_size + 1
        )
        ctx.enqueue_copy(row_offsets_device, row_offsets_host)

        var out_device = ctx.enqueue_create_buffer[q_type](out_size)

        ctx.synchronize()

        # Step 6: Build TileTensors and PagedKVCacheCollection
        var blocks_tt = TileTensor(
            blocks_device,
            row_major(
                Int64(total_pages),
                Idx[kv_dim2],
                Idx[NUM_LAYERS],
                Idx[PAGE_SIZE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        )
        var cache_lengths_tt = TileTensor(
            cache_lengths_device, row_major(Int64(batch_size))
        )
        var lookup_table_tt = TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(max_pages_per_batch)),
        )
        var kv_collection = PagedKVCacheCollection[
            kv_type,
            kv_params,
            PAGE_SIZE,
            MutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            MutAnyOrigin,
        ](
            blocks_tt.as_unsafe_any_origin(),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
        )

        var kv_cache = kv_collection.get_key_cache(0)

        var q_tt = TileTensor(
            q_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
        )

        var out_tt = TileTensor(
            out_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        )

        var row_offsets_tt = TileTensor(
            row_offsets_device,
            row_major(batch_size + 1),
        )

        # Step 7: Pre-compute scalar args and benchmark
        var mla_args = Self.V.scalar_args[
            num_heads, is_fp8_kv=(kv_type == .float8_e4m3fn)
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        var scalar_args_buf_tt = TileTensor(
            mla_args.unsafe_ptr(), row_major[3]()
        )

        @inline(.always)
        def kernel_launch(
            ctx: DeviceContext[],
        ) raises {
            var out_tt,
            var q_tt,
            var kv_cache,
            var row_offsets_tt,
            var scalar_args_buf_tt,
            imm,
        }:
            flare_mla_decoding[
                kernels=Self.V.Kernels,
                rank=3,
                config=MHAConfig[q_type](num_heads, DEPTH),
                ragged=True,
            ](
                out_tt,
                q_tt,
                kv_cache,
                NullMask(),
                row_offsets_tt,
                scale,
                ctx,
                scalar_args_buf_tt,
                q_max_seq_len=q_max_seq_len,
            )

        comptime nrun = 200

        # Warmup
        kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time(kernel_launch, nrun)) / Float64(
            nrun
        )
        var mstime = nstime / 1000000
        print(t"   {nrun} runs avg {mstime} ms")
        print()

        # Cleanup

        _ = blocks_device
        _ = cache_lengths_device
        _ = lookup_table_device
        _ = q_device
        _ = row_offsets_device
        _ = out_device
        _ = row_offsets_host^
        _ = lookup_table_host^
        _ = cache_lengths_host^
        _ = blocks_host^
        _ = out_host^
        _ = q_host^
        _ = blocks_bf16^

    # ===-----------------------------------------------------------------------===#
    # Core test: native FP8 paged path (Q=FP8, KV=FP8, output=BF16)
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    def run_test_paged_variable_native_fp8[
        num_heads: Int,
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        ctx: DeviceContext[],
        num_partitions: Optional[Int] = None,
    ) raises:
        """Test native FP8 MLA decode through the paged path.

        In the native FP8 path:
        - Q is float8_e4m3fn (sent to the kernel as FP8 via TMA)
        - KV cache is float8_e4m3fn (tensorwise, scale_block_size=0)
        - Output is bfloat16

        This exercises the 3-WG native FP8 kernel (MLA_SM100_Decode_QKV_FP8)
        with split-K + PDL through the paged KV cache path.

        For the GPU naive reference, we use BF16 Q and dequantized BF16 K
        since mha_gpu_naive does not support FP8 inputs.
        """
        comptime q_type = DType.float8_e4m3fn
        comptime kv_type = DType.float8_e4m3fn
        comptime ref_type = DType.bfloat16  # Reference/output type

        var batch_size = len(cache_lengths)
        print(
            t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
            t" q_type: {q_type}  kv_type: {kv_type}  output_type: {ref_type} "
            t" num_partitions:"
            t" {num_partitions.value() if num_partitions else -1}"
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
            total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

        comptime scale = Float32(0.125)
        comptime group = num_heads

        # -----------------------------------------------------------------------
        # Step 1: Create the paged KV cache on host (FP8)
        # -----------------------------------------------------------------------
        comptime kv_params = KVCacheStaticParams(
            num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
        )
        comptime kv_dim2 = 1

        var block_elems = (
            total_pages
            * kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )

        var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

        # Generate KV data in bf16 first, then cast to FP8.
        # Keep bf16 copy for the reference path.
        var blocks_bf16 = List(length=block_elems, fill=Scalar[ref_type](0))
        randn(
            blocks_bf16,
            mean=0.0,
            standard_deviation=0.5,
        )
        for i in range(block_elems):
            blocks_host[i] = blocks_bf16[i].cast[kv_type]()
        # Note: blocks_bf16 is kept alive for reference computation below.

        var _page_stride = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        var _tok_stride = kv_params.num_heads * kv_params.head_size

        var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

        # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
        # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
        var max_pages_per_batch = align_up(
            ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
        )
        var lut_size = batch_size * max_pages_per_batch
        var lookup_table_host = List(length=lut_size, fill=UInt32(0))

        var page_offset = 0
        for i in range(batch_size):
            var num_pages_i = ceildiv(
                cache_lengths[i] + q_max_seq_len, PAGE_SIZE
            )
            for p in range(num_pages_i):
                lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                    page_offset + p
                )
            page_offset += num_pages_i

        # Zero out tail slots in both FP8 and BF16 copies
        var cur_page = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + q_max_seq_len
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for pg in range(num_pages_i):
                var valid_toks = num_keys_i - pg * PAGE_SIZE
                if valid_toks > PAGE_SIZE:
                    valid_toks = PAGE_SIZE
                var base = cur_page * _page_stride + valid_toks * _tok_stride
                var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
                for z in range(zero_count):
                    blocks_host[base + z] = 0
                    blocks_bf16[base + z] = 0
                cur_page += 1

        # -----------------------------------------------------------------------
        # Step 2: Q tensor — generate in BF16, keep copy, cast to FP8
        # -----------------------------------------------------------------------
        var q_size = total_q_tokens * num_heads * DEPTH

        # BF16 Q for reference
        var q_bf16_host = List(length=q_size, fill=Scalar[ref_type](0))
        randn(
            q_bf16_host,
            mean=0.0,
            standard_deviation=0.5,
        )

        # FP8 Q for kernel
        var q_fp8_host = List(length=q_size, fill=Scalar[q_type](0))
        for i in range(q_size):
            q_fp8_host[i] = q_bf16_host[i].cast[q_type]()

        # -----------------------------------------------------------------------
        # Step 3: input_row_offsets (batch_size + 1 elements)
        # -----------------------------------------------------------------------
        var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
        row_offsets_host[0] = UInt32(0)
        for i in range(batch_size):
            row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

        # -----------------------------------------------------------------------
        # Step 4: Output tensor (BF16, not FP8)
        # -----------------------------------------------------------------------
        var out_size = total_q_tokens * num_heads * V_DEPTH
        var out_host = List(length=out_size, fill=Scalar[ref_type](0))

        # -----------------------------------------------------------------------
        # Step 5: Copy everything to device
        # -----------------------------------------------------------------------
        var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
        ctx.enqueue_copy(blocks_device, blocks_host)

        var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
            batch_size
        )
        ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

        var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
        ctx.enqueue_copy(lookup_table_device, lookup_table_host)

        # FP8 Q to device
        var q_device = ctx.enqueue_create_buffer[q_type](q_size)
        ctx.enqueue_copy(q_device, q_fp8_host)

        var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
            batch_size + 1
        )
        ctx.enqueue_copy(row_offsets_device, row_offsets_host)

        # Output is BF16
        var out_device = ctx.enqueue_create_buffer[ref_type](out_size)

        ctx.synchronize()

        # -----------------------------------------------------------------------
        # Step 6: Build TileTensors and PagedKVCacheCollection on device
        # -----------------------------------------------------------------------
        var blocks_tt = TileTensor(
            blocks_device,
            row_major(
                Int64(total_pages),
                Idx[kv_dim2],
                Idx[NUM_LAYERS],
                Idx[PAGE_SIZE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        )
        var cache_lengths_tt = TileTensor(
            cache_lengths_device, row_major(Int64(batch_size))
        )
        var lookup_table_tt = TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(max_pages_per_batch)),
        )
        var kv_collection = PagedKVCacheCollection[
            kv_type,
            kv_params,
            PAGE_SIZE,
            MutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            MutAnyOrigin,
        ](
            blocks_tt.as_unsafe_any_origin(),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
        )

        var kv_cache = kv_collection.get_key_cache(0)

        # Q is FP8, output is BF16
        var q_tt = TileTensor(
            q_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
        )

        var out_tt = TileTensor(
            out_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        )

        var row_offsets_tt = TileTensor(
            row_offsets_device,
            row_major(batch_size + 1),
        )

        # -----------------------------------------------------------------------
        # Step 7: Pre-compute scalar args and call the kernel (Q=FP8, KV=FP8, output=BF16)
        # -----------------------------------------------------------------------
        var mla_args = Self.V.scalar_args[
            num_heads, is_fp8_kv=(kv_type == .float8_e4m3fn)
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        var scalar_args_buf_tt = TileTensor(
            mla_args.unsafe_ptr(), row_major[3]()
        )
        print("  Launching native FP8 MLA decode kernel...")

        flare_mla_decoding[
            kernels=Self.V.Kernels,
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            q_max_seq_len=q_max_seq_len,
            num_partitions=num_partitions,
        )

        ctx.synchronize()
        print("  Kernel completed successfully (no crash).")

        # -----------------------------------------------------------------------
        # Step 8: Numerical verification using mha_gpu_naive reference
        # Uses BF16 Q and dequantized BF16 K for the reference.
        # -----------------------------------------------------------------------
        ctx.enqueue_copy(out_host, out_device)
        ctx.synchronize()

        print("  Computing GPU naive reference per batch and comparing...")

        # FP8 has larger quantization error, so use wider tolerances.
        var rtol = 8e-2
        var atol = 5e-1
        var total_checked = 0
        var max_abs_err = Float64(0)

        for b in range(batch_size):
            var cache_len = cache_lengths[b]
            var ref_num_keys = cache_len + q_max_seq_len

            # Extract contiguous BF16 K for this batch from bf16 paged blocks
            var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
            var k_b_host = List(length=k_b_size, fill=Scalar[ref_type](0))

            var page_base_b = 0
            for bi in range(b):
                page_base_b += ceildiv(
                    cache_lengths[bi] + q_max_seq_len, PAGE_SIZE
                )

            for tok in range(ref_num_keys):
                var page_idx, tok_in_page = divmod(tok, PAGE_SIZE)
                var physical_page = page_base_b + page_idx

                var src_offset = (
                    physical_page
                    * kv_dim2
                    * NUM_LAYERS
                    * PAGE_SIZE
                    * kv_params.num_heads
                    * kv_params.head_size
                    + tok_in_page * kv_params.num_heads * kv_params.head_size
                )
                var dst_offset = tok * KV_NUM_HEADS * DEPTH
                for d in range(KV_NUM_HEADS * DEPTH):
                    # Use BF16 (dequantized) version for reference
                    k_b_host[dst_offset + d] = blocks_bf16[src_offset + d]

            # BF16 Q for this batch: [1, 1, num_heads, depth]
            var q_b_size = 1 * num_heads * DEPTH
            var q_b_host = List(length=q_b_size, fill=Scalar[ref_type](0))
            for i in range(q_b_size):
                q_b_host[i] = q_bf16_host[b * num_heads * DEPTH + i]

            # Reference output: [1, 1, num_heads, depth] (full depth, BF16)
            var ref_b_size = 1 * num_heads * DEPTH
            var ref_b_host = List(length=ref_b_size, fill=Scalar[ref_type](0))

            # Copy to device
            var k_b_device = ctx.enqueue_create_buffer[ref_type](k_b_size)
            ctx.enqueue_copy(k_b_device, k_b_host)

            var q_b_device = ctx.enqueue_create_buffer[ref_type](q_b_size)
            ctx.enqueue_copy(q_b_device, q_b_host)

            var ref_b_device = ctx.enqueue_create_buffer[ref_type](ref_b_size)
            ctx.synchronize()

            # Build 4D TileTensors (all BF16 for reference)
            var q_b_tt = TileTensor(
                q_b_device,
                row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
            )
            var k_b_tt = TileTensor(
                k_b_device,
                row_major(
                    (
                        Idx[1],
                        ref_num_keys,
                        Idx[KV_NUM_HEADS],
                        Idx[DEPTH],
                    )
                ),
            )
            var ref_b_tt = TileTensor(
                ref_b_device,
                row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
            )

            # Run mha_gpu_naive with BF16 inputs
            mha_gpu_naive(
                q_b_tt,
                k_b_tt,
                k_b_tt,
                NullMask(),
                ref_b_tt,
                scale,
                1,  # batch_size
                1,  # seq_len
                ref_num_keys,
                num_heads,
                DEPTH,
                group,
                ctx,
            )

            ctx.synchronize()
            ctx.enqueue_copy(ref_b_host, ref_b_device)
            ctx.synchronize()

            # Compare first V_DEPTH=512 dims per head
            var out_offset = b * num_heads * V_DEPTH
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var expect = ref_b_host[d + DEPTH * h].cast[.float64]()
                    var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                        DType.float64
                    ]()
                    var abs_err = abs(actual - expect)
                    if abs_err > max_abs_err:
                        max_abs_err = abs_err
                    if abs_err > 2e-1:
                        print(b, h, d, actual, expect)
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

            total_checked += num_heads * V_DEPTH

            # Cleanup per-batch buffers
            _ = k_b_device
            _ = q_b_device
            _ = ref_b_device
            _ = ref_b_host^
            _ = q_b_host^
            _ = k_b_host^

        print(
            "  Verified:",
            total_checked,
            "elements, max_abs_err:",
            max_abs_err,
        )

        print("  PASS:", name, "\n")

        # Cleanup

        _ = blocks_device
        _ = cache_lengths_device
        _ = lookup_table_device
        _ = q_device
        _ = row_offsets_device
        _ = out_device
        _ = row_offsets_host^
        _ = lookup_table_host^
        _ = cache_lengths_host^
        _ = blocks_bf16^
        _ = blocks_host^
        _ = out_host^
        _ = q_bf16_host^
        _ = q_fp8_host^

    # ===-----------------------------------------------------------------------===#
    # Core test: native FP8 paged path with truly ragged Q (per-batch seq_len)
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    def run_test_paged_variable_ragged_q_native_fp8[
        mask_t: MHAMask,
        //,
        num_heads: Int,
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        seq_lens: List[Int],
        mask: mask_t,
        ctx: DeviceContext[],
    ) raises:
        """Native FP8 (Q=FP8, KV=FP8, output=BF16) MLA decode through the paged KV
        cache with per-batch variable Q sequence lengths (ragged Q).

        This is the AMD MTP fold exercised through the production paged + ragged
        layout path (the AMD fold is FP8-only and requires num_heads <= 16). Uniform
        `seq_lens` give a uniform multi-Q fold; differing `seq_lens` give the
        variable-draft-length case, which is supported: the comptime `q_seq_len`
        (== q_max_seq_len) is only the padding ceiling, while the runtime
        per-sequence length drives the SRD valid-row clamp, the per-token causal
        offset, the split-K stat writer, and the ragged reducer, so short sequences'
        pad rows are dropped. Pass `CausalMask()` to exercise that per-token causal
        offset (the discriminating case for variable Q); `NullMask()` covers the
        unmasked reduction.

        `mask` is applied to both the kernel and the per-batch GPU-naive reference,
        which uses BF16 Q and dequantized BF16 K with `num_keys = cache_len + this
        batch's seq_len` — so `mha_gpu_naive`'s `score_row = y + cache_len` matches
        the kernel's per-token `num_keys - seq_len + token`.
        """
        comptime q_type = DType.float8_e4m3fn
        comptime kv_type = DType.float8_e4m3fn
        comptime ref_type = DType.bfloat16

        var batch_size = len(cache_lengths)
        assert (
            len(seq_lens) == batch_size
        ), "cache_lengths and seq_lens must have same length"
        var q_max_seq_len = 0
        var total_q_tokens = 0
        for i in range(batch_size):
            if seq_lens[i] > q_max_seq_len:
                q_max_seq_len = seq_lens[i]
            total_q_tokens += seq_lens[i]

        print(
            t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
            t" q_type: {q_type}  kv_type: {kv_type}  q_max_seq_len:"
            t" {q_max_seq_len}  total_q_tokens: {total_q_tokens}"
        )
        for i in range(batch_size):
            print(
                t"  batch {i} : cache_len= {cache_lengths[i]}  q_seq_len="
                t" {seq_lens[i]}"
            )

        var max_cache_len = 0
        var total_pages = 0
        for i in range(batch_size):
            if cache_lengths[i] > max_cache_len:
                max_cache_len = cache_lengths[i]
            total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

        comptime scale = Float32(0.125)
        comptime group = num_heads

        # -----------------------------------------------------------------------
        # Step 1: Create the paged KV cache on host (FP8 + BF16 reference copy)
        # -----------------------------------------------------------------------
        comptime kv_params = KVCacheStaticParams(
            num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
        )
        comptime kv_dim2 = 1

        var block_elems = (
            total_pages
            * kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )

        var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
        var blocks_bf16 = List(length=block_elems, fill=Scalar[ref_type](0))
        randn(
            blocks_bf16,
            mean=0.0,
            standard_deviation=0.5,
        )
        for i in range(block_elems):
            blocks_host[i] = blocks_bf16[i].cast[kv_type]()

        var _page_stride = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        var _tok_stride = kv_params.num_heads * kv_params.head_size

        var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

        var max_num_keys_any_batch = 0
        for i in range(batch_size):
            var nk = cache_lengths[i] + seq_lens[i]
            if nk > max_num_keys_any_batch:
                max_num_keys_any_batch = nk
        var max_pages_per_batch = align_up(
            ceildiv(max_num_keys_any_batch, PAGE_SIZE), 8
        )
        var lut_size = batch_size * max_pages_per_batch
        var lookup_table_host = List(length=lut_size, fill=UInt32(0))

        var page_offset = 0
        for i in range(batch_size):
            var num_pages_i = ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)
            for p in range(num_pages_i):
                lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                    page_offset + p
                )
            page_offset += num_pages_i

        # Zero out tail slots in both FP8 and BF16 copies.
        var cur_page = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + seq_lens[bi]
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for pg in range(num_pages_i):
                var valid_toks = num_keys_i - pg * PAGE_SIZE
                if valid_toks > PAGE_SIZE:
                    valid_toks = PAGE_SIZE
                var base = cur_page * _page_stride + valid_toks * _tok_stride
                var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
                for z in range(zero_count):
                    blocks_host[base + z] = 0
                    blocks_bf16[base + z] = 0
                cur_page += 1

        # -----------------------------------------------------------------------
        # Step 2: Q tensor — generate BF16, keep copy, cast to FP8
        # -----------------------------------------------------------------------
        var q_size = total_q_tokens * num_heads * DEPTH
        var q_bf16_host = List(length=q_size, fill=Scalar[ref_type](0))
        randn(q_bf16_host, mean=0.0, standard_deviation=0.5)
        var q_fp8_host = List(length=q_size, fill=Scalar[q_type](0))
        for i in range(q_size):
            q_fp8_host[i] = q_bf16_host[i].cast[q_type]()

        # -----------------------------------------------------------------------
        # Step 3: input_row_offsets — truly ragged per batch
        # -----------------------------------------------------------------------
        var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
        row_offsets_host[0] = UInt32(0)
        for i in range(batch_size):
            row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

        # -----------------------------------------------------------------------
        # Step 4: Output tensor (BF16)
        # -----------------------------------------------------------------------
        var out_size = total_q_tokens * num_heads * V_DEPTH
        var out_host = List(length=out_size, fill=Scalar[ref_type](0))

        # -----------------------------------------------------------------------
        # Step 5: Copy everything to device
        # -----------------------------------------------------------------------
        var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
        ctx.enqueue_copy(blocks_device, blocks_host)

        var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
            batch_size
        )
        ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

        var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
        ctx.enqueue_copy(lookup_table_device, lookup_table_host)

        var q_device = ctx.enqueue_create_buffer[q_type](q_size)
        ctx.enqueue_copy(q_device, q_fp8_host)

        var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
            batch_size + 1
        )
        ctx.enqueue_copy(row_offsets_device, row_offsets_host)

        var out_device = ctx.enqueue_create_buffer[ref_type](out_size)

        ctx.synchronize()

        # -----------------------------------------------------------------------
        # Step 6: Build TileTensors and PagedKVCacheCollection on device
        # -----------------------------------------------------------------------
        var blocks_tt = TileTensor(
            blocks_device,
            row_major(
                Int64(total_pages),
                Idx[kv_dim2],
                Idx[NUM_LAYERS],
                Idx[PAGE_SIZE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        )
        var cache_lengths_tt = TileTensor(
            cache_lengths_device, row_major(Int64(batch_size))
        )
        var lookup_table_tt = TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(max_pages_per_batch)),
        )
        var kv_collection = PagedKVCacheCollection[
            kv_type,
            kv_params,
            PAGE_SIZE,
            MutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            MutAnyOrigin,
        ](
            blocks_tt.as_unsafe_any_origin(),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
        )

        var kv_cache = kv_collection.get_key_cache(0)

        var q_tt = TileTensor(
            q_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
        )

        var out_tt = TileTensor(
            out_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        )

        var row_offsets_tt = TileTensor(
            row_offsets_device,
            row_major(batch_size + 1),
        )

        # -----------------------------------------------------------------------
        # Step 7: Pre-compute scalar args and call the kernel
        # -----------------------------------------------------------------------
        var mla_args = Self.V.scalar_args[
            num_heads, is_fp8_kv=(kv_type == .float8_e4m3fn)
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        var scalar_args_buf_tt = TileTensor(
            mla_args.unsafe_ptr(), row_major[3]()
        )

        print("  Launching native FP8 MLA decode kernel (ragged Q)...")

        flare_mla_decoding[
            kernels=Self.V.Kernels,
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            mask,
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            q_max_seq_len=q_max_seq_len,
        )

        ctx.synchronize()
        print("  Kernel completed successfully (no crash).")

        # -----------------------------------------------------------------------
        # Step 8: Numerical verification using mha_gpu_naive reference (BF16)
        # -----------------------------------------------------------------------
        ctx.enqueue_copy(out_host, out_device)
        ctx.synchronize()

        print("  Computing GPU naive reference per batch and comparing...")

        var rtol = 8e-2
        var atol = 5e-1
        var total_checked = 0
        var max_abs_err = Float64(0)

        var q_token_offset = 0

        for b in range(batch_size):
            var cache_len = cache_lengths[b]
            var b_seq_len = seq_lens[b]
            var ref_num_keys = cache_len + b_seq_len

            # Extract contiguous BF16 K for this batch from bf16 paged blocks.
            var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
            var k_b_host = List(length=k_b_size, fill=Scalar[ref_type](0))

            var page_base_b = 0
            for bi in range(b):
                page_base_b += ceildiv(
                    cache_lengths[bi] + seq_lens[bi], PAGE_SIZE
                )

            for tok in range(ref_num_keys):
                var page_idx, tok_in_page = divmod(tok, PAGE_SIZE)
                var physical_page = page_base_b + page_idx

                var src_offset = (
                    physical_page
                    * kv_dim2
                    * NUM_LAYERS
                    * PAGE_SIZE
                    * kv_params.num_heads
                    * kv_params.head_size
                    + tok_in_page * kv_params.num_heads * kv_params.head_size
                )
                var dst_offset = tok * KV_NUM_HEADS * DEPTH
                for d in range(KV_NUM_HEADS * DEPTH):
                    k_b_host[dst_offset + d] = blocks_bf16[src_offset + d]

            # BF16 Q for this batch: [1, b_seq_len, num_heads, depth]
            var q_b_size = b_seq_len * num_heads * DEPTH
            var q_b_host = List(length=q_b_size, fill=Scalar[ref_type](0))
            var q_batch_start = q_token_offset * num_heads * DEPTH
            for i in range(q_b_size):
                q_b_host[i] = q_bf16_host[q_batch_start + i]

            var ref_b_size = b_seq_len * num_heads * DEPTH
            var ref_b_host = List(length=ref_b_size, fill=Scalar[ref_type](0))

            var k_b_device = ctx.enqueue_create_buffer[ref_type](k_b_size)
            ctx.enqueue_copy(k_b_device, k_b_host)

            var q_b_device = ctx.enqueue_create_buffer[ref_type](q_b_size)
            ctx.enqueue_copy(q_b_device, q_b_host)

            var ref_b_device = ctx.enqueue_create_buffer[ref_type](ref_b_size)
            ctx.synchronize()

            var q_b_tt = TileTensor(
                q_b_device,
                row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
            )
            var k_b_tt = TileTensor(
                k_b_device,
                row_major(
                    (
                        Idx[1],
                        ref_num_keys,
                        Idx[KV_NUM_HEADS],
                        Idx[DEPTH],
                    )
                ),
            )
            var ref_b_tt = TileTensor(
                ref_b_device,
                row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
            )

            mha_gpu_naive(
                q_b_tt,
                k_b_tt,
                k_b_tt,
                mask,
                ref_b_tt,
                scale,
                1,  # batch_size
                b_seq_len,  # this batch's seq_len
                ref_num_keys,  # cache_len + this batch's seq_len
                num_heads,
                DEPTH,
                group,
                ctx,
            )

            ctx.synchronize()
            ctx.enqueue_copy(ref_b_host, ref_b_device)
            ctx.synchronize()

            for s in range(b_seq_len):
                var out_offset = (q_token_offset + s) * num_heads * V_DEPTH
                var ref_s_offset = s * num_heads * DEPTH
                for h in range(num_heads):
                    for d in range(V_DEPTH):
                        var expect = ref_b_host[
                            ref_s_offset + d + DEPTH * h
                        ].cast[DType.float64]()
                        var actual = out_host[
                            out_offset + V_DEPTH * h + d
                        ].cast[DType.float64]()
                        var abs_err = abs(actual - expect)
                        if abs_err > max_abs_err:
                            max_abs_err = abs_err
                        if abs_err > 2e-1:
                            print(
                                t"batch {b} tok {s} head {h} dim {d} actual"
                                t" {actual} expect {expect}"
                            )
                        assert_almost_equal(
                            actual, expect, atol=atol, rtol=rtol
                        )

                total_checked += num_heads * V_DEPTH

            q_token_offset += b_seq_len

            _ = k_b_device
            _ = q_b_device
            _ = ref_b_device
            _ = ref_b_host^
            _ = q_b_host^
            _ = k_b_host^

        print(
            "  Verified:",
            total_checked,
            "elements, max_abs_err:",
            max_abs_err,
        )

        print("  PASS:", name, "\n")

        _ = blocks_device
        _ = cache_lengths_device
        _ = lookup_table_device
        _ = q_device
        _ = row_offsets_device
        _ = out_device
        _ = row_offsets_host^
        _ = lookup_table_host^
        _ = cache_lengths_host^
        _ = blocks_bf16^
        _ = blocks_host^
        _ = out_host^
        _ = q_bf16_host^
        _ = q_fp8_host^

    # ===-----------------------------------------------------------------------===#
    # Benchmark: native FP8 paged path (Q=FP8, KV=FP8, output=BF16)
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    def run_bench_paged_variable_native_fp8[
        num_heads: Int,
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        ctx: DeviceContext[],
    ) raises:
        """Benchmark the native FP8 paged path (no numerical verification)."""
        comptime q_type = DType.float8_e4m3fn
        comptime kv_type = DType.float8_e4m3fn
        comptime ref_type = DType.bfloat16

        var batch_size = len(cache_lengths)
        print(
            t"bench: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
            t" q_type: {q_type}  kv_type: {kv_type}  output_type: {ref_type}"
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
            total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

        comptime scale = Float32(0.125)

        # Step 1: Paged KV cache on host (FP8)
        comptime kv_params = KVCacheStaticParams(
            num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
        )
        comptime kv_dim2 = 1

        var block_elems = (
            total_pages
            * kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )

        var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
        var blocks_bf16 = List(length=block_elems, fill=Scalar[ref_type](0))
        randn(
            blocks_bf16,
            mean=0.0,
            standard_deviation=0.5,
        )
        for i in range(block_elems):
            blocks_host[i] = blocks_bf16[i].cast[kv_type]()

        var _page_stride = (
            kv_dim2
            * NUM_LAYERS
            * PAGE_SIZE
            * kv_params.num_heads
            * kv_params.head_size
        )
        var _tok_stride = kv_params.num_heads * kv_params.head_size

        var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

        # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
        # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
        var max_pages_per_batch = align_up(
            ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
        )
        var lut_size = batch_size * max_pages_per_batch
        var lookup_table_host = List(length=lut_size, fill=UInt32(0))

        var page_offset = 0
        for i in range(batch_size):
            var num_pages_i = ceildiv(
                cache_lengths[i] + q_max_seq_len, PAGE_SIZE
            )
            for p in range(num_pages_i):
                lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                    page_offset + p
                )
            page_offset += num_pages_i

        var cur_page = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + q_max_seq_len
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for pg in range(num_pages_i):
                var valid_toks = num_keys_i - pg * PAGE_SIZE
                if valid_toks > PAGE_SIZE:
                    valid_toks = PAGE_SIZE
                var base = cur_page * _page_stride + valid_toks * _tok_stride
                var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
                for z in range(zero_count):
                    blocks_host[base + z] = 0
                cur_page += 1

        # Step 2: Q tensor (FP8)
        var q_size = total_q_tokens * num_heads * DEPTH
        var q_bf16_tmp = List(length=q_size, fill=Scalar[ref_type](0))
        randn(
            q_bf16_tmp,
            mean=0.0,
            standard_deviation=0.5,
        )
        var q_host = List(length=q_size, fill=Scalar[q_type](0))
        for i in range(q_size):
            q_host[i] = q_bf16_tmp[i].cast[q_type]()

        # Step 3: input_row_offsets
        var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
        row_offsets_host[0] = UInt32(0)
        for i in range(batch_size):
            row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

        # Step 4: Output tensor (BF16)
        var out_size = total_q_tokens * num_heads * V_DEPTH
        var out_host = List(length=out_size, fill=Scalar[ref_type](0))

        # Step 5: Copy to device
        var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
        ctx.enqueue_copy(blocks_device, blocks_host)

        var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
            batch_size
        )
        ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

        var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
        ctx.enqueue_copy(lookup_table_device, lookup_table_host)

        var q_device = ctx.enqueue_create_buffer[q_type](q_size)
        ctx.enqueue_copy(q_device, q_host)

        var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
            batch_size + 1
        )
        ctx.enqueue_copy(row_offsets_device, row_offsets_host)

        var out_device = ctx.enqueue_create_buffer[ref_type](out_size)

        ctx.synchronize()

        # Step 6: Build TileTensors and PagedKVCacheCollection
        var blocks_tt = TileTensor(
            blocks_device,
            row_major(
                Int64(total_pages),
                Idx[kv_dim2],
                Idx[NUM_LAYERS],
                Idx[PAGE_SIZE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        )
        var cache_lengths_tt = TileTensor(
            cache_lengths_device, row_major(Int64(batch_size))
        )
        var lookup_table_tt = TileTensor(
            lookup_table_device,
            row_major(Int64(batch_size), Int64(max_pages_per_batch)),
        )
        var kv_collection = PagedKVCacheCollection[
            kv_type,
            kv_params,
            PAGE_SIZE,
            MutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            MutAnyOrigin,
        ](
            blocks_tt.as_unsafe_any_origin(),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
        )

        var kv_cache = kv_collection.get_key_cache(0)

        var q_tt = TileTensor(
            q_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
        )

        var out_tt = TileTensor(
            out_device,
            row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
        )

        var row_offsets_tt = TileTensor(
            row_offsets_device,
            row_major(batch_size + 1),
        )

        # Step 7: Pre-compute scalar args and benchmark
        var mla_args = Self.V.scalar_args[
            num_heads, is_fp8_kv=(kv_type == .float8_e4m3fn)
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        var scalar_args_buf_tt = TileTensor(
            mla_args.unsafe_ptr(), row_major[3]()
        )

        @inline(.always)
        def kernel_launch(
            ctx: DeviceContext[],
        ) raises {
            var out_tt,
            var q_tt,
            var kv_cache,
            var row_offsets_tt,
            var scalar_args_buf_tt,
            imm,
        }:
            flare_mla_decoding[
                kernels=Self.V.Kernels,
                rank=3,
                config=MHAConfig[q_type](num_heads, DEPTH),
                ragged=True,
            ](
                out_tt,
                q_tt,
                kv_cache,
                NullMask(),
                row_offsets_tt,
                scale,
                ctx,
                scalar_args_buf_tt,
                q_max_seq_len=q_max_seq_len,
            )

        comptime nrun = 200

        # Warmup
        kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time(kernel_launch, nrun)) / Float64(
            nrun
        )
        var mstime = nstime / 1000000
        print(t"   {nrun} runs avg {mstime} ms")
        print()

        # Cleanup

        _ = blocks_device
        _ = cache_lengths_device
        _ = lookup_table_device
        _ = q_device
        _ = row_offsets_device
        _ = out_device
        _ = row_offsets_host^
        _ = lookup_table_host^
        _ = cache_lengths_host^
        _ = blocks_host^
        _ = out_host^
        _ = q_host^
        _ = q_bf16_tmp^
        _ = blocks_bf16^

    # ===-----------------------------------------------------------------------===#
    # Helper functions to reduce duplication in main()
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    def run_both_kv_types[
        num_heads: Int
    ](
        name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext[]
    ) raises:
        """Run correctness test for both bf16 and fp8 KV types."""
        Self.run_test_paged_variable[.bfloat16, DType.bfloat16, num_heads](
            name + "_bf16", cache_lengths, ctx
        )
        Self.run_test_paged_variable[.bfloat16, DType.float8_e4m3fn, num_heads](
            name + "_fp8", cache_lengths, ctx
        )

    @staticmethod
    def run_uniform_both[
        num_heads: Int
    ](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext[]) raises:
        """Run correctness test with uniform cache lengths for both KV types."""
        Self.run_both_kv_types[num_heads](name, make_uniform(count, value), ctx)

    @staticmethod
    def run_bench_both_kv_types[
        num_heads: Int
    ](
        name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext[]
    ) raises:
        """Run benchmark for both bf16 and fp8 KV types."""
        Self.run_bench_paged_variable[.bfloat16, DType.bfloat16, num_heads](
            name + "_bf16", cache_lengths, ctx
        )
        Self.run_bench_paged_variable[
            .bfloat16, DType.float8_e4m3fn, num_heads
        ](name + "_fp8", cache_lengths, ctx)

    @staticmethod
    def run_bench_uniform_both[
        num_heads: Int
    ](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext[]) raises:
        """Run benchmark with uniform cache lengths for both KV types."""
        Self.run_bench_both_kv_types[num_heads](
            name, make_uniform(count, value), ctx
        )

    @staticmethod
    def run_multiq_both_kv_types[
        num_heads: Int
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        q_max_seq_len: Int,
        ctx: DeviceContext[],
    ) raises:
        """Run multi-Q correctness test for both bf16 and fp8 KV types."""
        Self.run_test_paged_variable_multiq[
            .bfloat16, DType.bfloat16, num_heads
        ](name + "_bf16", cache_lengths, q_max_seq_len, ctx)
        Self.run_test_paged_variable_multiq[
            DType.bfloat16, DType.float8_e4m3fn, num_heads
        ](name + "_fp8", cache_lengths, q_max_seq_len, ctx)

    @staticmethod
    def run_multiq_uniform_both[
        num_heads: Int
    ](
        name: StringLiteral,
        count: Int,
        value: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext[],
    ) raises:
        """Run multi-Q test with uniform cache lengths for both KV types."""
        Self.run_multiq_both_kv_types[num_heads](
            name, make_uniform(count, value), q_max_seq_len, ctx
        )

    @staticmethod
    def run_ragged_q_both_kv_types[
        num_heads: Int
    ](
        name: StringLiteral,
        cache_lengths: List[Int],
        seq_lens: List[Int],
        ctx: DeviceContext[],
    ) raises:
        """Run ragged-Q correctness test for both bf16 and fp8 KV types."""
        Self.run_test_paged_variable_ragged_q[
            .bfloat16, DType.bfloat16, num_heads
        ](name + "_bf16", cache_lengths, seq_lens, ctx)
        Self.run_test_paged_variable_ragged_q[
            DType.bfloat16, DType.float8_e4m3fn, num_heads
        ](name + "_fp8", cache_lengths, seq_lens, ctx)

    @staticmethod
    def run_all_three_kv_types[
        num_heads: Int
    ](
        name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext[]
    ) raises:
        """Run correctness test for bf16, fp8 (converter), and native fp8."""
        Self.run_both_kv_types[num_heads](name, cache_lengths, ctx)
        Self.run_test_paged_variable_native_fp8[num_heads](
            name + "_native_fp8", cache_lengths, ctx
        )

    @staticmethod
    def run_uniform_all_three[
        num_heads: Int
    ](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext[]) raises:
        """Run correctness test with uniform cache lengths for all three KV types.
        """
        Self.run_all_three_kv_types[num_heads](
            name, make_uniform(count, value), ctx
        )

    @staticmethod
    def run_bench_all_three_kv_types[
        num_heads: Int
    ](
        name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext[]
    ) raises:
        """Run benchmark for bf16, fp8 (converter), and native fp8."""
        Self.run_bench_both_kv_types[num_heads](name, cache_lengths, ctx)
        Self.run_bench_paged_variable_native_fp8[num_heads](
            name + "_native_fp8", cache_lengths, ctx
        )

    @staticmethod
    def run_bench_uniform_all_three[
        num_heads: Int
    ](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext[]) raises:
        """Run benchmark with uniform cache lengths for all three KV types."""
        Self.run_bench_all_three_kv_types[num_heads](
            name, make_uniform(count, value), ctx
        )

    # ===-----------------------------------------------------------------------===#
    # Entry point
    # ===-----------------------------------------------------------------------===#
