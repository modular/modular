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
# Test mha_decoding (the batched decode kernel) via the KV cache
# flash_attention overload, which sets is_token_generation=True and
# dispatches to mha_decoding. Uses hash comparison for bitwise
# reproducibility on AMD MI355.
#
# This test is separate from test_batch_kv_cache_flash_attention_*
# because that test compares continuous-vs-paged (both go through the
# same mha_decoding kernel, so they can't detect bugs in mha_decoding
# itself). This test compares against known-good hashes from main.

from std.math import rsqrt
from std.memory import bitcast
from std.random import seed
from std.collections import Set

from std.gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from std.memory import memcpy, memset_zero
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask
from std.sys import has_amd_gpu_accelerator
from std.testing import assert_true

from std.utils import IndexList


def compute_hash[
    type: DType
](ptr: UnsafePointer[Scalar[type], _], size: Int) -> UInt64:
    var h: UInt64 = 14695981039346656037
    for i in range(size):
        var val = ptr[i].cast[DType.float32]()
        var bits = bitcast[DType.uint32, 1](val)
        h ^= bits.cast[DType.uint64]()
        h *= 1099511628211
    return h


def test_decode_kv_cache[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    dtype: DType,
](
    cache_lengths: List[Int],
    ctx: DeviceContext,
    expected_hash: UInt64 = 0,
) raises:
    """Test mha_decoding by calling the KV cache flash_attention overload
    with max_prompt_length=1 (triggering is_token_generation=True).

    Compares paged KV cache output against continuous KV cache output
    AND optionally against a known-good hash from main.
    """
    comptime page_size = 256
    var batch_size = len(cache_lengths)
    var num_layers = 2
    var layer_idx = 1

    # TG: all seq_len=1
    var valid_lengths = List[Int]()
    for _ in range(batch_size):
        valid_lengths.append(1)

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    # Layouts
    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime kv_block_6d_layout = Layout.row_major[6]()
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime lookup_table_layout = Layout(UNKNOWN_VALUE)

    # Row offsets
    var row_offsets = ManagedLayoutTensor[DType.uint32, row_offsets_layout](
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
        ctx,
    )
    var row_offsets_host = row_offsets.tensor[update=False]()
    var running_offset: UInt32 = 0
    for i in range(batch_size):
        row_offsets_host[i] = running_offset
        running_offset += UInt32(valid_lengths[i])
    row_offsets_host[batch_size] = running_offset

    # Cache lengths
    var cache_lens = ManagedLayoutTensor[DType.uint32, cache_lengths_layout](
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lens_host = cache_lens.tensor[update=False]()
    for i in range(batch_size):
        cache_lens_host[i] = UInt32(cache_lengths[i])

    # Q (random, ragged)
    var q = ManagedLayoutTensor[dtype, q_layout](
        RuntimeLayout[q_layout].row_major(
            IndexList[3](total_length, num_q_heads, kv_params.head_size)
        ),
        ctx,
    )
    random(q.tensor[update=False]())

    # Output
    var output = ManagedLayoutTensor[dtype, q_layout](
        RuntimeLayout[q_layout].row_major(
            IndexList[3](total_length, num_q_heads, kv_params.head_size)
        ),
        ctx,
    )

    # Continuous KV blocks
    var num_continuous_blocks = batch_size + 2
    var kv_block_continuous = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        RuntimeLayout[kv_block_6d_layout].row_major(
            IndexList[6](
                num_continuous_blocks,
                2,
                num_layers,
                max_full_context_length,
                kv_params.num_heads,
                kv_params.head_size,
            )
        ),
        ctx,
    )
    random(kv_block_continuous.tensor[update=False]())

    # Lookup table for continuous batching
    var lookup_table = ManagedLayoutTensor[DType.uint32, lookup_table_layout](
        RuntimeLayout[lookup_table_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var lookup_host = lookup_table.tensor[update=False]()
    for i in range(batch_size):
        lookup_host[i] = UInt32(i)

    # Build continuous collection
    var kv_continuous = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        kv_block_continuous.device_tensor(),
        cache_lens.device_tensor(),
        lookup_table.device_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var scale = rsqrt(Float32(kv_params.head_size))

    # Run flash_attention with continuous KV cache
    flash_attention[ragged=True](
        output.device_tensor(),
        q.device_tensor(),
        kv_continuous.get_key_cache(layer_idx),
        kv_continuous.get_value_cache(layer_idx),
        CausalMask(),
        row_offsets.device_tensor(),
        scale,
        ctx,
    )

    var out_host = output.tensor()
    var actual_hash = compute_hash(
        out_host.ptr,
        total_length * num_q_heads * kv_params.head_size,
    )
    print("HASH:", actual_hash)

    if expected_hash != 0:
        if actual_hash != expected_hash:
            print("HASH MISMATCH: expected", expected_hash, "got", actual_hash)
            raise Error("Hash mismatch for mha_decoding output")
        else:
            print("HASH OK")

    _ = row_offsets^
    _ = cache_lens^
    _ = kv_block_continuous^
    _ = lookup_table^


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        # Hash values are AMD MI355-specific (different GPUs produce
        # different MMA results). Only check hashes on AMD.
        comptime amd = has_amd_gpu_accelerator()
        comptime group4_hash = UInt64(9912283832381023013) if amd else UInt64(0)
        comptime group16_small_hash = UInt64(
            13849357457032651557
        ) if amd else UInt64(0)
        comptime group16_large_hash = UInt64(
            16395116742607741733
        ) if amd else UInt64(0)

        # group=4 (baseline)
        print("TG group=4 depth=128 bs=4")
        test_decode_kv_cache[
            32,
            KVCacheStaticParams(num_heads=8, head_size=128),
            DType.bfloat16,
        ]([500, 700, 200, 300], ctx, expected_hash=group4_hash)

        # group=16 small caches (single partition, no split-k)
        print("TG group=16 bs=4 small caches (no split-k)")
        test_decode_kv_cache[
            16,
            KVCacheStaticParams(num_heads=1, head_size=128),
            DType.bfloat16,
        ]([50, 100, 150, 200], ctx, expected_hash=group16_small_hash)

        # group=16 large caches (split-k)
        print("TG group=16 bs=4 large caches (split-k)")
        test_decode_kv_cache[
            16,
            KVCacheStaticParams(num_heads=1, head_size=128),
            DType.bfloat16,
        ]([500, 700, 200, 300], ctx, expected_hash=group16_large_hash)
