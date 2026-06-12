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

"""Regression test for PAQ-2333: GPU attention kernel hang with inflight
batching on Gemma3 27B.

The hang is a TMA transaction barrier deadlock (SYNCS.PHASECHK.TRANS64.TRYWAIT)
in the SM100 FA4 attention kernel when processing a mixed TG+CE batch. Certain
combinations of seq_lens and cache_lens trigger the deadlock.
"""

from std.collections import Set
from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int
from layout._utils import ManagedLayoutTensor
from std.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache_test_utils import assert_no_nan_inf, padded_lut_cols
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask
from std.utils import IndexList


def test_paged_ragged_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    num_paged_blocks: Int,
    ctx: DeviceContext,
) raises:
    comptime page_size = get_defined_int["page_size", 256]()
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

    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()

    var row_offsets_shape = IndexList[1](batch_size + 1)
    var cache_lengths_shape = IndexList[1](batch_size)
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )
    var output_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )

    var row_offsets_rl = RuntimeLayout[row_offsets_layout].row_major(
        row_offsets_shape
    )
    var cache_lengths_rl = RuntimeLayout[cache_lengths_layout].row_major(
        cache_lengths_shape
    )
    var q_ragged_rl = RuntimeLayout[q_ragged_layout].row_major(q_ragged_shape)
    var output_rl = RuntimeLayout[output_layout].row_major(output_shape)

    var input_row_offsets = ManagedLayoutTensor[
        DType.uint32, row_offsets_layout
    ](row_offsets_rl, ctx)
    var cache_lengths_managed = ManagedLayoutTensor[
        DType.uint32, cache_lengths_layout
    ](cache_lengths_rl, ctx)
    var q_ragged = ManagedLayoutTensor[dtype, q_ragged_layout](q_ragged_rl, ctx)
    var test_output = ManagedLayoutTensor[dtype, output_layout](output_rl, ctx)

    var input_row_offsets_host = input_row_offsets.tensor[update=False]()
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets_host[batch_size] = running_offset

    var q_ragged_tensor = q_ragged.tensor()
    random(q_ragged_tensor)

    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    # Pad LUT inner dim to honor `PagedKVCache.populate`'s SIMD padding
    # invariant — see `padded_lut_cols`.
    var paged_lut_shape = IndexList[2](
        batch_size,
        padded_lut_cols(ceildiv(max_full_context_length, page_size)),
    )

    var kv_block_paged_rl = RuntimeLayout[kv_block_6d_layout].row_major(
        kv_block_paged_shape
    )
    var paged_lut_rl = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var kv_block_paged = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_rl, ctx
    )
    var paged_lut = ManagedLayoutTensor[DType.uint32, paged_lut_layout](
        paged_lut_rl, ctx
    )

    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged.tensor[update=False]().ptr,
        kv_block_paged_rl,
    )
    random(kv_block_paged_tensor)

    var paged_lut_tensor = paged_lut.tensor[update=False]()
    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            paged_lut_set.add(randval)
            paged_lut_tensor[bs, block_idx] = UInt32(randval)

    var kv_block_paged_lt = kv_block_paged.device_tensor()
    var cache_lengths_lt = cache_lengths_managed.device_tensor()
    var paged_lut_lt = paged_lut.device_tensor()

    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        kv_block_paged_lt,
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var q_ragged_lt = q_ragged.device_tensor()
    var test_output_lt = test_output.device_tensor()

    var ce_count = 0
    var tg_count = 0
    for i in range(batch_size):
        if valid_lengths[i] == 1:
            tg_count += 1
        else:
            ce_count += 1

    print(
        "Running: batch_size=",
        batch_size,
        "total_tokens=",
        total_length,
        "CE=",
        ce_count,
        "TG=",
        tg_count,
        "max_prompt_len=",
        max_prompt_length,
        "max_context_len=",
        max_full_context_length,
        "num_pages=",
        num_paged_blocks,
    )

    flash_attention[ragged=True](
        test_output_lt,
        q_ragged_lt,
        kv_collection.get_key_cache(layer_idx),
        kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.synchronize()
    assert_no_nan_inf(test_output, "gemma3_hang_output")
    print("  -> OK")


def main() raises:
    seed(42)

    # Gemma3 27B config: 32 Q heads, 16 KV heads, head_dim=128
    comptime kv_params = KVCacheStaticParams(num_heads=16, head_size=128)
    comptime num_q_heads = 32
    comptime num_layers = 2
    comptime layer_idx = 1

    with DeviceContext() as ctx:
        # Mixed TG+CE batch with randomized shapes that triggers TMA deadlock.
        # 230 TG requests (seq_len=1) with cache_lens 100-1000
        # 40 CE requests with seq_lens 50-1000
        # 647 paged KV cache blocks (matching Gemma3 27B server config)
        print("=== PAQ-2333 repro: 230 TG + 40 CE, 647 pages, seed=42 ===")
        var active_lens = List[Int]()
        var cache_lens = List[Int]()
        for _ in range(230):
            active_lens.append(1)
            cache_lens.append(Int(random_ui64(100, 1000)))
        for _ in range(40):
            active_lens.append(Int(random_ui64(50, 1000)))
            cache_lens.append(0)

        test_paged_ragged_attention[num_q_heads, DType.bfloat16, kv_params](
            active_lens, cache_lens, num_layers, layer_idx, 647, ctx
        )
