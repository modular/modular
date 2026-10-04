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

Production hit a TMA transaction barrier deadlock
(SYNCS.PHASECHK.TRANS64.TRYWAIT) in the SM100 FA4 attention kernel on a mixed
TG+CE batch. This file drives that batch -- 230 TG (seq_len 1, cache 100-1000)
plus 40 CE (seq_len 50-1000) over 270 requests -- and gates on the kernel both
returning and producing no NaN/Inf.

It does not, however, pin the deadlock. As first committed the batch
oversubscribed its own KV block pool, so it died in host setup before any
launch and never reached FA4 at any page size (see `num_blocks` below). A pass
here therefore means "this shape mix completes", not "the TMA deadlock class is
fixed".
"""

from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int
from max.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Coord, Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from kv_cache_test_utils import (
    assert_no_nan_inf,
    padded_lut_cols,
    random_distinct,
)
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask


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
    # Pages the batch needs: one per `page_size` keys per request, rounded up
    # per request (pages are never shared between requests here).
    var total_pages = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + valid_lengths[i], page_size)

    # `num_paged_blocks` is the production pool size the repro quotes, but the
    # randomized batch is not sized against it: per-request round-up costs up
    # to `page_size - 1` keys per request, and at 270 requests that overshoots
    # the pool at every page size in the sweep (695 / 1275 / 2406 blocks needed
    # vs 647).
    #
    # The LUT below hands out DISTINCT blocks, so an undersized pool is
    # unsatisfiable, and it fails LOUDLY but in the wrong place:
    # `random_distinct(n, k)` takes the `k`-prefix of `randperm(n)`, and
    # `List.shrink` calls `abort()` when `k > n`. So the repro dies in host
    # setup before any launch -- which is exactly what happened here for its
    # whole history, turning a kernel hang repro into a host-side failure that
    # still looked like a timeout. Grow the pool to what the batch needs.
    #
    # `total_pages` is the EXACT requirement, not a bound: it is accumulated by
    # the loop above with the same `ceildiv(cache + valid, page_size)`
    # expression the LUT loop below consumes, so `k <= n` holds by
    # construction. Sizing from `max_full_context_length` instead (as
    # `test_batch_kv_cache_flash_attention_causal_mask_ragged_paged.mojo` does)
    # is also correct but ~1.7x looser here, and being page-size-invariant it
    # would hide the fact that the requirement GROWS as `page_size` shrinks --
    # which is the reason this repro never ran.
    var num_blocks = max(num_paged_blocks, total_pages)

    var q_layout = row_major(
        total_length, Idx[num_q_heads], Idx[kv_params.head_size]
    )

    var input_row_offsets = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size + 1)), ctx
    )
    var cache_lengths_managed = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    var q_ragged = HostDeviceTileTensor[dtype](q_layout, ctx)
    var test_output = HostDeviceTileTensor[dtype](q_layout, ctx)

    var input_row_offsets_host = input_row_offsets.host_tensor()
    var cache_lengths_host = cache_lengths_managed.host_tensor()

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets_host[batch_size] = running_offset
    input_row_offsets.to_device()
    cache_lengths_managed.to_device()

    random(q_ragged.host_tensor())
    q_ragged.to_device()

    var kv_block_paged = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
        ctx,
    )
    # Pad LUT inner dim to honor `PagedKVCache.populate`'s SIMD padding
    # invariant — see `padded_lut_cols`.
    var paged_lut = HostDeviceTileTensor[.uint32](
        row_major(
            Int64(batch_size),
            Int64(padded_lut_cols(ceildiv(max_full_context_length, page_size))),
        ),
        ctx,
    )

    random(kv_block_paged.host_tensor())
    kv_block_paged.to_device()

    var paged_lut_tensor = paged_lut.host_tensor()
    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. `num_blocks >= total_pages`
    # by construction above.
    var paged_blocks = random_distinct(num_blocks, total_pages)
    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(ceildiv(seq_len, page_size)):
            paged_lut_tensor[bs, block_idx] = UInt32(paged_blocks[page_pos])
            page_pos += 1
    paged_lut.to_device()

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        kv_block_paged.device_tensor().as_unsafe_any_origin(),
        cache_lengths_managed.device_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

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
        num_blocks,
        "pages_needed=",
        total_pages,
    )

    flash_attention[ragged=True](
        test_output.device_tensor(),
        q_ragged.device_tensor(),
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
        #
        # 647 is the Gemma3 27B server pool size this repro quotes, but it is
        # only a floor: the batch needs more than that at every swept page size,
        # so the pool grows to what it actually needs. The run line below prints
        # the pool that was allocated (`num_pages=`) and the requirement it was
        # sized from (`pages_needed=`).
        print("=== PAQ-2333 repro: 230 TG + 40 CE, seed=42 ===")
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
