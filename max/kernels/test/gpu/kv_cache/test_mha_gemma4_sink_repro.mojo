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
"""Minimal repro for LLVM AMDGPU register-allocator assertion on gfx950.

Reproduces the `IntervalMap.h:639 "Overlapping insert"` assertion hit by
`//max/tests/integration/architectures/gemma4:test_attention::test_attention_global`.

Crashing combination (bisected to commit 97e211bb0 "MHA decode through
amd_structured/"):

  - gfx950 structured prefill path
  - depth = 512 (global Gemma4 head_dim)
  - CausalMask
  - bfloat16
  - sink = True

The gemma4 test never passes `sink_weights` at runtime, but MOGG's
`unswitch[call_flash_attention](Bool(sink_weights))` forces comptime
specialization of both `sink=True` and `sink=False` kernels, and the
`sink=True` instantiation is what trips the LLVM assertion during
`AMDGPURewriteAGPRCopyMFMA::eliminateSpillsOfReassignedVGPRs`.

The assertion only fires under LLVM builds with asserts enabled (bazel
`k8-dbg`). Release-LLVM builds (direct `mojo` CLI) silently produce code
that may or may not be correct — run this through `./bazelw test`.
"""

from std.math import ceildiv, rsqrt
from std.random import seed
from max.gpu.host import DeviceContext

from kv_cache_test_utils import random_distinct
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Coord, Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import (
    MHADecodeDispatchMetadata,
    flash_attention,
)
from nn.kv_cache_ragged import generic_flash_attention_kv_cache_ragged
from nn.attention.mha_mask import CausalMask


def execute_sink_prefill_repro[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    comptime page_size = 256

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

    var q_layout = row_major(
        total_length, Idx[num_q_heads], Idx[kv_params.head_size]
    )

    var input_row_offsets = HostDeviceTileTensor[.uint32](
        row_major(Coord(Int64(batch_size + 1))), ctx
    )
    var cache_lengths_managed = HostDeviceTileTensor[.uint32](
        row_major(Coord(Int64(batch_size))), ctx
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

    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size + 2
    )

    var kv_block_paged = HostDeviceTileTensor[dtype](
        row_major(
            Coord(
                Int64(num_paged_blocks),
                Idx[2],
                Int64(num_layers),
                Idx[page_size],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            )
        ),
        ctx,
    )
    var paged_lut = HostDeviceTileTensor[.uint32](
        row_major(
            Coord(
                Int64(batch_size),
                Int64(ceildiv(max_full_context_length, page_size)),
            )
        ),
        ctx,
    )

    random(kv_block_paged.host_tensor())
    kv_block_paged.to_device()

    var paged_lut_tensor = paged_lut.host_tensor()
    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. Total pages needed is
    # <= num_paged_blocks by construction.
    var total_pages = 0
    for bs in range(batch_size):
        total_pages += ceildiv(cache_lengths[bs] + valid_lengths[bs], page_size)
    var paged_blocks = random_distinct(num_paged_blocks, total_pages)
    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(0, ceildiv(seq_len, page_size)):
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
    # The collection spells its block strides symbolically in `kv_params`,
    # which the compiler cannot fold against `row_major`'s; the two layouts
    # are structurally identical.
    var kv_collection_paged_device = Collection(
        rebind[Collection.blocks_tt_type](
            kv_block_paged.device_tensor().as_unsafe_any_origin()
        ),
        cache_lengths_managed.device_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    # Match the dispatch_metadata that MOGG assembles from the graph inputs
    # in `_unmarshal_mha_decode_dispatch_metadata` — hard-coded for the
    # gemma4 prefill case.
    var decode_dispatch_metadata = MHADecodeDispatchMetadata(
        batch_size,
        max_prompt_length,
        0,
        max_full_context_length,
    )

    # Route through the exact MOGG-level entry point used by
    # `mo.mha.ragged.paged`. `_flash_attention_dispatch` inside runs
    # `unswitch[call_flash_attention](Bool(sink_weights))`, forcing BOTH
    # sink=True and sink=False specializations to compile — which is the
    # distinguishing property of the gemma4 graph compile.
    var ctx_ptr = ctx
    generic_flash_attention_kv_cache_ragged[
        target="gpu",
        mask_str="causal",
    ](
        q_ragged.device_tensor(),
        input_row_offsets.device_tensor(),
        kv_collection_paged_device,
        UInt32(layer_idx),
        rsqrt(Float32(kv_params.head_size)),
        test_output.device_tensor(),
        ctx_ptr,
        decode_dispatch_metadata,
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        seed(42)

        # Gemma4 global shape: num_q_heads=32, num_kv_heads=4, head_dim=512,
        # seq_len=11 prefill, CausalMask, bf16, paged, ragged — this is the
        # exact specialization that crashes during AMDGPURewriteAGPRCopyMFMA.
        print("Gemma4 global sink=True prefill repro")
        var seq_lens: List = [11]
        var cache_sizes: List = [0]
        execute_sink_prefill_repro[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=4, head_size=512),
        ](seq_lens, cache_sizes, 2, 0, ctx)

        print("PASS")
