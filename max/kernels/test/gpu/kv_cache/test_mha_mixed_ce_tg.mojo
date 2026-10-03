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

from std.math import rsqrt

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from std.memory import unsafe_memcpy
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable


def execute_ragged_flash_attention[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    type: DType,
](ctx: DeviceContext,) raises:
    comptime num_paged_blocks = 32
    comptime page_size = 128
    comptime PagedCollectionType = PagedKVCacheCollection[
        type,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var num_layers = 1
    var layer_idx = 0

    var true_ce_prompt_lens: List = [100, 200, 300, 400]
    var mixed_ce_prompt_lens: List = [50, 100, 150, 100]

    var true_ce_cache_lens: List = [0, 0, 0, 0]
    var mixed_ce_cache_lens: List = [50, 100, 150, 300]

    var batch_size = len(true_ce_prompt_lens)

    var true_ce_cache_lengths_table = CacheLengthsTable.build(
        true_ce_prompt_lens, true_ce_cache_lens, ctx
    )
    var mixed_ce_cache_lengths_table = CacheLengthsTable.build(
        mixed_ce_prompt_lens, mixed_ce_cache_lens, ctx
    )

    var true_ce_total_length = true_ce_cache_lengths_table.total_length
    var mixed_ce_total_length = mixed_ce_cache_lengths_table.total_length
    var true_ce_max_full_context_length = (
        true_ce_cache_lengths_table.max_full_context_length
    )
    var mixed_ce_max_full_context_length = (
        mixed_ce_cache_lengths_table.max_full_context_length
    )
    var true_ce_max_prompt_length = (
        true_ce_cache_lengths_table.max_seq_length_batch
    )
    var mixed_ce_max_prompt_length = (
        mixed_ce_cache_lengths_table.max_seq_length_batch
    )

    # Q ragged tensors
    var true_ce_q_layout = row_major(
        true_ce_total_length, Idx[num_q_heads], Idx[kv_params.head_size]
    )
    var mixed_ce_q_layout = row_major(
        mixed_ce_total_length, Idx[num_q_heads], Idx[kv_params.head_size]
    )

    var true_ce_q_ragged = HostDeviceTileTensor[type](true_ce_q_layout, ctx)
    var true_ce_q_ragged_host = true_ce_q_ragged.host_tensor()
    random(true_ce_q_ragged_host)

    var mixed_ce_q_ragged = HostDeviceTileTensor[type](mixed_ce_q_layout, ctx)
    var mixed_ce_q_ragged_host = mixed_ce_q_ragged.host_tensor()

    var true_ce_row_offsets_host_ptr = (
        true_ce_cache_lengths_table.input_row_offsets.host_ptr
    )
    var mixed_ce_row_offsets_host_ptr = (
        mixed_ce_cache_lengths_table.input_row_offsets.host_ptr
    )

    var head_stride = num_q_heads * kv_params.head_size
    for bs_idx in range(batch_size):
        var mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        var true_ce_row_offset = Int(true_ce_row_offsets_host_ptr[bs_idx])
        var mixed_ce_row_offset = Int(mixed_ce_row_offsets_host_ptr[bs_idx])

        var mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        var true_ce_offset = (
            true_ce_q_ragged_host.unsafe_ptr()
            + (true_ce_row_offset + mixed_ce_cache_len) * head_stride
        )
        var mixed_ce_offset = (
            mixed_ce_q_ragged_host.unsafe_ptr()
            + mixed_ce_row_offset * head_stride
        )

        unsafe_memcpy(
            dest=mixed_ce_offset,
            src=true_ce_offset,
            count=mixed_ce_prompt_len * head_stride,
        )

    true_ce_q_ragged.to_device()
    mixed_ce_q_ragged.to_device()

    # Initialize output buffers
    var mixed_ce_output = HostDeviceTileTensor[type](mixed_ce_q_layout, ctx)
    var true_ce_output = HostDeviceTileTensor[type](true_ce_q_layout, ctx)

    # Initialize KVCache
    var kv_block_paged = HostDeviceTileTensor[type](
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
    random(kv_block_paged.host_tensor())
    kv_block_paged.to_device()

    var paged_lut = PagedLookupTable[page_size].build(
        true_ce_prompt_lens,
        true_ce_cache_lens,
        true_ce_max_full_context_length,
        num_paged_blocks,
        ctx,
    )

    # The collection spells its block strides symbolically in `kv_params`,
    # which the compiler cannot fold against `row_major`'s; the two layouts
    # are structurally identical.
    var kv_blocks_device = rebind[PagedCollectionType.blocks_tt_type](
        kv_block_paged.device_tensor().as_unsafe_any_origin()
    )
    var true_ce_kv_collection_device = PagedCollectionType(
        kv_blocks_device,
        true_ce_cache_lengths_table.cache_lengths.device_tile_tensor(),
        paged_lut.device_tile_tensor(),
        UInt32(true_ce_max_prompt_length),
        UInt32(true_ce_max_full_context_length),
    )

    var mixed_ce_kv_collection_device = PagedCollectionType(
        kv_blocks_device,
        mixed_ce_cache_lengths_table.cache_lengths.device_tile_tensor(),
        paged_lut.device_tile_tensor(),
        UInt32(mixed_ce_max_prompt_length),
        UInt32(mixed_ce_max_full_context_length),
    )

    # "true CE" execution
    print("true")
    flash_attention[ragged=True](
        true_ce_output.device_tensor(),
        true_ce_q_ragged.device_tensor(),
        true_ce_kv_collection_device.get_key_cache(layer_idx),
        true_ce_kv_collection_device.get_value_cache(layer_idx),
        CausalMask(),
        true_ce_cache_lengths_table.input_row_offsets.device_tile_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )

    # "mixed CE" execution
    print("mixed")
    flash_attention[ragged=True](
        mixed_ce_output.device_tensor(),
        mixed_ce_q_ragged.device_tensor(),
        mixed_ce_kv_collection_device.get_key_cache(layer_idx),
        mixed_ce_kv_collection_device.get_value_cache(layer_idx),
        CausalMask(),
        mixed_ce_cache_lengths_table.input_row_offsets.device_tile_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    mixed_ce_output.to_host()
    true_ce_output.to_host()
    var mixed_ce_output_host = mixed_ce_output.host_tensor()
    var true_ce_output_host = true_ce_output.host_tensor()

    for bs in range(batch_size):
        var mixed_ce_prompt_len = mixed_ce_prompt_lens[bs]
        var mixed_ce_row_offset = Int(mixed_ce_row_offsets_host_ptr[bs])
        var true_ce_row_offset = Int(true_ce_row_offsets_host_ptr[bs])
        var mixed_ce_cache_len = mixed_ce_cache_lens[bs]

        var true_ce_ragged_offset = true_ce_row_offset + mixed_ce_cache_len
        var mixed_ce_ragged_offset = mixed_ce_row_offset
        for s in range(mixed_ce_prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    var true_ce_val = true_ce_output_host[
                        true_ce_ragged_offset + s, h, hd
                    ]
                    var mixed_ce_val = mixed_ce_output_host[
                        mixed_ce_ragged_offset + s, h, hd
                    ]
                    try:
                        # 1 BF16 ULP tolerance: the SM100 1Q vs 2Q paths
                        # (dispatched per-call based on max_prompt_len)
                        # use different FP reduction orders, so cross-
                        # dispatch comparisons here aren't bit-identical
                        # even when both paths are correct. rtol=1e-2 /
                        # atol=1e-5 matches the convention used in the
                        # SM100 MHA test suite (e.g. test_mha_causal_mask).
                        assert_almost_equal(
                            true_ce_val,
                            mixed_ce_val,
                            atol=1e-5,
                            rtol=1e-2,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                        )
                        raise e^

    # Keep helper ownership explicit until helper internals are migrated.
    _ = true_ce_cache_lengths_table^
    _ = mixed_ce_cache_lengths_table^
    _ = paged_lut^


def main() raises:
    with DeviceContext() as ctx:
        # group=4 fp32 (original config)
        execute_ragged_flash_attention[
            32, KVCacheStaticParams(num_heads=8, head_size=128), DType.float32
        ](ctx)
        print("PASS: group=4 fp32 paged KV cache")
        # group=16 bf16 (405B TP=8 config)
        execute_ragged_flash_attention[
            16, KVCacheStaticParams(num_heads=1, head_size=128), DType.bfloat16
        ](ctx)
        print("PASS: group=16 bf16 paged KV cache")
