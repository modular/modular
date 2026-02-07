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

from math import rsqrt

from gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from memory import memcpy, LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils import IndexList

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable


def execute_ragged_flash_attention(
    ctx: DeviceContext,
):
    comptime num_q_heads = 32
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    comptime type = DType.float32
    comptime num_paged_blocks = 32
    comptime page_size = 128
    comptime PagedCollectionType = PagedKVCacheCollection[
        type, kv_params, page_size
    ]
    var num_layers = 1
    var layer_idx = 0

    var true_ce_prompt_lens = [100, 200, 300, 400]
    var mixed_ce_prompt_lens = [50, 100, 150, 100]

    var true_ce_cache_lens = [0, 0, 0, 0]
    var mixed_ce_cache_lens = [50, 100, 150, 300]

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
    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, Int(kv_params.head_size)
    )
    var true_ce_q_size = (
        true_ce_total_length * num_q_heads * Int(kv_params.head_size)
    )
    var mixed_ce_q_size = (
        mixed_ce_total_length * num_q_heads * Int(kv_params.head_size)
    )

    var true_ce_q_ragged_host_ptr = UnsafePointer[Scalar[type]].alloc(
        true_ce_q_size
    )
    var true_ce_q_ragged_host = LayoutTensor[type, q_layout](
        true_ce_q_ragged_host_ptr,
        RuntimeLayout[q_layout].row_major(
            IndexList[3](
                true_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    )
    random(true_ce_q_ragged_host)

    var true_ce_q_ragged_device = ctx.enqueue_create_buffer[type](
        true_ce_q_size
    )
    ctx.enqueue_copy(true_ce_q_ragged_device, true_ce_q_ragged_host_ptr)

    var mixed_ce_q_ragged_host_ptr = UnsafePointer[Scalar[type]].alloc(
        mixed_ce_q_size
    )
    var mixed_ce_q_ragged_host = LayoutTensor[type, q_layout](
        mixed_ce_q_ragged_host_ptr,
        RuntimeLayout[q_layout].row_major(
            IndexList[3](
                mixed_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    )

    true_ce_row_offsets_host_ptr = (
        true_ce_cache_lengths_table.input_row_offsets.host_ptr
    )
    mixed_ce_row_offsets_host_ptr = (
        mixed_ce_cache_lengths_table.input_row_offsets.host_ptr
    )

    var head_stride = num_q_heads * Int(kv_params.head_size)
    for bs_idx in range(batch_size):
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        true_ce_row_offset = Int(true_ce_row_offsets_host_ptr[bs_idx])
        mixed_ce_row_offset = Int(mixed_ce_row_offsets_host_ptr[bs_idx])

        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        true_ce_offset = (
            true_ce_q_ragged_host_ptr
            + (true_ce_row_offset + mixed_ce_cache_len) * head_stride
        )
        mixed_ce_offset = (
            mixed_ce_q_ragged_host_ptr + mixed_ce_row_offset * head_stride
        )

        memcpy(
            dest=mixed_ce_offset,
            src=true_ce_offset,
            count=mixed_ce_prompt_len * head_stride,
        )

    var mixed_ce_q_ragged_device = ctx.enqueue_create_buffer[type](
        mixed_ce_q_size
    )
    ctx.enqueue_copy(mixed_ce_q_ragged_device, mixed_ce_q_ragged_host_ptr)

    # Initialize output buffers
    var mixed_ce_output_host_ptr = UnsafePointer[Scalar[type]].alloc(
        mixed_ce_q_size
    )
    var mixed_ce_output_host = LayoutTensor[type, q_layout](
        mixed_ce_output_host_ptr,
        RuntimeLayout[q_layout].row_major(
            IndexList[3](
                mixed_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    )
    var mixed_ce_output_device = ctx.enqueue_create_buffer[type](
        mixed_ce_q_size
    )

    var true_ce_output_host_ptr = UnsafePointer[Scalar[type]].alloc(
        true_ce_q_size
    )
    var true_ce_output_host = LayoutTensor[type, q_layout](
        true_ce_output_host_ptr,
        RuntimeLayout[q_layout].row_major(
            IndexList[3](
                true_ce_total_length, num_q_heads, Int(kv_params.head_size)
            )
        ),
    )
    var true_ce_output_device = ctx.enqueue_create_buffer[type](true_ce_q_size)

    # Initialize KVCache
    comptime kv_layout = Layout.row_major[6]()
    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var kv_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_paged_host_ptr = UnsafePointer[Scalar[type]].alloc(
        kv_block_size
    )
    var kv_block_paged_host = LayoutTensor[type, kv_layout](
        kv_block_paged_host_ptr,
        RuntimeLayout[kv_layout].row_major(kv_shape),
    )
    random(kv_block_paged_host)

    var paged_lut = PagedLookupTable[page_size].build(
        true_ce_prompt_lens,
        true_ce_cache_lens,
        true_ce_max_full_context_length,
        num_paged_blocks,
        ctx,
    )

    var kv_block_paged_device = ctx.enqueue_create_buffer[type](kv_block_size)
    ctx.enqueue_copy(kv_block_paged_device, kv_block_paged_host_ptr)

    # Create device LayoutTensors for KV cache
    var kv_block_runtime = RuntimeLayout[kv_layout].row_major(kv_shape)

    true_ce_kv_collection_device = PagedCollectionType(
        LayoutTensor[type, kv_layout, MutAnyOrigin](
            kv_block_paged_device.unsafe_ptr(),
            kv_block_runtime,
        ),
        true_ce_cache_lengths_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(true_ce_max_prompt_length),
        UInt32(true_ce_max_full_context_length),
    )

    mixed_ce_kv_collection_device = PagedCollectionType(
        LayoutTensor[type, kv_layout, MutAnyOrigin](
            kv_block_paged_device.unsafe_ptr(),
            kv_block_runtime,
        ),
        mixed_ce_cache_lengths_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(mixed_ce_max_prompt_length),
        UInt32(mixed_ce_max_full_context_length),
    )

    # Create device LayoutTensors for flash_attention
    var true_ce_q_runtime = RuntimeLayout[q_layout].row_major(
        IndexList[3](
            true_ce_total_length, num_q_heads, Int(kv_params.head_size)
        )
    )
    var mixed_ce_q_runtime = RuntimeLayout[q_layout].row_major(
        IndexList[3](
            mixed_ce_total_length, num_q_heads, Int(kv_params.head_size)
        )
    )

    # "true CE" execution
    print("true")
    flash_attention[ragged=True](
        LayoutTensor[type, q_layout, MutAnyOrigin](
            true_ce_output_device.unsafe_ptr(),
            true_ce_q_runtime,
        ),
        LayoutTensor[type, q_layout, ImmutAnyOrigin](
            true_ce_q_ragged_device.unsafe_ptr(),
            true_ce_q_runtime,
        ),
        true_ce_kv_collection_device.get_key_cache(layer_idx),
        true_ce_kv_collection_device.get_value_cache(layer_idx),
        CausalMask(),
        IdentityScoreMod(),
        true_ce_cache_lengths_table.input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.synchronize()

    # "mixed CE" execution
    print("mixed")
    flash_attention[ragged=True](
        LayoutTensor[type, q_layout, MutAnyOrigin](
            mixed_ce_output_device.unsafe_ptr(),
            mixed_ce_q_runtime,
        ),
        LayoutTensor[type, q_layout, ImmutAnyOrigin](
            mixed_ce_q_ragged_device.unsafe_ptr(),
            mixed_ce_q_runtime,
        ),
        mixed_ce_kv_collection_device.get_key_cache(layer_idx),
        mixed_ce_kv_collection_device.get_value_cache(layer_idx),
        CausalMask(),
        IdentityScoreMod(),
        mixed_ce_cache_lengths_table.input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.enqueue_copy(mixed_ce_output_host_ptr, mixed_ce_output_device)
    ctx.enqueue_copy(true_ce_output_host_ptr, true_ce_output_device)
    ctx.synchronize()

    for bs in range(batch_size):
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs]
        mixed_ce_row_offset = Int(mixed_ce_row_offsets_host_ptr[bs])
        true_ce_row_offset = Int(true_ce_row_offsets_host_ptr[bs])
        mixed_ce_cache_len = mixed_ce_cache_lens[bs]

        true_ce_ragged_offset = true_ce_row_offset + mixed_ce_cache_len
        mixed_ce_ragged_offset = mixed_ce_row_offset
        for s in range(mixed_ce_prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    true_ce_val = true_ce_output_host[
                        true_ce_ragged_offset + s, h, Int(hd)
                    ]
                    mixed_ce_val = mixed_ce_output_host[
                        mixed_ce_ragged_offset + s, h, Int(hd)
                    ]
                    try:
                        assert_almost_equal(
                            true_ce_val,
                            mixed_ce_val,
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

    # Cleanup host memory
    true_ce_q_ragged_host_ptr.free()
    mixed_ce_q_ragged_host_ptr.free()
    mixed_ce_output_host_ptr.free()
    true_ce_output_host_ptr.free()
    kv_block_paged_host_ptr.free()

    # Cleanup device buffers
    _ = true_ce_q_ragged_device^
    _ = mixed_ce_q_ragged_device^
    _ = mixed_ce_output_device^
    _ = true_ce_output_device^
    _ = kv_block_paged_device^

    # Cleanup managed objects.
    _ = true_ce_cache_lengths_table^
    _ = mixed_ce_cache_lengths_table^
    _ = paged_lut^


def main():
    with DeviceContext() as ctx:
        execute_ragged_flash_attention(ctx)
