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

from collections import OptionalReg, Set
from math import rsqrt
from memory import memcpy
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from random import random_ui64, seed

from gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils import Index, IndexList

from sys import has_amd_gpu_accelerator

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def execute_ragged_flash_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    sink: Bool = False,
](
    valid_lengths: List[Int],
    max_seq_len_cache: Int,
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ]

    var batch_size = len(valid_lengths)
    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )
    debug_assert(
        len(valid_lengths) == len(cache_lengths),
        "expected valid_lengths and cache_lengths size to be equal",
    )

    # Define layouts
    comptime input_row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_static_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, Int(kv_params.head_size)
    )
    comptime q_padded_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_q_heads, Int(kv_params.head_size)
    )
    comptime kv_block_static_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )

    var total_length = 0
    var max_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_context_length = max(
            max_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    # Create device buffers
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    var valid_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )

    # Initialize row offsets and lengths
    var input_row_offsets_host_ptr = UnsafePointer[UInt32].alloc(batch_size + 1)
    var running_total = 0
    for i in range(batch_size):
        input_row_offsets_host_ptr[i] = UInt32(running_total)
        running_total += valid_lengths[i]
    input_row_offsets_host_ptr[batch_size] = UInt32(running_total)
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host_ptr)

    with cache_lengths_device.map_to_host() as cache_lengths_host:
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(cache_lengths[i])

    with valid_lengths_device.map_to_host() as valid_lengths_host:
        for i in range(batch_size):
            valid_lengths_host[i] = UInt32(valid_lengths[i])

    # Create q tensors
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, Int(kv_params.head_size)
    )
    var q_ragged_runtime_layout = RuntimeLayout[
        q_ragged_static_layout
    ].row_major(q_ragged_shape)

    var q_padded_shape = IndexList[4](
        batch_size, max_prompt_length, num_q_heads, Int(kv_params.head_size)
    )
    var q_padded_runtime_layout = RuntimeLayout[
        q_padded_static_layout
    ].row_major(q_padded_shape)

    var q_ragged_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_padded_device = ctx.enqueue_create_buffer[dtype](
        q_padded_shape.flattened_length()
    )

    # Initialize q_ragged with random data
    with q_ragged_device.map_to_host() as q_ragged_host:
        var q_ragged_tensor = LayoutTensor[dtype, q_ragged_static_layout](
            q_ragged_host, q_ragged_runtime_layout
        )
        random(q_ragged_tensor)

        # Also initialize q_padded by copying from q_ragged
        with q_padded_device.map_to_host() as q_padded_host:
            var q_padded_tensor = LayoutTensor[dtype, q_padded_static_layout](
                q_padded_host, q_padded_runtime_layout
            )
            # copy over the ragged values to the padded tensor.
            # Don't worry about padded values, we won't read them.
            for bs in range(batch_size):
                unpadded_seq_len = valid_lengths[bs]
                ragged_start_idx = Int(input_row_offsets_host_ptr[bs])
                padded_ptr = q_padded_tensor.ptr + (
                    bs
                    * max_prompt_length
                    * num_q_heads
                    * Int(kv_params.head_size)
                )
                ragged_ptr = q_ragged_tensor.ptr + (
                    ragged_start_idx * num_q_heads * Int(kv_params.head_size)
                )
                memcpy(
                    dest=padded_ptr,
                    src=ragged_ptr,
                    count=unpadded_seq_len
                    * num_q_heads
                    * Int(kv_params.head_size),
                )

    # Free the host pointer after use
    ctx.synchronize()
    input_row_offsets_host_ptr.free()

    # Create output tensors
    var ref_output_shape = IndexList[4](
        batch_size, max_prompt_length, num_q_heads, Int(kv_params.head_size)
    )
    var ref_output_runtime_layout = RuntimeLayout[
        q_padded_static_layout
    ].row_major(ref_output_shape)
    var ref_output_device = ctx.enqueue_create_buffer[dtype](
        ref_output_shape.flattened_length()
    )

    var test_output_shape = IndexList[3](
        total_length, num_q_heads, Int(kv_params.head_size)
    )
    var test_output_runtime_layout = RuntimeLayout[
        q_ragged_static_layout
    ].row_major(test_output_shape)
    var test_output_device = ctx.enqueue_create_buffer[dtype](
        test_output_shape.flattened_length()
    )

    # Initialize kv_block with random data using regular host memory
    # (not host-pinned memory via map_to_host) to avoid exhausting
    # the limited host-pinned memory buffer cache
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_len_cache,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_runtime_layout = RuntimeLayout[
        kv_block_static_layout
    ].row_major(kv_block_shape)

    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )

    var kv_block_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        kv_block_shape.flattened_length()
    )
    var kv_block_host_tensor = LayoutTensor[dtype, kv_block_static_layout](
        kv_block_host_ptr, kv_block_runtime_layout
    )
    random(kv_block_host_tensor)
    ctx.enqueue_copy(kv_block_device, kv_block_host_ptr)
    ctx.synchronize()
    kv_block_host_ptr.free()

    # Create lookup table
    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )

    # Initialize lookup table with random block indices
    with lookup_table_device.map_to_host() as lookup_table_host:
        var block_idx_set = Set[Int]()
        var idx = 0
        while idx < batch_size:
            var randval = Int(random_ui64(0, num_blocks - 1))
            if randval in block_idx_set:
                continue
            block_idx_set.add(randval)
            lookup_table_host[idx] = UInt32(randval)
            idx += 1

    # Create layout tensors for GPU operations
    var input_row_offsets_runtime_layout = RuntimeLayout[
        input_row_offsets_layout
    ].row_major(Index(batch_size + 1))

    var input_row_offsets_tensor = LayoutTensor[
        DType.uint32, input_row_offsets_layout
    ](input_row_offsets_device, input_row_offsets_runtime_layout)

    var valid_lengths_tensor = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        valid_lengths_device,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(batch_size)
        ),
    )

    var cache_lengths_tensor = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
    )

    var lookup_table_tensor = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
    )

    var kv_block_tensor = LayoutTensor[dtype, kv_block_static_layout](
        kv_block_device, kv_block_runtime_layout
    )

    var kv_collection_device = CollectionType(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_tensor.runtime_layout.shape.value,
                kv_block_tensor.runtime_layout.stride.value,
            ),
        ),
        cache_lengths_tensor,
        lookup_table_tensor,
        UInt32(max_prompt_length),
        UInt32(max_context_length),
    )
    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    # Create sink weights
    var sink_weights_shape = IndexList[1](num_q_heads)
    var sink_weights_device = ctx.enqueue_create_buffer[dtype](num_q_heads)

    # Initialize sink weights with varying negative values
    with sink_weights_device.map_to_host() as sink_weights_host:
        for h in range(num_q_heads):
            sink_weights_host[h] = Scalar[dtype](-2.0 - 0.5 * Float64(h))

    var sink_weights_device_tensor: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None

    comptime if sink:
        sink_weights_device_tensor = LayoutTensor[
            dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            sink_weights_device,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                sink_weights_shape
            ),
        )

    var q_ragged_tensor = LayoutTensor[dtype, q_ragged_static_layout](
        q_ragged_device, q_ragged_runtime_layout
    )
    var q_padded_tensor = LayoutTensor[dtype, q_padded_static_layout](
        q_padded_device, q_padded_runtime_layout
    )
    var test_output_tensor = LayoutTensor[dtype, q_ragged_static_layout](
        test_output_device, test_output_runtime_layout
    )
    var ref_output_tensor = LayoutTensor[dtype, q_padded_static_layout](
        ref_output_device, ref_output_runtime_layout
    )

    # ragged execution with sink weights
    flash_attention[ragged=True, sink=sink](
        test_output_tensor,
        q_ragged_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        IdentityScoreMod(),
        input_row_offsets_tensor,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
        sink_weights=sink_weights_device_tensor,
    )
    ctx.synchronize()

    # padded execution
    flash_attention[sink=sink, naive_kernel=True](
        ref_output_tensor,
        q_padded_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        IdentityScoreMod(),
        valid_lengths_tensor,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
        sink_weights=sink_weights_device_tensor,
    )

    ctx.synchronize()

    # Verify results
    with test_output_device.map_to_host() as test_out_host:
        with ref_output_device.map_to_host() as ref_out_host:
            with input_row_offsets_device.map_to_host() as row_offsets_host:
                var test_out_tensor = LayoutTensor[
                    dtype, q_ragged_static_layout
                ](test_out_host, test_output_runtime_layout)
                var ref_out_tensor = LayoutTensor[
                    dtype, q_padded_static_layout
                ](ref_out_host, ref_output_runtime_layout)

                for bs in range(batch_size):
                    prompt_len = valid_lengths[bs]
                    ragged_offset = Int(row_offsets_host[bs])
                    for s in range(prompt_len):
                        for h in range(num_q_heads):
                            for hd in range(kv_params.head_size):
                                var ref_val = ref_out_tensor[bs, s, h, Int(hd)]
                                var test_val = test_out_tensor[
                                    ragged_offset + s, h, Int(hd)
                                ]
                                try:
                                    # amd uses more aggressive split-k partitioning
                                    var rtol_bf16 = (
                                        2e-2 if has_amd_gpu_accelerator() else 1e-2
                                    )
                                    assert_almost_equal(
                                        ref_val,
                                        test_val,
                                        rtol=rtol_bf16 if dtype
                                        == DType.bfloat16 else 1e-4,
                                        atol=5e-3,  # numerical instability between naive and optimized kernels
                                    )
                                except e:
                                    print(
                                        "MISMATCH:",
                                        bs,
                                        s,
                                        h,
                                        hd,
                                        ref_val,
                                        test_val,
                                    )
                                    raise e^

    # Explicitly free device buffers to return memory to the buffer cache
    _ = input_row_offsets_device^
    _ = cache_lengths_device^
    _ = valid_lengths_device^
    _ = q_ragged_device^
    _ = q_padded_device^
    _ = ref_output_device^
    _ = test_output_device^
    _ = kv_block_device^
    _ = lookup_table_device^
    _ = sink_weights_device^


def execute_flash_attention_suite(ctx: DeviceContext):
    comptime dtypes = (DType.float32, DType.bfloat16)

    for bs in [1, 16]:
        comptime for dtype_idx in range(len(dtypes)):
            comptime dtype = dtypes[dtype_idx]

            ce_cache_sizes = List[Int]()
            ce_seq_lens = List[Int]()
            tg_cache_sizes = List[Int]()
            tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(Int(random_ui64(512, 1024)))
                ce_seq_lens.append(Int(random_ui64(512, 1024)))
                ce_cache_sizes.append(0)
            print("CE", bs, dtype)
            execute_ragged_flash_attention[
                llama_num_q_heads, dtype, kv_params_llama3
            ](ce_seq_lens, 1024, ce_cache_sizes, 2, 1, ctx)

            print("TG", bs, dtype)
            execute_ragged_flash_attention[
                llama_num_q_heads, dtype, kv_params_llama3
            ](tg_seq_lens, 1024, tg_cache_sizes, 2, 0, ctx)

    # edge cases
    var short_ce_seq_len: List[Int] = [2]
    var short_ce_cache_size: List[Int] = [0]
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3
    ](short_ce_seq_len, 1024, short_ce_cache_size, 2, 1, ctx)


def test_flash_attention_with_sink_weights(ctx: DeviceContext):
    var valid_lengths: List[Int] = [100, 200, 300]
    var max_seq_len_cache = 1024
    var cache_lengths: List[Int] = [100, 200, 300]
    var num_layers = 1
    var layer_idx = 0

    execute_ragged_flash_attention[
        llama_num_q_heads, DType.float32, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )

    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )

    valid_lengths: List[Int] = [1, 1, 1]
    print("Testing TG")
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.float32, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )
    print("Testing TG BF16")
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)

        # Test sink weights functionality
        test_flash_attention_with_sink_weights(ctx)
