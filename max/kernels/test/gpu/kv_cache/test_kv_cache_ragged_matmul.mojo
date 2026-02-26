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

from collections import Set
from math import ceildiv
from random import random_ui64, seed

from buffer import NDBuffer
from gpu.host import DeviceBuffer, DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from linalg.matmul.gpu import _matmul_gpu
from memory import memcpy, LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from nn.kv_cache_ragged import (
    _fused_qkv_matmul_kv_cache_ragged_impl,
    _matmul_k_cache_ragged_impl,
    _matmul_kv_cache_ragged_impl,
)
from testing import assert_almost_equal

from utils import IndexList

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def _initialize_ragged_inputs[
    dtype: DType, hidden_size: Int
](
    input_row_offsets_host_ptr: UnsafePointer[Scalar[DType.uint32]],
    batch_size: Int,
    prompt_lens: List[Int],
    ctx: DeviceContext,
) -> Tuple[
    DeviceBuffer[DType.uint32],
    DeviceBuffer[dtype],
    DeviceBuffer[dtype],
    Int,  # total_length
    Int,  # max_seq_length_batch
]:
    """Initializes input row offsets and hidden state ragged tensor inputs."""
    total_length = 0
    max_seq_length_batch = -1
    for i in range(batch_size):
        input_row_offsets_host_ptr[i] = UInt32(total_length)

        curr_len = prompt_lens[i]
        total_length += curr_len
        if curr_len > max_seq_length_batch:
            max_seq_length_batch = curr_len

    input_row_offsets_host_ptr[batch_size] = UInt32(total_length)
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host_ptr)

    # Initialize ragged hidden state.
    var ragged_size = total_length * hidden_size
    var hidden_state_ragged_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        ragged_size
    )
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    var hidden_state_ragged_host = LayoutTensor[dtype, hidden_state_layout](
        hidden_state_ragged_host_ptr,
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](total_length, hidden_size)
        ),
    )
    random(hidden_state_ragged_host)

    var hidden_state_ragged_device = ctx.enqueue_create_buffer[dtype](
        ragged_size
    )
    ctx.enqueue_copy(hidden_state_ragged_device, hidden_state_ragged_host_ptr)

    # Initialize padded hidden state.
    var padded_size = batch_size * max_seq_length_batch * hidden_size
    var hidden_state_padded_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        padded_size
    )
    var hidden_state_padded_host = LayoutTensor[dtype, hidden_state_layout](
        hidden_state_padded_host_ptr,
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](batch_size * max_seq_length_batch, hidden_size)
        ),
    )

    # Copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        unpadded_seq_len = prompt_lens[bs]
        ragged_start_idx = Int(input_row_offsets_host_ptr[bs])
        for s in range(unpadded_seq_len):
            padded_ptr = (
                hidden_state_padded_host_ptr
                + (bs * max_seq_length_batch + s) * hidden_size
            )
            ragged_ptr = (
                hidden_state_ragged_host_ptr
                + (ragged_start_idx + s) * hidden_size
            )
            memcpy(dest=padded_ptr, src=ragged_ptr, count=hidden_size)

    var hidden_state_padded_device = ctx.enqueue_create_buffer[dtype](
        padded_size
    )
    ctx.enqueue_copy(hidden_state_padded_device, hidden_state_padded_host_ptr)

    # Sync here so that HtoD transfers complete prior to host buffer dtor.
    ctx.synchronize()

    hidden_state_ragged_host_ptr.free()
    hidden_state_padded_host_ptr.free()

    return (
        input_row_offsets_device,
        hidden_state_ragged_device,
        hidden_state_padded_device,
        total_length,
        max_seq_length_batch,
    )


def execute_matmul_kv_cache_ragged[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    """Tests the KV cache matmul.

    Note that here `prompt_lens` indicates the sequence length of the hidden
    states, although in general the sequence may not originate from a prompt.
    For example, in cross attention the sequence would be from a sequence of
    patch embeddings of an image.
    """
    comptime hidden_size = num_q_heads * Int(kv_params.head_size)
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime num_blocks = 32

    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ]

    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        (
            "mismatch between cache_sizes and prompt_lens, both should be"
            " batch_size in length"
        ),
    )

    batch_size = len(prompt_lens)

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size + 1
    )
    var init_result = _initialize_ragged_inputs[dtype, hidden_size](
        input_row_offsets_host_ptr, batch_size, prompt_lens, ctx
    )
    var input_row_offsets_device = init_result[0]
    var hidden_state_ragged_device = init_result[1]
    var hidden_state_padded_device = init_result[2]
    var total_length = init_result[3]
    var max_seq_length_batch = init_result[4]

    # Define layouts
    comptime weight_layout = Layout.row_major(
        2 * Int(kv_hidden_size), hidden_size
    )
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major[6]()

    # Initialize the weights.
    var weight_size = 2 * Int(kv_hidden_size) * hidden_size
    var weight_host_ptr = UnsafePointer[Scalar[dtype]].alloc(weight_size)
    var weight_shape = IndexList[2](2 * Int(kv_hidden_size), hidden_size)
    var weight_host = LayoutTensor[dtype, weight_layout](
        weight_host_ptr,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    random(weight_host)

    var weight_device = ctx.enqueue_create_buffer[dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize reference output.
    var padded_batch_dim = batch_size * max_seq_length_batch
    var ref_output_size = padded_batch_dim * 2 * Int(kv_hidden_size)
    var ref_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        ref_output_size
    )
    var ref_output_shape = IndexList[2](
        padded_batch_dim, 2 * Int(kv_hidden_size)
    )
    comptime ref_output_layout = Layout.row_major(
        UNKNOWN_VALUE, 2 * Int(kv_hidden_size)
    )
    var ref_output_host = LayoutTensor[dtype, ref_output_layout](
        ref_output_host_ptr,
        RuntimeLayout[ref_output_layout].row_major(ref_output_shape),
    )
    var ref_output_device = ctx.enqueue_create_buffer[dtype](ref_output_size)

    # Initialize our KVCache.
    var cache_lengths = ManagedLayoutTensor[DType.uint32, layout_1d](
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths.tensor[update=False]()
    var max_prompt_len = 0
    var max_context_len = 0
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_sizes[i])
        max_prompt_len = max(max_prompt_len, prompt_lens[i])
        max_context_len = max(max_context_len, cache_sizes[i] + prompt_lens[i])

    var kv_block_size = (
        num_blocks
        * 2
        * num_layers
        * max_seq_length_cache
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_length_cache,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block = ManagedLayoutTensor[dtype, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ctx,
    )

    var lookup_table = ManagedLayoutTensor[DType.uint32, layout_1d](
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var lookup_table_host = lookup_table.tensor[update=False]()

    # Hacky way to select random blocks.
    block_idx_set = Set[Int]()
    idx = 0
    while idx < batch_size:
        randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_host[idx] = UInt32(randval)
        idx += 1

    # Create runtime layouts
    var cache_len_runtime = cache_lengths_host.runtime_layout

    kv_collection_device = CollectionType(
        kv_block.device_tensor(),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            cache_lengths.device_tensor().ptr,
            cache_lengths.device_tensor().runtime_layout,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            lookup_table.device_tensor().ptr,
            lookup_table.device_tensor().runtime_layout,
        ),
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )

    k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    kv_collection_host = CollectionType(
        kv_block.tensor(),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            cache_lengths.tensor().ptr,
            cache_lengths.tensor().runtime_layout,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            lookup_table.tensor().ptr,
            lookup_table.tensor().runtime_layout,
        ),
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )

    k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # Create device LayoutTensors for kernel calls
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    var hidden_state_ragged_tensor = LayoutTensor[
        dtype, hidden_state_layout, MutAnyOrigin
    ](
        hidden_state_ragged_device.unsafe_ptr(),
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](total_length, hidden_size)
        ),
    )
    var input_row_offsets_tensor = LayoutTensor[
        DType.uint32, layout_1d, ImmutAnyOrigin
    ](
        input_row_offsets_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var weight_device_tensor = LayoutTensor[dtype, weight_layout, MutAnyOrigin](
        weight_device.unsafe_ptr(),
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )

    # Execute test.
    _matmul_kv_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_tensor,
        input_row_offsets_tensor,
        weight_device_tensor,
        k_cache_device,
        v_cache_device,
        ctx,
    )

    # Execute reference.
    var ref_output_ndbuffer = NDBuffer[dtype, 2](
        ref_output_device.unsafe_ptr(),
        ref_output_shape,
    )
    var hidden_state_padded_ndbuffer = NDBuffer[dtype, 2](
        hidden_state_padded_device.unsafe_ptr(),
        IndexList[2](padded_batch_dim, hidden_size),
    )
    var weight_ndbuffer = NDBuffer[dtype, 2](
        weight_device.unsafe_ptr(),
        weight_shape,
    )
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_ndbuffer,
        hidden_state_padded_ndbuffer,
        weight_ndbuffer,
        ctx,
    )

    _ = kv_block.tensor()
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    for bs in range(batch_size):
        prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                assert_almost_equal(
                    ref_output_host[bs * max_seq_length_batch + s, Int(k_dim)],
                    k_cache_host.load[width=1](
                        bs,
                        Int(head_idx),
                        cache_sizes[bs] + s,
                        Int(head_dim_idx),
                    ),
                    rtol=rtol,
                )

            for v_dim in range(kv_hidden_size):
                head_idx = v_dim // kv_params.head_size
                head_dim_idx = v_dim % kv_params.head_size
                assert_almost_equal(
                    ref_output_host[
                        bs * max_seq_length_batch + s,
                        Int(kv_hidden_size + v_dim),
                    ],
                    v_cache_host.load[width=1](
                        bs,
                        Int(head_idx),
                        cache_sizes[bs] + s,
                        Int(head_dim_idx),
                    ),
                    rtol=rtol,
                )

    # Cleanup host memory
    input_row_offsets_host_ptr.free()
    weight_host_ptr.free()
    ref_output_host_ptr.free()

    # Cleanup device buffers
    _ = hidden_state_ragged_device^
    _ = hidden_state_padded_device^
    _ = weight_device^
    _ = ref_output_device^
    _ = input_row_offsets_device^


def execute_matmul_k_cache_ragged[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    comptime hidden_size = num_q_heads * Int(kv_params.head_size)
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size

    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime CollectionType = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ]
    var batch_size = len(prompt_lens)
    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        "expected prompt_lens and cache_sizes size to be equal",
    )

    # Define layouts
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major[6]()
    comptime weight_layout = Layout.row_major(Int(kv_hidden_size), hidden_size)
    comptime ref_output_layout = Layout.row_major(
        UNKNOWN_VALUE, Int(kv_hidden_size)
    )
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)

    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block = ManagedLayoutTensor[dtype, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ctx,
    )

    var cache_lengths_table = CacheLengthsTable.build(
        prompt_lens, cache_sizes, ctx
    )

    var max_full_context_length = cache_lengths_table.max_full_context_length
    var max_seq_length_batch = cache_lengths_table.max_seq_length_batch

    var paged_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_full_context_length, num_paged_blocks, ctx
    )

    kv_collection_device = CollectionType(
        kv_block.device_tensor(),
        cache_lengths_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    k_cache_device = kv_collection_device.get_key_cache(layer_idx)

    kv_collection_host = CollectionType(
        kv_block.tensor(),
        cache_lengths_table.cache_lengths.host_tensor(),
        paged_lut.host_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    k_cache_host = kv_collection_host.get_key_cache(layer_idx)

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size + 1
    )
    var init_result = _initialize_ragged_inputs[dtype, hidden_size](
        input_row_offsets_host_ptr, batch_size, prompt_lens, ctx
    )
    var input_row_offsets_device = init_result[0]
    var hidden_state_ragged_device = init_result[1]
    var hidden_state_padded_device = init_result[2]
    var ragged_total_length = init_result[3]
    var init_max_seq_length_batch = init_result[4]

    # Initialize the weights.
    var weight_size = Int(kv_hidden_size) * hidden_size
    var weight_shape = IndexList[2](Int(kv_hidden_size), hidden_size)
    var weight_host_ptr = UnsafePointer[Scalar[dtype]].alloc(weight_size)
    var weight_host = LayoutTensor[dtype, weight_layout](
        weight_host_ptr,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    random(weight_host)

    var weight_device = ctx.enqueue_create_buffer[dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize reference output.
    var padded_batch_dim = batch_size * init_max_seq_length_batch
    max_seq_length_batch = init_max_seq_length_batch
    var ref_output_size = padded_batch_dim * Int(kv_hidden_size)
    var ref_output_shape = IndexList[2](padded_batch_dim, Int(kv_hidden_size))
    var ref_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        ref_output_size
    )
    var ref_output_host = LayoutTensor[dtype, ref_output_layout](
        ref_output_host_ptr,
        RuntimeLayout[ref_output_layout].row_major(ref_output_shape),
    )
    var ref_output_device = ctx.enqueue_create_buffer[dtype](ref_output_size)

    # Create device LayoutTensors for kernel calls
    var hidden_state_ragged_tensor = LayoutTensor[
        dtype, hidden_state_layout, MutAnyOrigin
    ](
        hidden_state_ragged_device.unsafe_ptr(),
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](ragged_total_length, hidden_size)
        ),
    )
    var input_row_offsets_tensor = LayoutTensor[
        DType.uint32, layout_1d, ImmutAnyOrigin
    ](
        input_row_offsets_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var weight_device_tensor = LayoutTensor[dtype, weight_layout, MutAnyOrigin](
        weight_device.unsafe_ptr(),
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )

    # Execute test.
    _matmul_k_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_tensor,
        input_row_offsets_tensor,
        weight_device_tensor,
        k_cache_device,
        ctx,
    )

    # Execute reference.
    var ref_output_ndbuffer = NDBuffer[dtype, 2](
        ref_output_device.unsafe_ptr(),
        ref_output_shape,
    )
    var hidden_state_padded_ndbuffer = NDBuffer[dtype, 2](
        hidden_state_padded_device.unsafe_ptr(),
        IndexList[2](padded_batch_dim, hidden_size),
    )
    var weight_ndbuffer = NDBuffer[dtype, 2](
        weight_device.unsafe_ptr(),
        weight_shape,
    )
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_ndbuffer,
        hidden_state_padded_ndbuffer,
        weight_ndbuffer,
        ctx,
    )

    _ = kv_block.tensor()
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    for bs in range(batch_size):
        prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                assert_almost_equal(
                    ref_output_host[bs * max_seq_length_batch + s, Int(k_dim)],
                    k_cache_host.load[width=1](
                        bs,
                        Int(head_idx),
                        cache_sizes[bs] + s,
                        Int(head_dim_idx),
                    ),
                    rtol=rtol,
                )

    # Cleanup host memory
    input_row_offsets_host_ptr.free()
    weight_host_ptr.free()
    ref_output_host_ptr.free()

    # Cleanup device buffers
    _ = hidden_state_ragged_device^
    _ = hidden_state_padded_device^
    _ = weight_device^
    _ = ref_output_device^
    _ = input_row_offsets_device^

    # Cleanup managed objects.
    _ = cache_lengths_table^
    _ = paged_lut^


def generic_assert_output_equals[
    cache_t: KVCacheT, dtype: DType, //, num_q_heads: Int, rtol: Float64
](
    k_cache: cache_t,
    v_cache: cache_t,
    ref_output_device: DeviceBuffer[dtype],
    ref_output_shape: IndexList[2],
    test_output_device: DeviceBuffer[dtype],
    test_output_shape: IndexList[2],
    prompt_lens: List[Int],
    max_seq_length_batch: Int,
    ctx: DeviceContext,
):
    comptime assert cache_t.dtype == dtype, "type mismatch"
    comptime kv_params = cache_t.kv_params
    comptime hidden_size = num_q_heads * Int(kv_params.head_size)
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime fused_hidden_size = 2 * Int(kv_hidden_size) + hidden_size
    comptime ref_output_layout = Layout.row_major(
        UNKNOWN_VALUE, fused_hidden_size
    )
    comptime test_output_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)

    # Allocate host memory and copy from device
    var ref_output_size = ref_output_shape[0] * ref_output_shape[1]
    var ref_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        ref_output_size
    )
    var ref_output_host = LayoutTensor[dtype, ref_output_layout](
        ref_output_host_ptr,
        RuntimeLayout[ref_output_layout].row_major(ref_output_shape),
    )

    var test_output_size = test_output_shape[0] * test_output_shape[1]
    var test_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        test_output_size
    )
    var test_output_host = LayoutTensor[dtype, test_output_layout](
        test_output_host_ptr,
        RuntimeLayout[test_output_layout].row_major(test_output_shape),
    )

    ctx.enqueue_copy(test_output_host_ptr, test_output_device)
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    batch_size = len(prompt_lens)

    ragged_offset = 0
    for bs in range(batch_size):
        prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for q_dim in range(hidden_size):
                try:
                    assert_almost_equal(
                        ref_output_host[
                            bs * max_seq_length_batch + s,
                            q_dim,
                        ],
                        test_output_host[ragged_offset + s, q_dim],
                        rtol=rtol,
                    )
                except e:
                    print("Q", bs, s, q_dim)
                    raise e^

            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                try:
                    assert_almost_equal(
                        ref_output_host[
                            bs * max_seq_length_batch + s,
                            hidden_size + Int(k_dim),
                        ],
                        k_cache.load[width=1](
                            bs,
                            Int(head_idx),
                            k_cache.cache_length(bs) + s,
                            Int(head_dim_idx),
                        ).cast[dtype](),
                        rtol=rtol,
                    )
                except e:
                    print("K", bs, s, k_dim)
                    raise e^

            for v_dim in range(kv_hidden_size):
                head_idx = v_dim // kv_params.head_size
                head_dim_idx = v_dim % kv_params.head_size
                try:
                    assert_almost_equal(
                        ref_output_host[
                            bs * max_seq_length_batch + s,
                            hidden_size + Int(kv_hidden_size + v_dim),
                        ],
                        v_cache.load[width=1](
                            bs,
                            Int(head_idx),
                            v_cache.cache_length(bs) + s,
                            Int(head_dim_idx),
                        ).cast[dtype](),
                        rtol=rtol,
                    )
                except e:
                    print("V", bs, s, v_dim)
                    raise e^

        ragged_offset += prompt_len

    # Cleanup host memory
    ref_output_host_ptr.free()
    test_output_host_ptr.free()


def generic_execute_fused_qkv_cache_ragged[
    cache_t: KVCacheT,
    //,
    kv_params: KVCacheStaticParams,
    dtype: DType,
    num_q_heads: Int,
](
    prompt_lens: List[Int],
    cache_sizes: List[Int],
    k_cache: cache_t,
    v_cache: cache_t,
    ctx: DeviceContext,
) -> Tuple[
    DeviceBuffer[dtype],
    IndexList[2],  # ref_output_shape
    DeviceBuffer[dtype],
    IndexList[2],  # test_output_shape
]:
    """Executes fused QKV matmul, writing results kv_cache objects.

    Returns:
      - Tuple containing ref_output_device, ref_output_shape,
        test_output_device, test_output_shape.
    """
    comptime hidden_size = num_q_heads * Int(kv_params.head_size)
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime fused_hidden_size = (2 * Int(kv_hidden_size)) + hidden_size
    comptime num_blocks = 32
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime weight_layout = Layout.row_major(fused_hidden_size, hidden_size)
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)

    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        (
            "mismatch between cache_sizes and prompt_lens, both should be"
            " batch_size in length"
        ),
    )

    batch_size = len(prompt_lens)

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        num_blocks,
        ")",
    )

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size + 1
    )
    var init_result = _initialize_ragged_inputs[dtype, hidden_size](
        input_row_offsets_host_ptr, batch_size, prompt_lens, ctx
    )
    var input_row_offsets_device = init_result[0]
    var hidden_state_ragged_device = init_result[1]
    var hidden_state_padded_device = init_result[2]
    var total_length = init_result[3]
    var max_seq_length_batch = init_result[4]

    # Initialize the weights
    var weight_size = fused_hidden_size * hidden_size
    var weight_shape = IndexList[2](fused_hidden_size, hidden_size)
    var weight_host_ptr = UnsafePointer[Scalar[dtype]].alloc(weight_size)
    var weight_host = LayoutTensor[dtype, weight_layout](
        weight_host_ptr,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    random(weight_host)

    var weight_device = ctx.enqueue_create_buffer[dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize reference output
    var padded_batch_dim = batch_size * max_seq_length_batch
    var ref_output_size = padded_batch_dim * fused_hidden_size
    var ref_output_shape = IndexList[2](padded_batch_dim, fused_hidden_size)
    var ref_output_device = ctx.enqueue_create_buffer[dtype](ref_output_size)

    # Initialize test output
    var test_output_size = total_length * hidden_size
    var test_output_shape = IndexList[2](total_length, hidden_size)
    var test_output_device = ctx.enqueue_create_buffer[dtype](test_output_size)

    # Create device LayoutTensors for kernel calls
    var hidden_state_ragged_tensor = LayoutTensor[
        dtype, hidden_state_layout, MutAnyOrigin
    ](
        hidden_state_ragged_device.unsafe_ptr(),
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](total_length, hidden_size)
        ),
    )
    var input_row_offsets_tensor = LayoutTensor[
        DType.uint32, layout_1d, ImmutAnyOrigin
    ](
        input_row_offsets_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var weight_device_tensor = LayoutTensor[dtype, weight_layout, MutAnyOrigin](
        weight_device.unsafe_ptr(),
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    var test_output_device_tensor = LayoutTensor[
        dtype, hidden_state_layout, MutAnyOrigin
    ](
        test_output_device.unsafe_ptr(),
        RuntimeLayout[hidden_state_layout].row_major(test_output_shape),
    )

    # Execute the matmul
    _fused_qkv_matmul_kv_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_tensor,
        input_row_offsets_tensor,
        weight_device_tensor,
        k_cache,
        v_cache,
        test_output_device_tensor,
        ctx,
    )

    # Execute reference
    var ref_output_ndbuffer = NDBuffer[dtype, 2](
        ref_output_device.unsafe_ptr(),
        ref_output_shape,
    )
    var hidden_state_padded_ndbuffer = NDBuffer[dtype, 2](
        hidden_state_padded_device.unsafe_ptr(),
        IndexList[2](padded_batch_dim, hidden_size),
    )
    var weight_ndbuffer = NDBuffer[dtype, 2](
        weight_device.unsafe_ptr(),
        weight_shape,
    )
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_ndbuffer,
        hidden_state_padded_ndbuffer,
        weight_ndbuffer,
        ctx,
    )

    # Cleanup host memory
    input_row_offsets_host_ptr.free()
    weight_host_ptr.free()

    # Cleanup device buffers that are no longer needed
    _ = hidden_state_ragged_device^
    _ = hidden_state_padded_device^
    _ = weight_device^
    _ = input_row_offsets_device^

    return (
        ref_output_device,
        ref_output_shape,
        test_output_device,
        test_output_shape,
    )


def execute_paged_fused_qkv_matmul[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime CollectionType = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ]
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major[6]()

    var batch_size = len(prompt_lens)
    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        "expected prompt_lens and cache_sizes size to be equal",
    )

    var cache_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )

    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block = ManagedLayoutTensor[dtype, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ctx,
    )

    var cache_lengths_table = CacheLengthsTable.build(
        prompt_lens, cache_sizes, ctx
    )

    var max_full_context_length = cache_lengths_table.max_full_context_length
    var max_seq_length_batch = cache_lengths_table.max_seq_length_batch

    var paged_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_full_context_length, num_paged_blocks, ctx
    )

    kv_collection_device = CollectionType(
        kv_block.device_tensor(),
        cache_lengths_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    kv_collection_host = CollectionType(
        kv_block.tensor(),
        cache_lengths_table.cache_lengths.host_tensor(),
        paged_lut.host_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # Execute the matmul
    var results = generic_execute_fused_qkv_cache_ragged[
        kv_params, dtype, num_q_heads
    ](prompt_lens, cache_sizes, k_cache_device, v_cache_device, ctx)

    var ref_output_device = results[0]
    var ref_output_shape = results[1]
    var test_output_device = results[2]
    var test_output_shape = results[3]

    _ = kv_block.tensor()

    generic_assert_output_equals[num_q_heads=num_q_heads, rtol=rtol](
        k_cache_host,
        v_cache_host,
        ref_output_device,
        ref_output_shape,
        test_output_device,
        test_output_shape,
        prompt_lens,
        max_seq_length_batch,
        ctx,
    )

    # Cleanup device buffers
    _ = ref_output_device^
    _ = test_output_device^

    # Cleanup managed objects.
    _ = cache_lengths_table^
    _ = paged_lut^


def execute_cont_batch_fused_qkv_matmul[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ]
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major[6]()

    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        (
            "mismatch between cache_sizes and prompt_lens, both should be"
            " batch_size in length"
        ),
    )

    # Initialize our KVCache
    var batch_size = len(cache_sizes)
    var cache_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )
    var max_seq_length_batch = -1
    var max_context_length = 0

    for i in range(batch_size):
        cache_lengths_host_ptr[i] = UInt32(cache_sizes[i])
        max_context_length = max(
            max_context_length, cache_sizes[i] + prompt_lens[i]
        )
        if prompt_lens[i] > max_seq_length_batch:
            max_seq_length_batch = prompt_lens[i]

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host_ptr)

    var kv_block_size = (
        num_blocks
        * 2
        * num_layers
        * max_seq_length_cache
        * Int(kv_params.num_heads)
        * Int(kv_params.head_size)
    )
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_length_cache,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_host_ptr = UnsafePointer[Scalar[dtype]].alloc(kv_block_size)
    var kv_block_device = ctx.enqueue_create_buffer[dtype](kv_block_size)

    var lookup_table_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )

    # Hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_host_ptr[idx] = UInt32(randval)
        idx += 1

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(lookup_table_device, lookup_table_host_ptr)

    # Create runtime layouts
    var kv_block_runtime = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var cache_len_runtime = RuntimeLayout[layout_1d].row_major(
        IndexList[1](batch_size)
    )

    var kv_collection_device = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_device.unsafe_ptr(),
            kv_block_runtime,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            cache_lengths_device.unsafe_ptr(),
            cache_len_runtime,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            lookup_table_device.unsafe_ptr(),
            cache_len_runtime,
        ),
        UInt32(max_seq_length_batch),
        UInt32(max_context_length),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    var kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_host_ptr,
            kv_block_runtime,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            cache_lengths_host_ptr,
            cache_len_runtime,
        ),
        LayoutTensor[DType.uint32, layout_1d, ImmutAnyOrigin](
            lookup_table_host_ptr,
            cache_len_runtime,
        ),
        UInt32(max_seq_length_batch),
        UInt32(max_context_length),
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # Execute the matmul
    var results = generic_execute_fused_qkv_cache_ragged[
        kv_params, dtype, num_q_heads
    ](prompt_lens, cache_sizes, k_cache_device, v_cache_device, ctx)

    var ref_output_device = results[0]
    var ref_output_shape = results[1]
    var test_output_device = results[2]
    var test_output_shape = results[3]

    ctx.enqueue_copy(kv_block_host_ptr, kv_block_device)

    generic_assert_output_equals[num_q_heads=num_q_heads, rtol=rtol](
        k_cache_host,
        v_cache_host,
        ref_output_device,
        ref_output_shape,
        test_output_device,
        test_output_shape,
        prompt_lens,
        max_seq_length_batch,
        ctx,
    )

    # Cleanup host memory
    cache_lengths_host_ptr.free()
    kv_block_host_ptr.free()
    lookup_table_host_ptr.free()

    # Cleanup device buffers
    _ = kv_block_device^
    _ = lookup_table_device^
    _ = cache_lengths_device^
    _ = ref_output_device^
    _ = test_output_device^


# TODO implement fused qkv matmul for paged
def execute_fused_matmul_suite(ctx: DeviceContext):
    comptime dtypes_tolerances = ((DType.float32, 1e-3), (DType.bfloat16, 1e-2))

    comptime for dtype_idx in range(2):
        comptime dtype = dtypes_tolerances[dtype_idx][0]
        comptime rtol = dtypes_tolerances[dtype_idx][1]

        for bs in [1, 16]:
            ce_cache_sizes = List[Int]()
            ce_seq_lens = List[Int]()
            tg_cache_sizes = List[Int]()
            tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                # TODO increase sizes here to ensure we cross page boundary.
                tg_cache_sizes.append(Int(random_ui64(512, 700)))
                ce_seq_lens.append(Int(random_ui64(512, 700)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_cont_batch_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)
            execute_paged_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)
            execute_matmul_kv_cache_ragged[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](
                ce_seq_lens,
                max_seq_length_cache=1024,
                cache_sizes=ce_cache_sizes,
                num_layers=4,
                layer_idx=1,
                ctx=ctx,
            )
            execute_matmul_k_cache_ragged[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)

            # llama3 token gen
            execute_cont_batch_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)
            execute_paged_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)
            execute_matmul_kv_cache_ragged[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](
                tg_seq_lens,
                max_seq_length_cache=1024,
                cache_sizes=tg_cache_sizes,
                num_layers=4,
                layer_idx=3,
                ctx=ctx,
            )
            execute_matmul_k_cache_ragged[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite(ctx)
