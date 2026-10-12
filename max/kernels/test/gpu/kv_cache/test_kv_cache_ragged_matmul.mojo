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

from std.random import random_ui64, seed
from std.math.uutils import udivmod
from std.sys.defines import get_defined_string

from max.gpu.host import DeviceBuffer, DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from linalg.matmul.gpu import _matmul_gpu
from std.memory import unsafe_memcpy
from nn.kv_cache_ragged import (
    _fused_qkv_matmul_kv_cache_ragged_impl,
    _matmul_k_cache_ragged_impl,
    _matmul_kv_cache_ragged_impl,
)
from std.testing import assert_almost_equal

from std.utils import IndexList

from kv_cache_test_utils import (
    CacheLengthsTable,
    PagedLookupTable,
    random_distinct,
)

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def _initialize_ragged_inputs[
    dtype: DType, hidden_size: Int
](
    input_row_offsets_host_ptr: MutPointer[UInt32, _],
    batch_size: Int,
    prompt_lens: List[Int],
    ctx: DeviceContext,
) raises -> Tuple[
    DeviceBuffer[.uint32],
    DeviceBuffer[dtype],
    DeviceBuffer[dtype],
    Int,  # total_length
    Int,  # max_seq_length_batch
]:
    """Initializes input row offsets and hidden state ragged tensor inputs."""
    var total_length = 0
    var max_seq_length_batch = -1
    for i in range(batch_size):
        input_row_offsets_host_ptr[i] = UInt32(total_length)

        var curr_len = prompt_lens[i]
        total_length += curr_len
        if curr_len > max_seq_length_batch:
            max_seq_length_batch = curr_len

    input_row_offsets_host_ptr[batch_size] = UInt32(total_length)
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host_ptr)

    # Initialize ragged hidden state.
    var ragged_size = total_length * hidden_size
    var hidden_state_ragged_host_ptr = alloc[Scalar[dtype]](ragged_size)
    var hidden_state_ragged_host = TileTensor(
        hidden_state_ragged_host_ptr,
        row_major(total_length, Idx[hidden_size]),
    )
    random(hidden_state_ragged_host)

    var hidden_state_ragged_device = ctx.enqueue_create_buffer[dtype](
        ragged_size
    )
    ctx.enqueue_copy(hidden_state_ragged_device, hidden_state_ragged_host_ptr)

    # Initialize padded hidden state.
    var padded_size = batch_size * max_seq_length_batch * hidden_size
    var hidden_state_padded_host_ptr = alloc[Scalar[dtype]](padded_size)

    # Copy over the ragged values to the padded tensor.
    # Only the unpadded rows are checked against the reference output.
    for bs in range(batch_size):
        var unpadded_seq_len = prompt_lens[bs]
        var ragged_start_idx = Int(input_row_offsets_host_ptr[bs])
        for s in range(unpadded_seq_len):
            var padded_ptr = (
                hidden_state_padded_host_ptr
                + (bs * max_seq_length_batch + s) * hidden_size
            )
            var ragged_ptr = (
                hidden_state_ragged_host_ptr
                + (ragged_start_idx + s) * hidden_size
            )
            unsafe_memcpy(dest=padded_ptr, src=ragged_ptr, count=hidden_size)

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
) raises:
    """Tests the KV cache matmul.

    Note that here `prompt_lens` indicates the sequence length of the hidden
    states, although in general the sequence may not originate from a prompt.
    For example, in cross attention the sequence would be from a sequence of
    patch embeddings of an image.
    """
    comptime hidden_size = num_q_heads * kv_params.head_size
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime num_blocks = 32

    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params, MutAnyOrigin, ImmutAnyOrigin, ImmutAnyOrigin
    ]

    assert len(prompt_lens) == len(cache_sizes), (
        "mismatch between cache_sizes and prompt_lens, both should be"
        " batch_size in length"
    )

    var batch_size = len(prompt_lens)

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = alloc[UInt32](batch_size + 1)
    var init_result = _initialize_ragged_inputs[dtype, hidden_size](
        input_row_offsets_host_ptr, batch_size, prompt_lens, ctx
    )
    var input_row_offsets_device = init_result[0]
    var hidden_state_ragged_device = init_result[1]
    var hidden_state_padded_device = init_result[2]
    var total_length = init_result[3]
    var max_seq_length_batch = init_result[4]

    # Initialize the weights.
    var weight_size = 2 * kv_hidden_size * hidden_size
    var weight_host_ptr = alloc[Scalar[dtype]](weight_size)
    var weight_shape = IndexList[2](2 * kv_hidden_size, hidden_size)
    var weight_host = TileTensor(
        weight_host_ptr,
        row_major(Idx[2 * kv_hidden_size], Idx[hidden_size]),
    )
    random(weight_host)

    var weight_device = ctx.enqueue_create_buffer[dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize reference output.
    var padded_batch_dim = batch_size * max_seq_length_batch
    var ref_output_size = padded_batch_dim * 2 * kv_hidden_size
    var ref_output_host_ptr = alloc[Scalar[dtype]](ref_output_size)
    var ref_output_shape = IndexList[2](padded_batch_dim, 2 * kv_hidden_size)
    var ref_output_host = TileTensor(
        ref_output_host_ptr,
        row_major(padded_batch_dim, Idx[2 * kv_hidden_size]),
    )
    var ref_output_device = ctx.enqueue_create_buffer[dtype](ref_output_size)

    # Initialize our KVCache.
    var cache_lengths = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    var cache_lengths_host = cache_lengths.host_tensor().as_unsafe_any_origin()
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
        * kv_params.num_heads
        * kv_params.head_size
    )
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_length_cache,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime BlocksLayout = CollectionType.blocks_tt_layout
    var blocks_shape = Coord[*BlocksLayout.shape_types]()
    blocks_shape[0] = Int64(kv_block_shape[0])
    blocks_shape[2] = Int64(kv_block_shape[2])
    blocks_shape[1] = Int64(kv_block_shape[1])
    blocks_shape[3] = Int64(kv_block_shape[3])
    var blocks_strides = Coord[*BlocksLayout.stride_types]()
    blocks_strides[0] = Int64(
        kv_block_shape[1]
        * kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    blocks_strides[1] = Int64(
        kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    blocks_strides[2] = Int64(
        kv_block_shape[3] * kv_block_shape[4] * kv_block_shape[5]
    )
    var blocks_layout = BlocksLayout(blocks_shape, blocks_strides)
    var kv_block = HostDeviceTileTensor[dtype](blocks_layout, ctx)

    var lookup_table = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    var lookup_table_host = lookup_table.host_tensor().as_unsafe_any_origin()

    # Assign each batch entry a distinct block. `random_ui64` is inclusive, so
    # the original draw range `[0, num_blocks - 1]` is a population of
    # `num_blocks` blocks.
    var lut_blocks = random_distinct(num_blocks, batch_size)
    for idx in range(batch_size):
        lookup_table_host[idx] = UInt32(lut_blocks[idx])

    cache_lengths.to_device()
    lookup_table.to_device()
    var kv_collection_device = CollectionType(
        kv_block.device_tensor().as_unsafe_any_origin(),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lookup_table.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    var kv_collection_host = CollectionType(
        kv_block.host_tensor().as_unsafe_any_origin(),
        cache_lengths.host_tensor().as_imm().as_unsafe_any_origin(),
        lookup_table.host_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    var hidden_state_ragged_tensor = TileTensor(
        hidden_state_ragged_device,
        row_major(total_length, Idx[hidden_size]),
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_device, row_major(Int64(batch_size + 1))
    ).as_imm()
    var weight_device_tensor = TileTensor(
        weight_device,
        row_major(Idx[2 * kv_hidden_size], Idx[hidden_size]),
    )

    # Execute test.
    _matmul_kv_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_tensor.as_imm().as_unsafe_any_origin(),
        input_row_offsets_tensor,
        weight_device_tensor.as_imm().as_unsafe_any_origin(),
        k_cache_device,
        v_cache_device,
        ctx,
    )

    # Execute reference.
    var ref_output_tile = TileTensor(
        ref_output_device, row_major(Coord(ref_output_shape))
    )
    var hidden_state_padded_tile = TileTensor(
        hidden_state_padded_device,
        row_major(padded_batch_dim, hidden_size),
    )
    var weight_tile = weight_device_tensor.reshape(Coord(weight_shape))
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_tile,
        hidden_state_padded_tile,
        weight_tile,
        ctx,
    )

    kv_block.to_host()
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    for bs in range(batch_size):
        var prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(k_dim, kv_params.head_size)
                assert_almost_equal(
                    ref_output_host[bs * max_seq_length_batch + s, k_dim],
                    k_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                    rtol=rtol,
                )

            for v_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(v_dim, kv_params.head_size)
                assert_almost_equal(
                    ref_output_host[
                        bs * max_seq_length_batch + s,
                        kv_hidden_size + v_dim,
                    ],
                    v_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
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
) raises:
    comptime hidden_size = num_q_heads * kv_params.head_size
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size

    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime CollectionType = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var batch_size = len(prompt_lens)
    assert len(prompt_lens) == len(
        cache_sizes
    ), "expected prompt_lens and cache_sizes size to be equal"

    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * kv_params.head_size
    )
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime BlocksLayout = CollectionType.blocks_tt_layout
    var blocks_shape = Coord[*BlocksLayout.shape_types]()
    blocks_shape[0] = Int64(kv_block_shape[0])
    blocks_shape[2] = Int64(kv_block_shape[2])
    var blocks_strides = Coord[*BlocksLayout.stride_types]()
    blocks_strides[0] = Int64(
        kv_block_shape[1]
        * kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    blocks_strides[1] = Int64(
        kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    var blocks_layout = BlocksLayout(blocks_shape, blocks_strides)
    var kv_block = HostDeviceTileTensor[dtype](blocks_layout, ctx)

    var cache_lengths_table = CacheLengthsTable.build(
        prompt_lens, cache_sizes, ctx
    )

    var max_full_context_length = cache_lengths_table.max_full_context_length
    var max_seq_length_batch = cache_lengths_table.max_seq_length_batch

    var paged_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_full_context_length, num_paged_blocks, ctx
    )

    var kv_collection_device = CollectionType(
        kv_block.device_tensor().as_unsafe_any_origin(),
        cache_lengths_table.cache_lengths.device_tile_tensor(),
        paged_lut.device_tile_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)

    var kv_collection_host = CollectionType(
        kv_block.host_tensor().as_unsafe_any_origin(),
        cache_lengths_table.cache_lengths.host_tile_tensor(),
        paged_lut.host_tile_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = alloc[UInt32](batch_size + 1)
    var init_result = _initialize_ragged_inputs[dtype, hidden_size](
        input_row_offsets_host_ptr, batch_size, prompt_lens, ctx
    )
    var input_row_offsets_device = init_result[0]
    var hidden_state_ragged_device = init_result[1]
    var hidden_state_padded_device = init_result[2]
    var ragged_total_length = init_result[3]
    var init_max_seq_length_batch = init_result[4]

    # Initialize the weights.
    var weight_size = kv_hidden_size * hidden_size
    var weight_shape = IndexList[2](kv_hidden_size, hidden_size)
    var weight_host_ptr = alloc[Scalar[dtype]](weight_size)
    var weight_host = TileTensor(
        weight_host_ptr, row_major(Idx[kv_hidden_size], Idx[hidden_size])
    )
    random(weight_host)

    var weight_device = ctx.enqueue_create_buffer[dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize reference output.
    var padded_batch_dim = batch_size * init_max_seq_length_batch
    max_seq_length_batch = init_max_seq_length_batch
    var ref_output_size = padded_batch_dim * kv_hidden_size
    var ref_output_shape = IndexList[2](padded_batch_dim, kv_hidden_size)
    var ref_output_host_ptr = alloc[Scalar[dtype]](ref_output_size)
    var ref_output_host = TileTensor(
        ref_output_host_ptr,
        row_major(padded_batch_dim, Idx[kv_hidden_size]),
    )
    var ref_output_device = ctx.enqueue_create_buffer[dtype](ref_output_size)

    var hidden_state_ragged_tensor = TileTensor(
        hidden_state_ragged_device,
        row_major(ragged_total_length, Idx[hidden_size]),
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_device, row_major(Int64(batch_size + 1))
    ).as_imm()
    var weight_device_tensor = TileTensor(
        weight_device, row_major(Idx[kv_hidden_size], Idx[hidden_size])
    )

    # Execute test.
    _matmul_k_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_tensor.as_imm().as_unsafe_any_origin(),
        input_row_offsets_tensor,
        weight_device_tensor.as_imm().as_unsafe_any_origin(),
        k_cache_device,
        ctx,
    )

    # Execute reference.
    var ref_output_tile = TileTensor(
        ref_output_device, row_major(Coord(ref_output_shape))
    )
    var hidden_state_padded_tile = TileTensor(
        hidden_state_padded_device,
        row_major(padded_batch_dim, hidden_size),
    )
    var weight_tile = weight_device_tensor.reshape(Coord(weight_shape))
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_tile,
        hidden_state_padded_tile,
        weight_tile,
        ctx,
    )

    kv_block.to_host()
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    for bs in range(batch_size):
        var prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(k_dim, kv_params.head_size)
                assert_almost_equal(
                    ref_output_host[bs * max_seq_length_batch + s, k_dim],
                    k_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
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

    # Release the shared metadata buffers.
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
) raises:
    comptime assert cache_t.dtype == dtype, "type mismatch"
    comptime kv_params = cache_t.kv_params
    comptime hidden_size = num_q_heads * kv_params.head_size
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime fused_hidden_size = 2 * kv_hidden_size + hidden_size

    # Allocate host memory and copy from device
    var ref_output_size = ref_output_shape[0] * ref_output_shape[1]
    var ref_output_host_ptr = alloc[Scalar[dtype]](ref_output_size)
    var ref_output_host = TileTensor(
        ref_output_host_ptr,
        row_major(ref_output_shape[0], Idx[fused_hidden_size]),
    )

    var test_output_size = test_output_shape[0] * test_output_shape[1]
    var test_output_host_ptr = alloc[Scalar[dtype]](test_output_size)
    var test_output_host = TileTensor(
        test_output_host_ptr,
        row_major(test_output_shape[0], Idx[hidden_size]),
    )

    ctx.enqueue_copy(test_output_host_ptr, test_output_device)
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    var batch_size = len(prompt_lens)

    var ragged_offset = 0
    for bs in range(batch_size):
        var prompt_len = prompt_lens[bs]
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
                var head_idx, head_dim_idx = udivmod(k_dim, kv_params.head_size)
                try:
                    assert_almost_equal(
                        ref_output_host[
                            bs * max_seq_length_batch + s,
                            hidden_size + k_dim,
                        ],
                        k_cache.load[width=1](
                            bs,
                            head_idx,
                            k_cache.cache_length(bs) + s,
                            head_dim_idx,
                        ).cast[dtype](),
                        rtol=rtol,
                    )
                except e:
                    print("K", bs, s, k_dim)
                    raise e^

            for v_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(v_dim, kv_params.head_size)
                try:
                    assert_almost_equal(
                        ref_output_host[
                            bs * max_seq_length_batch + s,
                            hidden_size + kv_hidden_size + v_dim,
                        ],
                        v_cache.load[width=1](
                            bs,
                            head_idx,
                            v_cache.cache_length(bs) + s,
                            head_dim_idx,
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


# HACK: `k_cache` and `v_cache` use disjoint regions within each block
# (kv_idx 0 vs 1), but share the collection's mutable origins. They are
# only ever stored to at disjoint offsets, but the exclusivity checker cannot
# prove that and rejects passing both as separately-writable arguments. Disable
# the nested-origin exclusivity check as a stopgap; the proper fix is to give the
# k/v views provably-disjoint origins instead of sharing the collection's.
@__unsafe_nested_origins_read_only
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
) raises -> Tuple[
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
    comptime hidden_size = num_q_heads * kv_params.head_size
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime fused_hidden_size = (2 * kv_hidden_size) + hidden_size
    comptime num_blocks = 32

    assert len(prompt_lens) == len(cache_sizes), (
        "mismatch between cache_sizes and prompt_lens, both should be"
        " batch_size in length"
    )

    var batch_size = len(prompt_lens)

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        num_blocks,
        ")",
    )

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = alloc[UInt32](batch_size + 1)
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
    var weight_host_ptr = alloc[Scalar[dtype]](weight_size)
    var weight_host = TileTensor(
        weight_host_ptr,
        row_major(Idx[fused_hidden_size], Idx[hidden_size]),
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

    var hidden_state_ragged_tensor = TileTensor(
        hidden_state_ragged_device,
        row_major(total_length, Idx[hidden_size]),
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_device, row_major(Int64(batch_size + 1))
    ).as_imm()
    var weight_device_tensor = TileTensor(
        weight_device,
        row_major(Idx[fused_hidden_size], Idx[hidden_size]),
    )
    var test_output_device_tensor = TileTensor(
        test_output_device, row_major(total_length, Idx[hidden_size])
    )

    # Execute the matmul
    _fused_qkv_matmul_kv_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_tensor.as_imm().as_unsafe_any_origin(),
        input_row_offsets_tensor,
        weight_device_tensor.as_imm().as_unsafe_any_origin(),
        k_cache,
        v_cache,
        test_output_device_tensor,
        ctx,
    )

    # Execute reference
    var ref_output_tile = TileTensor(
        ref_output_device, row_major(Coord(ref_output_shape))
    )
    var hidden_state_padded_tile = TileTensor(
        hidden_state_padded_device,
        row_major(padded_batch_dim, hidden_size),
    )
    var weight_tile = weight_device_tensor.reshape(Coord(weight_shape))
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_tile,
        hidden_state_padded_tile,
        weight_tile,
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
) raises:
    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime CollectionType = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]

    var batch_size = len(prompt_lens)
    assert len(prompt_lens) == len(
        cache_sizes
    ), "expected prompt_lens and cache_sizes size to be equal"

    var cache_lengths_host_ptr = alloc[UInt32](batch_size)

    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * kv_params.head_size
    )
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime BlocksLayout = CollectionType.blocks_tt_layout
    var blocks_shape = Coord[*BlocksLayout.shape_types]()
    blocks_shape[0] = Int64(kv_block_shape[0])
    blocks_shape[2] = Int64(kv_block_shape[2])
    var blocks_strides = Coord[*BlocksLayout.stride_types]()
    blocks_strides[0] = Int64(
        kv_block_shape[1]
        * kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    blocks_strides[1] = Int64(
        kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    var blocks_layout = BlocksLayout(blocks_shape, blocks_strides)
    var kv_block = HostDeviceTileTensor[dtype](blocks_layout, ctx)

    var cache_lengths_table = CacheLengthsTable.build(
        prompt_lens, cache_sizes, ctx
    )

    var max_full_context_length = cache_lengths_table.max_full_context_length
    var max_seq_length_batch = cache_lengths_table.max_seq_length_batch

    var paged_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_full_context_length, num_paged_blocks, ctx
    )

    var kv_collection_device = CollectionType(
        kv_block.device_tensor().as_unsafe_any_origin(),
        cache_lengths_table.cache_lengths.device_tile_tensor(),
        paged_lut.device_tile_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    var kv_collection_host = CollectionType(
        kv_block.host_tensor().as_unsafe_any_origin(),
        cache_lengths_table.cache_lengths.host_tile_tensor(),
        paged_lut.host_tile_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
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

    kv_block.to_host()

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

    # Release the shared metadata buffers.
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
) raises:
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params, MutAnyOrigin, ImmutAnyOrigin, ImmutAnyOrigin
    ]

    assert len(prompt_lens) == len(cache_sizes), (
        "mismatch between cache_sizes and prompt_lens, both should be"
        " batch_size in length"
    )

    # Initialize our KVCache
    var batch_size = len(cache_sizes)
    var cache_lengths_host_ptr = alloc[UInt32](batch_size)
    var max_seq_length_batch = -1
    var max_context_length = 0

    for i in range(batch_size):
        cache_lengths_host_ptr[i] = UInt32(cache_sizes[i])
        max_context_length = max(
            max_context_length, cache_sizes[i] + prompt_lens[i]
        )
        if prompt_lens[i] > max_seq_length_batch:
            max_seq_length_batch = prompt_lens[i]

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host_ptr)

    var kv_block_size = (
        num_blocks
        * 2
        * num_layers
        * max_seq_length_cache
        * kv_params.num_heads
        * kv_params.head_size
    )
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_length_cache,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var kv_block_host_ptr = alloc[Scalar[dtype]](kv_block_size)
    var kv_block_device = ctx.enqueue_create_buffer[dtype](kv_block_size)

    var lookup_table_host_ptr = alloc[UInt32](batch_size)

    # Assign each batch entry a distinct block. `random_ui64` is inclusive, so
    # the original draw range `[0, num_blocks - 1]` is a population of
    # `num_blocks` blocks.
    var lut_blocks = random_distinct(num_blocks, batch_size)
    for idx in range(batch_size):
        lookup_table_host_ptr[idx] = UInt32(lut_blocks[idx])

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host_ptr)

    comptime BlocksLayout = CollectionType.blocks_tt_layout
    var blocks_shape = Coord[*BlocksLayout.shape_types]()
    blocks_shape[0] = Int64(kv_block_shape[0])
    blocks_shape[2] = Int64(kv_block_shape[2])
    blocks_shape[1] = Int64(kv_block_shape[1])
    blocks_shape[3] = Int64(kv_block_shape[3])
    var blocks_strides = Coord[*BlocksLayout.stride_types]()
    blocks_strides[0] = Int64(
        kv_block_shape[1]
        * kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    blocks_strides[1] = Int64(
        kv_block_shape[2]
        * kv_block_shape[3]
        * kv_block_shape[4]
        * kv_block_shape[5]
    )
    blocks_strides[2] = Int64(
        kv_block_shape[3] * kv_block_shape[4] * kv_block_shape[5]
    )
    var blocks_layout = BlocksLayout(blocks_shape, blocks_strides)
    var kv_blocks_device_view = TileTensor(kv_block_device, blocks_layout)
    var cache_lengths_device_view = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lookup_table_device_view = TileTensor(
        lookup_table_device, row_major(Int64(batch_size))
    )
    var kv_collection_device = CollectionType(
        kv_blocks_device_view.as_unsafe_any_origin(),
        cache_lengths_device_view.as_imm().as_unsafe_any_origin(),
        lookup_table_device_view.as_imm().as_unsafe_any_origin(),
        UInt32(max_seq_length_batch),
        UInt32(max_context_length),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    var kv_blocks_host_view = TileTensor(kv_block_host_ptr, blocks_layout)
    var cache_lengths_host_view = TileTensor(
        cache_lengths_host_ptr, row_major(Int64(batch_size))
    )
    var lookup_table_host_view = TileTensor(
        lookup_table_host_ptr, row_major(Int64(batch_size))
    )
    var kv_collection_host = CollectionType(
        kv_blocks_host_view.as_unsafe_any_origin(),
        cache_lengths_host_view.as_imm().as_unsafe_any_origin(),
        lookup_table_host_view.as_imm().as_unsafe_any_origin(),
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


def execute_fused_matmul_suite[
    dtype: DType, rtol: Float64
](ctx: DeviceContext) raises:
    comptime test_kernel = get_defined_string["test_kernel", "all"]()

    for bs in [1, 16]:
        var ce_cache_sizes = List[Int]()
        var ce_seq_lens = List[Int]()
        var tg_cache_sizes = List[Int]()
        var tg_seq_lens = List[Int]()
        for _ in range(bs):
            tg_seq_lens.append(1)
            # TODO increase sizes here to ensure we cross page boundary.
            tg_cache_sizes.append(Int(random_ui64(512, 700)))
            ce_seq_lens.append(Int(random_ui64(512, 700)))
            ce_cache_sizes.append(0)

        # llama3 context encoding
        comptime if test_kernel == "all" or test_kernel == "fused_cont":
            execute_cont_batch_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)
        comptime if test_kernel == "all" or test_kernel == "fused_paged":
            execute_paged_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)
        comptime if test_kernel == "all" or test_kernel == "kv_cont":
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
        comptime if test_kernel == "all" or test_kernel == "k_paged":
            execute_matmul_k_cache_ragged[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)

        # llama3 token gen
        comptime if test_kernel == "all" or test_kernel == "fused_cont":
            execute_cont_batch_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)
        comptime if test_kernel == "all" or test_kernel == "fused_paged":
            execute_paged_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)
        comptime if test_kernel == "all" or test_kernel == "kv_cont":
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
        comptime if test_kernel == "all" or test_kernel == "k_paged":
            execute_matmul_k_cache_ragged[
                llama_num_q_heads, dtype, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)


def main() raises:
    seed(42)

    comptime test_dtype = get_defined_string["test_dtype", "all"]()
    comptime assert (
        test_dtype == "all"
        or test_dtype == "bfloat16"
        or test_dtype == "float32"
    ), "test_dtype must be one of: all, bfloat16, float32"

    comptime test_kernel = get_defined_string["test_kernel", "all"]()
    comptime assert (
        test_kernel == "all"
        or test_kernel == "fused_cont"
        or test_kernel == "fused_paged"
        or test_kernel == "kv_cont"
        or test_kernel == "k_paged"
    ), (
        "test_kernel must be one of: all, fused_cont, fused_paged, kv_cont,"
        " k_paged"
    )

    with DeviceContext() as ctx:
        comptime if test_dtype == "all" or test_dtype == "float32":
            execute_fused_matmul_suite[.float32, 1e-3](ctx)
        comptime if test_dtype == "all" or test_dtype == "bfloat16":
            execute_fused_matmul_suite[.bfloat16, 1e-2](ctx)
