# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from buffer import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from memory import memcpy
from nn.kv_cache_ragged import (
    _matmul_k_cache_ragged_scale_impl,
)
from testing import assert_almost_equal, assert_equal

from utils import IndexList

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


alias block_scale = 128


def _initialize_ragged_inputs[
    dtype: DType, hidden_size: Int
](
    mut input_row_offsets_host: HostNDBuffer[DType.uint32, 1],
    batch_size: Int,
    prompt_lens: List[Int],
    ctx: DeviceContext,
) -> (
    DeviceNDBuffer[DType.uint32, 1],
    DeviceNDBuffer[dtype, 2, DimList(Dim(), hidden_size)],
    DeviceNDBuffer[dtype, 2, DimList(Dim(), hidden_size)],
):
    """Initializes input row offsets and hidden state ragged tensor inputs."""
    var total_length = 0
    var max_seq_length_batch = -1
    for i in range(batch_size):
        input_row_offsets_host.tensor[i] = total_length

        var curr_len = prompt_lens[i]
        total_length += curr_len
        if curr_len > max_seq_length_batch:
            max_seq_length_batch = curr_len

    input_row_offsets_host.tensor[batch_size] = total_length
    var input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)

    # Initialize ragged hidden state.
    var hidden_state_ragged_host = HostNDBuffer[
        dtype, 2, DimList(Dim(), hidden_size)
    ](IndexList[2](total_length, hidden_size))

    random(hidden_state_ragged_host.tensor)

    var hidden_state_ragged_device = hidden_state_ragged_host.copy_to_device(
        ctx
    )

    # Initialize padded hidden state.
    var hidden_state_padded_host = HostNDBuffer[
        dtype, 2, DimList(Dim(), hidden_size)
    ](IndexList[2](batch_size * max_seq_length_batch, hidden_size))

    # Copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        var unpadded_seq_len = prompt_lens[bs]
        var ragged_start_idx = Int(input_row_offsets_host.tensor[bs])
        for s in range(unpadded_seq_len):
            var padded_ptr = hidden_state_padded_host.tensor._offset(
                IndexList[2](bs * max_seq_length_batch + s, 0)
            )
            var ragged_ptr = hidden_state_ragged_host.tensor._offset(
                IndexList[2](ragged_start_idx + s, 0)
            )
            memcpy(dest=padded_ptr, src=ragged_ptr, count=hidden_size)

    var hidden_state_padded_device = hidden_state_padded_host.copy_to_device(
        ctx
    )

    # Sync here so that HtoD transfers complete prior to host buffer dtor.
    ctx.synchronize()

    _ = hidden_state_ragged_host^
    _ = hidden_state_padded_host^

    return (
        input_row_offsets_device,
        hidden_state_ragged_device,
        hidden_state_padded_device,
    )


def execute_matmul_k_cache_ragged_scale[
    num_q_heads: Int,
    dtype: DType,
    weight_dtype: DType,
    scale_dtype: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
    atol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    """Tests the scaled KV cache matmul for key projections.

    This test follows the same pattern as execute_matmul_k_cache_ragged but
    includes input_scale and weight_scale parameters for scaled FP8 operations.
    """
    alias hidden_size = num_q_heads * kv_params.head_size
    alias kv_hidden_size = kv_params.num_heads * kv_params.head_size

    alias num_paged_blocks = 32
    alias page_size = 512
    alias CollectionType = PagedKVCacheCollection[dtype, kv_params, page_size]
    var batch_size = len(prompt_lens)

    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        "expected prompt_lens and cache_sizes size to be equal",
    )

    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var kv_block_host = HostNDBuffer[dtype, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    var total_length = 0
    var max_full_context_length = 0
    var max_seq_length_batch = 0
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = cache_sizes[i]
        max_full_context_length = max(
            max_full_context_length, Int(cache_sizes[i] + prompt_lens[i])
        )
        max_seq_length_batch = max(max_seq_length_batch, prompt_lens[i])
        total_length += prompt_lens[i]

    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    var paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )

    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_sizes[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval

    var paged_lut_device = paged_lut_host.copy_to_device(ctx)
    var kv_block_device = kv_block_host.copy_to_device(ctx)

    var kv_collection_device = CollectionType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_seq_length_batch,
        max_full_context_length,
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)

    var kv_collection_host = CollectionType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        paged_lut_host.tensor,
        max_seq_length_batch,
        max_full_context_length,
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        DimList(batch_size + 1)
    )
    var input_row_offsets_device, hidden_state_ragged_device, hidden_state_padded_device = _initialize_ragged_inputs[
        weight_dtype, hidden_size
    ](
        input_row_offsets_host, batch_size, prompt_lens, ctx
    )

    # Initialize the weights.
    var weight_host = HostNDBuffer[
        weight_dtype, 2, DimList(kv_hidden_size, hidden_size)
    ](IndexList[2](kv_hidden_size, hidden_size))
    random(weight_host.tensor)
    var weight_device = weight_host.copy_to_device(ctx)

    # Initialize scales for blockwise scaling.
    alias static_weight_scale_shape = DimList(
        ceildiv(kv_hidden_size, block_scale), ceildiv(hidden_size, block_scale)
    )
    var dynamic_input_scale_shape = DimList(
        ceildiv(hidden_size, block_scale),
        hidden_state_ragged_device.tensor.dim(0),
    )
    var dynamic_weight_scale_shape = DimList(
        ceildiv(kv_hidden_size, block_scale), ceildiv(hidden_size, block_scale)
    )

    var input_scale_host = HostNDBuffer[scale_dtype, 2](
        dynamic_input_scale_shape
    )
    var weight_scale_host = HostNDBuffer[
        scale_dtype, 2, static_weight_scale_shape
    ](dynamic_weight_scale_shape)

    random(input_scale_host.tensor)
    random(weight_scale_host.tensor)

    var input_scale_device = input_scale_host.copy_to_device(ctx)
    var weight_scale_device = weight_scale_host.copy_to_device(ctx)

    # Initialize reference output.
    var ref_output_host = HostNDBuffer[
        dtype, 2, DimList(Dim(), kv_hidden_size)
    ](IndexList[2](hidden_state_ragged_device.tensor.dim(0), kv_hidden_size))
    var ref_output_device = ref_output_host.copy_to_device(ctx)

    # Execute test with scaled implementation.
    _matmul_k_cache_ragged_scale_impl[
        target="gpu",
        scales_granularity_mnk = IndexList[3](1, block_scale, block_scale),
    ](
        hidden_state_ragged_device.tensor,
        input_row_offsets_device.tensor,
        weight_device.tensor,
        input_scale_device.tensor,
        weight_scale_device.tensor,
        k_cache_device,
        ctx,
    )

    # Execute reference using naive blockwise scaled matmul.
    # Convert weight to same dtype as input for reference computation.
    var weight_ref_host = HostNDBuffer[
        weight_dtype, 2, DimList(kv_hidden_size, hidden_size)
    ](IndexList[2](kv_hidden_size, hidden_size))
    var weight_ref_device = weight_ref_host.copy_to_device(ctx)

    # Copy weight data
    ctx.enqueue_copy(weight_ref_host.tensor.data, weight_device.buffer)
    ctx.synchronize()
    ctx.enqueue_copy(weight_ref_device.buffer, weight_ref_host.tensor.data)

    # Create scale tensors for reference computation
    var ref_input_scale_host = HostNDBuffer[scale_dtype, 2](
        dynamic_input_scale_shape
    )
    var ref_weight_scale_host = HostNDBuffer[
        scale_dtype, 2, static_weight_scale_shape
    ](dynamic_weight_scale_shape)

    # Fill with the same scale values
    for i in range(input_scale_host.tensor.dim(0)):
        for j in range(input_scale_host.tensor.dim(1)):
            ref_input_scale_host.tensor[i, j] = input_scale_host.tensor[i, j]
    for i in range(weight_scale_host.tensor.dim(0)):
        for j in range(weight_scale_host.tensor.dim(1)):
            ref_weight_scale_host.tensor[i, j] = weight_scale_host.tensor[i, j]

    var ref_input_scale_device = ref_input_scale_host.copy_to_device(ctx)
    var ref_weight_scale_device = ref_weight_scale_host.copy_to_device(ctx)

    # Use naive blockwise scaled matmul as reference
    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk = IndexList[3](1, block_scale, block_scale),
    ](
        ref_output_device.tensor,
        hidden_state_ragged_device.tensor,
        weight_ref_device.tensor,
        ref_input_scale_device.tensor,
        ref_weight_scale_device.tensor,
        ctx,
    )

    ctx.enqueue_copy(kv_block_host.tensor.data, kv_block_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    # Verify results
    var ref_out = ref_output_host.tensor
    for bs in range(batch_size):
        var prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                var head_idx = k_dim // kv_params.head_size
                var head_dim_idx = k_dim % kv_params.head_size
                var a = ref_out[
                    Int(input_row_offsets_host.tensor[bs]) + s, k_dim
                ]
                var b = k_cache_host.load[width=1](
                    bs,
                    head_idx,
                    cache_sizes[bs] + s,
                    head_dim_idx,
                )
                assert_almost_equal(a, b, atol=atol, rtol=rtol)

    # Cleanup
    _ = hidden_state_ragged_device^
    _ = hidden_state_padded_device^
    _ = weight_host^
    _ = weight_device^
    _ = weight_ref_host^
    _ = weight_ref_device^
    _ = input_scale_host^
    _ = input_scale_device^
    _ = weight_scale_host^
    _ = weight_scale_device^
    _ = ref_input_scale_host^
    _ = ref_input_scale_device^
    _ = ref_weight_scale_host^
    _ = ref_weight_scale_device^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = kv_block_device^
    _ = kv_block_host^
    _ = paged_lut_host^
    _ = paged_lut_device^
    _ = cache_lengths_device^
    _ = cache_lengths_host^
    _ = input_row_offsets_device^
    _ = input_row_offsets_host^


def execute_fused_matmul_suite_float8_e4m3fn(ctx: DeviceContext):
    """Test suite specifically for FP8 scaled matmul operations."""
    alias dtype = DType.float8_e4m3fn
    alias rtol = 1e-2
    alias atol = 1e-2
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

        # Context encoding test
        execute_matmul_k_cache_ragged_scale[
            llama_num_q_heads,
            DType.float32,
            dtype,
            dtype,
            kv_params_llama3,
            rtol,
            atol,
        ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)

        # Token generation test
        execute_matmul_k_cache_ragged_scale[
            llama_num_q_heads,
            DType.float32,
            dtype,
            dtype,
            kv_params_llama3,
            rtol,
            atol,
        ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite_float8_e4m3fn(ctx)
