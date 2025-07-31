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


from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    random,
)
from random import random_float64
from linalg.grouped_matmul import naive_grouped_matmul
from testing import assert_almost_equal
from math import ceildiv
from utils import IndexList
from utils.index import Index
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.kv_cache_ragged import (
    k_grouped_matmul_ragged_paged,
    v_grouped_matmul_ragged_paged,
)


fn test_kv_grouped_matmul[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    kv_params: KVCacheStaticParams,
    page_size: Int,
    test_k: Bool,  # True for K, False for V
](
    num_sequences: Int,  # Number of sequences in the batch
    tokens_per_sequence: List[Int],  # Tokens for each sequence
    sequence_to_expert: List[Int],  # Which expert (LoRA) each sequence uses
    ctx: DeviceContext,
) raises:
    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    alias N = expert_shape[0]  # output_dim = num_heads * head_size
    alias K = expert_shape[1]  # input_dim

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_per_sequence = 0
    for i in range(num_sequences):
        total_num_tokens += tokens_per_sequence[i]
        max_num_tokens_per_sequence = max(
            max_num_tokens_per_sequence, tokens_per_sequence[i]
        )

    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_ref_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_sequences + 1
    )

    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_sequences)

    # Setup offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_sequences):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + tokens_per_sequence[i]
        )
        expert_ids_host.tensor[i] = sequence_to_expert[i]

    random(a_host.tensor)
    random(b_host.tensor)

    var a_dev = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var c_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_ref_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_sequences + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.int32, 1](num_sequences, ctx=ctx)

    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_dev.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)

    # Set up KV cache collection
    # In LoRA context: batch_size = number of sequences
    var batch_size = num_sequences
    alias layer_idx = 0
    alias num_layers = 1
    var num_blocks = ceildiv(total_num_tokens, page_size) * 2
    alias max_prompt_len = 128
    alias max_context_len = 256

    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](batch_size)
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = 0
    var cache_lengths_dev = cache_lengths_host.copy_to_device(ctx)

    var kv_block_host = HostNDBuffer[c_type, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    for i in range(kv_block_host.tensor.num_elements()):
        kv_block_host.tensor.data[i] = 0.0

    var kv_block_dev = kv_block_host.copy_to_device(ctx)

    var lookup_table_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_context_len, page_size))
    )

    var block_counter = 0
    for i in range(batch_size):
        for j in range(ceildiv(tokens_per_sequence[i], page_size)):
            lookup_table_host.tensor[i, j] = block_counter
            block_counter += 1
    var lookup_table_dev = lookup_table_host.copy_to_device(ctx)

    alias CollectionType = PagedKVCacheCollection[c_type, kv_params, page_size]
    var kv_collection_device = CollectionType(
        kv_block_dev.tensor,
        cache_lengths_dev.tensor,
        lookup_table_dev.tensor,
        max_prompt_len,
        max_context_len,
    )

    var kv_collection_host = CollectionType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        max_prompt_len,
        max_context_len,
    )

    naive_grouped_matmul(
        c_ref_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_per_sequence,
        num_sequences,
        ctx,
    )

    if test_k:
        k_grouped_matmul_ragged_paged[a_type, "gpu"](
            a_dev.tensor,
            b_dev.tensor,
            a_offsets_dev.tensor,
            expert_ids_dev.tensor,
            max_num_tokens_per_sequence,
            num_sequences,
            kv_collection_device,
            UInt32(layer_idx),
            ctx,
        )
    else:
        v_grouped_matmul_ragged_paged[a_type, "gpu"](
            a_dev.tensor,
            b_dev.tensor,
            a_offsets_dev.tensor,
            expert_ids_dev.tensor,
            max_num_tokens_per_sequence,
            num_sequences,
            kv_collection_device,
            UInt32(layer_idx),
            ctx,
        )

    ctx.enqueue_copy(kv_block_host.tensor.data, kv_block_dev.buffer)
    ctx.enqueue_copy(c_ref_host.tensor.data, c_ref_dev.buffer)
    ctx.synchronize()

    var total_values = (
        total_num_tokens * kv_params.num_heads * kv_params.head_size
    )
    rtol = 1e-2

    var token_offset = 0
    for seq_idx in range(num_sequences):
        var expert_id = sequence_to_expert[seq_idx]
        var num_tokens = tokens_per_sequence[seq_idx]

        for token_idx in range(num_tokens):
            for head_idx in range(kv_params.num_heads):
                for dim_idx in range(kv_params.head_size):
                    var ref_idx_m = token_offset + token_idx
                    var ref_idx_n = head_idx * kv_params.head_size + dim_idx
                    var ref_value = c_ref_host.tensor[ref_idx_m, ref_idx_n]

                    var cache_idx = token_idx
                    var block_idx = cache_idx // page_size
                    var block_offset = cache_idx % page_size
                    var lookup_idx = lookup_table_host.tensor[
                        seq_idx, block_idx
                    ]
                    var kv_idx = 0 if test_k else 1  # 0 for K, 1 for V
                    var actual_value = kv_block_host.tensor[
                        Int(lookup_idx),
                        kv_idx,
                        layer_idx,
                        block_offset,
                        head_idx,
                        dim_idx,
                    ]

                    assert_almost_equal(
                        actual_value,
                        ref_value,
                        msg=String(
                            "seq_idx: ",
                            seq_idx,
                            " token_idx: ",
                            token_idx,
                            " head_idx: ",
                            head_idx,
                            " dim_idx: ",
                            dim_idx,
                        ),
                        rtol=rtol,
                    )

        token_offset += num_tokens


fn test_lora_zero_weights_preserves_base_cache[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    kv_params: KVCacheStaticParams,
    page_size: Int,
    test_k: Bool,  # True for K, False for V
](
    num_sequences: Int,  # Number of sequences in the batch
    tokens_per_sequence: List[Int],  # Tokens for each sequence
    sequence_to_expert: List[Int],  # Which expert (LoRA) each sequence uses
    ctx: DeviceContext,
) raises:
    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    alias N = expert_shape[0]  # output_dim = num_heads * head_size
    alias K = expert_shape[1]  # input_dim

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_per_sequence = 0
    for i in range(num_sequences):
        total_num_tokens += tokens_per_sequence[i]
        max_num_tokens_per_sequence = max(
            max_num_tokens_per_sequence, tokens_per_sequence[i]
        )

    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_ref_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_sequences + 1
    )

    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_sequences)

    # Setup offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_sequences):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + tokens_per_sequence[i]
        )
        expert_ids_host.tensor[i] = sequence_to_expert[i]

    random(a_host.tensor)
    # Set LoRA weights (B buffer) to all zeros to test zero adaptation
    for i in range(b_host.tensor.num_elements()):
        b_host.tensor.data[i] = 0.0

    var a_dev = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var c_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_ref_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_sequences + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.int32, 1](num_sequences, ctx=ctx)

    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_dev.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)

    # Set up KV cache collection
    # In LoRA context: batch_size = number of sequences
    var batch_size = num_sequences
    alias layer_idx = 0
    alias num_layers = 1
    var num_blocks = ceildiv(total_num_tokens, page_size) * 2
    alias max_prompt_len = 128
    alias max_context_len = 256

    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](batch_size)
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = 0
    var cache_lengths_dev = cache_lengths_host.copy_to_device(ctx)

    var kv_block_host = HostNDBuffer[c_type, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    # Initialize KV cache to simulate base model values
    for i in range(kv_block_host.tensor.num_elements()):
        kv_block_host.tensor.data[i] = 1.0

    var kv_block_dev = kv_block_host.copy_to_device(ctx)

    var lookup_table_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_context_len, page_size))
    )

    var block_counter = 0
    for i in range(batch_size):
        for j in range(ceildiv(tokens_per_sequence[i], page_size)):
            lookup_table_host.tensor[i, j] = block_counter
            block_counter += 1
    var lookup_table_dev = lookup_table_host.copy_to_device(ctx)

    alias CollectionType = PagedKVCacheCollection[c_type, kv_params, page_size]
    var kv_collection_device = CollectionType(
        kv_block_dev.tensor,
        cache_lengths_dev.tensor,
        lookup_table_dev.tensor,
        max_prompt_len,
        max_context_len,
    )

    var kv_collection_host = CollectionType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        max_prompt_len,
        max_context_len,
    )

    # Get cache objects for proper access
    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)
    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # Reference computation: A × B where B=0 should produce zero LoRA delta
    naive_grouped_matmul(
        c_ref_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_per_sequence,
        num_sequences,
        ctx,
    )

    if test_k:
        k_grouped_matmul_ragged_paged[a_type, "gpu"](
            a_dev.tensor,
            b_dev.tensor,
            a_offsets_dev.tensor,
            expert_ids_dev.tensor,
            max_num_tokens_per_sequence,
            num_sequences,
            kv_collection_device,
            UInt32(layer_idx),
            ctx,
        )
    else:
        v_grouped_matmul_ragged_paged[a_type, "gpu"](
            a_dev.tensor,
            b_dev.tensor,
            a_offsets_dev.tensor,
            expert_ids_dev.tensor,
            max_num_tokens_per_sequence,
            num_sequences,
            kv_collection_device,
            UInt32(layer_idx),
            ctx,
        )

    ctx.enqueue_copy(kv_block_host.tensor.data, kv_block_dev.buffer)
    ctx.enqueue_copy(c_ref_host.tensor.data, c_ref_dev.buffer)
    ctx.synchronize()

    # Verify reference LoRA delta is zero (since LoRA weights are zero)
    for i in range(c_ref_host.tensor.num_elements()):
        assert_almost_equal(
            c_ref_host.tensor.data[i],
            0.0,
            msg="LoRA delta should be zero when LoRA weights are zero",
            rtol=1e-6,
        )

    var token_offset = 0
    for seq_idx in range(num_sequences):
        var expert_id = sequence_to_expert[seq_idx]
        var num_tokens = tokens_per_sequence[seq_idx]

        for token_idx in range(num_tokens):
            for head_idx in range(kv_params.num_heads):
                for dim_idx in range(kv_params.head_size):
                    var ref_idx_m = token_offset + token_idx
                    var ref_idx_n = head_idx * kv_params.head_size + dim_idx
                    var ref_value = c_ref_host.tensor[ref_idx_m, ref_idx_n]

                    # Access KV cache value using proper cache methods
                    var cache_position = (
                        Int(cache_lengths_host.tensor[seq_idx]) + token_idx
                    )
                    var cache_value: Scalar[out_type]
                    if test_k:
                        cache_value = k_cache_host.load[width=1](
                            seq_idx,
                            head_idx,
                            cache_position,
                            dim_idx,
                        )
                    else:
                        cache_value = v_cache_host.load[width=1](
                            seq_idx,
                            head_idx,
                            cache_position,
                            dim_idx,
                        )

                    # Expected: base_value + lora_delta = 1.0 + 0.0 = 1.0
                    var expected_cache_value = (
                        1.0 + ref_value
                    )  # base + LoRA delta
                    assert_almost_equal(
                        cache_value,
                        expected_cache_value,
                        msg=String(
                            "LoRA zero weights test - seq_idx: ",
                            seq_idx,
                            " token_idx: ",
                            token_idx,
                            " head_idx: ",
                            head_idx,
                            " dim_idx: ",
                            dim_idx,
                            (
                                " - Cache should preserve base value when LoRA"
                                " weights are zero"
                            ),
                        ),
                        rtol=1e-6,
                    )

        token_offset += num_tokens


fn test_lora_mixed_zero_nonzero_weights[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    kv_params: KVCacheStaticParams,
    page_size: Int,
    test_k: Bool,  # True for K, False for V
](
    num_sequences: Int,  # Number of sequences in the batch
    tokens_per_sequence: List[Int],  # Tokens for each sequence
    sequence_to_expert: List[Int],  # Which expert (LoRA) each sequence uses
    ctx: DeviceContext,
) raises:
    """Test that sequences using LoRA with values modify cache, while sequences using zero LoRA don't.
    """
    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    alias N = expert_shape[0]  # output_dim = num_heads * head_size
    alias K = expert_shape[1]  # input_dim

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_per_sequence = 0
    for i in range(num_sequences):
        total_num_tokens += tokens_per_sequence[i]
        max_num_tokens_per_sequence = max(
            max_num_tokens_per_sequence, tokens_per_sequence[i]
        )

    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_ref_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_sequences + 1
    )

    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_sequences)

    # Setup offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_sequences):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + tokens_per_sequence[i]
        )
        expert_ids_host.tensor[i] = sequence_to_expert[i]

    random(a_host.tensor)

    # Set up mixed LoRA weights: expert 0 has random values, expert 1 has zeros
    for expert_idx in range(num_experts):
        if expert_idx == 0:
            # First expert has random non-zero LoRA weights
            for n in range(N):
                for k in range(K):
                    b_host.tensor[expert_idx, n, k] = random_float64(
                        -0.5, 0.5
                    ).cast[b_type]()
        else:
            # Other experts have zero LoRA weights
            for n in range(N):
                for k in range(K):
                    b_host.tensor[expert_idx, n, k] = 0.0

    var a_dev = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var c_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_ref_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_sequences + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.int32, 1](num_sequences, ctx=ctx)

    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_dev.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)

    # Set up KV cache collection
    var batch_size = num_sequences
    alias layer_idx = 0
    alias num_layers = 1
    var num_blocks = ceildiv(total_num_tokens, page_size) * 2
    alias max_prompt_len = 128
    alias max_context_len = 256

    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](batch_size)
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = 0
    var cache_lengths_dev = cache_lengths_host.copy_to_device(ctx)

    var kv_block_host = HostNDBuffer[c_type, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    # Initialize KV cache to zeros (no base model values)
    for i in range(kv_block_host.tensor.num_elements()):
        kv_block_host.tensor.data[i] = 0.1

    var kv_block_dev = kv_block_host.copy_to_device(ctx)

    var lookup_table_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_context_len, page_size))
    )

    var block_counter = 0
    for i in range(batch_size):
        for j in range(ceildiv(tokens_per_sequence[i], page_size)):
            lookup_table_host.tensor[i, j] = block_counter
            block_counter += 1
    var lookup_table_dev = lookup_table_host.copy_to_device(ctx)

    alias CollectionType = PagedKVCacheCollection[c_type, kv_params, page_size]
    var kv_collection_device = CollectionType(
        kv_block_dev.tensor,
        cache_lengths_dev.tensor,
        lookup_table_dev.tensor,
        max_prompt_len,
        max_context_len,
    )

    var kv_collection_host = CollectionType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        max_prompt_len,
        max_context_len,
    )

    # Get cache objects for proper access
    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)
    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # Reference computation: A × B for mixed weights
    naive_grouped_matmul(
        c_ref_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_per_sequence,
        num_sequences,
        ctx,
    )

    if test_k:
        k_grouped_matmul_ragged_paged[a_type, "gpu"](
            a_dev.tensor,
            b_dev.tensor,
            a_offsets_dev.tensor,
            expert_ids_dev.tensor,
            max_num_tokens_per_sequence,
            num_sequences,
            kv_collection_device,
            UInt32(layer_idx),
            ctx,
        )
    else:
        v_grouped_matmul_ragged_paged[a_type, "gpu"](
            a_dev.tensor,
            b_dev.tensor,
            a_offsets_dev.tensor,
            expert_ids_dev.tensor,
            max_num_tokens_per_sequence,
            num_sequences,
            kv_collection_device,
            UInt32(layer_idx),
            ctx,
        )

    ctx.enqueue_copy(kv_block_host.tensor.data, kv_block_dev.buffer)
    ctx.enqueue_copy(c_ref_host.tensor.data, c_ref_dev.buffer)
    ctx.synchronize()

    var token_offset = 0
    for seq_idx in range(num_sequences):
        var expert_id = sequence_to_expert[seq_idx]
        var num_tokens = tokens_per_sequence[seq_idx]

        for token_idx in range(num_tokens):
            for head_idx in range(kv_params.num_heads):
                for dim_idx in range(kv_params.head_size):
                    var ref_idx_m = token_offset + token_idx
                    var ref_idx_n = head_idx * kv_params.head_size + dim_idx
                    var ref_value = c_ref_host.tensor[ref_idx_m, ref_idx_n]

                    # Access KV cache value using proper cache methods
                    var cache_position = (
                        Int(cache_lengths_host.tensor[seq_idx]) + token_idx
                    )
                    var cache_value: Scalar[out_type]
                    if test_k:
                        cache_value = k_cache_host.load[width=1](
                            seq_idx,
                            head_idx,
                            cache_position,
                            dim_idx,
                        )
                    else:
                        cache_value = v_cache_host.load[width=1](
                            seq_idx,
                            head_idx,
                            cache_position,
                            dim_idx,
                        )

                    # Expected: base_value + lora_delta = 0.0 + lora_delta
                    var expected_cache_value = (
                        0.1 + ref_value
                    )  # base (zero) + LoRA delta

                    var test_description: String
                    if expert_id == 0:
                        test_description = (
                            "Non-zero LoRA should modify zero cache"
                        )
                    else:
                        test_description = (
                            "Zero LoRA should preserve zero cache"
                        )

                    assert_almost_equal(
                        cache_value,
                        expected_cache_value,
                        msg=String(
                            "Mixed LoRA test - seq_idx: ",
                            seq_idx,
                            " (expert ",
                            expert_id,
                            ") token_idx: ",
                            token_idx,
                            " head_idx: ",
                            head_idx,
                            " dim_idx: ",
                            dim_idx,
                            " - ",
                            test_description,
                        ),
                        rtol=1e-6,
                    )

        token_offset += num_tokens


fn main() raises:
    with DeviceContext() as ctx:
        # Test K grouped matmul - Single matmul
        test_kv_grouped_matmul[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(8, 16),
            kv_params = KVCacheStaticParams(num_heads=2, head_size=4),
            page_size=8,
            test_k=True,
        ](1, List[Int](16), List[Int](0), ctx)

        # Test V grouped matmul - Single matmul
        test_kv_grouped_matmul[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(8, 16),
            kv_params = KVCacheStaticParams(num_heads=2, head_size=4),
            page_size=8,
            test_k=False,
        ](1, List[Int](16), List[Int](0), ctx)

        # Test K grouped matmul - Multiple sequences using different LoRAs
        test_kv_grouped_matmul[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(16, 32),
            kv_params = KVCacheStaticParams(num_heads=4, head_size=4),
            page_size=16,
            test_k=True,
        ](2, List[Int](16, 16), List[Int](1, 0), ctx)

        # Test V grouped matmul - Multiple sequences using different LoRAs
        test_kv_grouped_matmul[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(16, 32),
            kv_params = KVCacheStaticParams(num_heads=4, head_size=4),
            page_size=16,
            test_k=False,
        ](2, List[Int](16, 16), List[Int](0, 1), ctx)

        # Test K grouped matmul - Non-uniform token counts
        test_kv_grouped_matmul[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(16, 32),
            kv_params = KVCacheStaticParams(num_heads=4, head_size=4),
            page_size=8,
            test_k=True,
        ](3, List[Int](12, 20, 8), List[Int](0, 1, 2), ctx)

        # Test V grouped matmul - Non-uniform token counts with shared LoRAs
        test_kv_grouped_matmul[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=3,
            expert_shape = Index(8, 16),
            kv_params = KVCacheStaticParams(num_heads=2, head_size=4),
            page_size=4,
            test_k=False,
        ](4, List[Int](7, 15, 3, 11), List[Int](0, 1, 0, 2), ctx)

        # Test K grouped matmul - Large non-uniform sequences
        test_kv_grouped_matmul[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(32, 64),
            kv_params = KVCacheStaticParams(num_heads=8, head_size=4),
            page_size=16,
            test_k=True,
        ](5, List[Int](17, 93, 45, 31, 62), List[Int](0, 3, 2, 4, 1), ctx)

        # Test K grouped matmul - LoRA zero weights preserve base cache
        test_lora_zero_weights_preserves_base_cache[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=2,
            expert_shape = Index(8, 16),
            kv_params = KVCacheStaticParams(num_heads=2, head_size=4),
            page_size=8,
            test_k=True,
        ](1, List[Int](16), List[Int](0), ctx)

        # Test V grouped matmul - LoRA zero weights preserve base cache
        test_lora_zero_weights_preserves_base_cache[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=2,
            expert_shape = Index(8, 16),
            kv_params = KVCacheStaticParams(num_heads=2, head_size=4),
            page_size=8,
            test_k=False,
        ](1, List[Int](16), List[Int](0), ctx)

        # Test K grouped matmul - Mixed zero/non-zero LoRA weights
        test_lora_mixed_zero_nonzero_weights[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=2,
            expert_shape = Index(8, 16),
            kv_params = KVCacheStaticParams(num_heads=2, head_size=4),
            page_size=8,
            test_k=True,
        ](
            2, List[Int](8, 8), List[Int](0, 1), ctx
        )  # seq 0 uses expert 0 (values), seq 1 uses expert 1 (zeros)

        # Test V grouped matmul - Mixed zero/non-zero LoRA weights
        test_lora_mixed_zero_nonzero_weights[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=2,
            expert_shape = Index(8, 16),
            kv_params = KVCacheStaticParams(num_heads=2, head_size=4),
            page_size=8,
            test_k=False,
        ](
            2, List[Int](8, 8), List[Int](0, 1), ctx
        )  # seq 0 uses expert 0 (values), seq 1 uses expert 1 (zeros)
