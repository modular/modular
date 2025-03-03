# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s -t

from collections import Set
from math import ceildiv, isqrt
from random import random_ui64

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from memory import UnsafePointer, memcpy
from nn.fused_qk_rope import fused_qk_rope_ragged
from testdata.fused_qk_rope_goldens import freqs_cis_table_input
from testing import assert_almost_equal

from utils import IndexList


def _init_device_ndbuffer_from_goldens[
    type: DType, //, shape: DimList
](goldens: List[Scalar[type]], ctx: DeviceContext) -> DeviceNDBuffer[
    type, len(shape), shape=shape
]:
    """Initializes a device buffer with a set of golden values."""
    host_tensor = HostNDBuffer[type, len(shape), shape=shape]()
    memcpy(dest=host_tensor.tensor.data, src=goldens.data, count=len(goldens))

    # Copy tensor to device.
    device_tensor = DeviceNDBuffer[
        host_tensor.type, host_tensor.rank, shape = host_tensor.shape
    ](ctx=ctx)
    ctx.enqueue_copy(device_tensor.buffer, host_tensor.tensor.data)
    ctx.synchronize()

    # Ensure the host buffer outlives the copy.
    _ = host_tensor^

    return device_tensor


def execute_fused_qk_rope_ragged(
    ctx: DeviceContext,
):
    alias num_q_heads = 32
    alias kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    alias type = DType.float32
    alias num_paged_blocks = 32
    alias page_size = 128
    var num_layers = 1
    var layer_idx = 0

    alias max_seq_len = 1024

    var true_ce_prompt_lens = List[Int](100, 200, 300, 400)
    var mixed_ce_prompt_lens = List[Int](50, 100, 150, 100)

    var true_ce_cache_lens = List[Int](0, 0, 0, 0)
    var mixed_ce_cache_lens = List[Int](50, 100, 150, 300)

    var batch_size = len(true_ce_prompt_lens)

    var true_ce_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var true_ce_cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var mixed_ce_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var mixed_ce_cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_cache_length = 0
    var mixed_ce_max_cache_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_row_offsets_host.tensor[i] = true_ce_total_length
        mixed_ce_row_offsets_host.tensor[i] = mixed_ce_total_length
        true_ce_cache_lengths_host.tensor[i] = true_ce_cache_lens[i]
        mixed_ce_cache_lengths_host.tensor[i] = mixed_ce_cache_lens[i]

        true_ce_max_cache_length = max(
            true_ce_max_cache_length, true_ce_cache_lens[i]
        )
        mixed_ce_max_cache_length = max(
            mixed_ce_max_cache_length, mixed_ce_cache_lens[i]
        )
        true_ce_max_full_context_length = max(
            true_ce_max_full_context_length,
            true_ce_cache_lens[i] + true_ce_prompt_lens[i],
        )
        mixed_ce_max_full_context_length = max(
            mixed_ce_max_full_context_length,
            mixed_ce_cache_lens[i] + mixed_ce_prompt_lens[i],
        )

        true_ce_max_prompt_length = max(
            true_ce_max_prompt_length, true_ce_prompt_lens[i]
        )
        mixed_ce_max_prompt_length = max(
            mixed_ce_max_prompt_length, mixed_ce_prompt_lens[i]
        )

        true_ce_total_length += true_ce_prompt_lens[i]
        mixed_ce_total_length += mixed_ce_prompt_lens[i]

    true_ce_row_offsets_host.tensor[batch_size] = true_ce_total_length
    mixed_ce_row_offsets_host.tensor[batch_size] = mixed_ce_total_length
    true_ce_row_offsets_device = true_ce_row_offsets_host.copy_to_device(ctx)
    mixed_ce_row_offsets_device = mixed_ce_row_offsets_host.copy_to_device(ctx)
    true_ce_cache_lengths_device = true_ce_cache_lengths_host.copy_to_device(
        ctx
    )
    mixed_ce_cache_lengths_device = mixed_ce_cache_lengths_host.copy_to_device(
        ctx
    )
    true_ce_q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))
    random(true_ce_q_ragged_host.tensor)
    true_ce_q_ragged_device = true_ce_q_ragged_host.copy_to_device(ctx)

    mixed_ce_q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    for bs_idx in range(batch_size):
        true_ce_prompt_len = true_ce_prompt_lens[bs_idx]
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        true_ce_row_offset = true_ce_row_offsets_host.tensor[bs_idx]
        mixed_ce_row_offset = mixed_ce_row_offsets_host.tensor[bs_idx]

        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        true_ce_offset = true_ce_q_ragged_host.tensor._offset(
            IndexList[3](Int(true_ce_row_offset + mixed_ce_cache_len), 0, 0)
        )
        mixed_ce_offset = mixed_ce_q_ragged_host.tensor._offset(
            IndexList[3](Int(mixed_ce_row_offset), 0, 0)
        )

        memcpy(
            mixed_ce_offset,
            true_ce_offset,
            mixed_ce_prompt_len * num_q_heads * kv_params.head_size,
        )

    mixed_ce_q_ragged_device = mixed_ce_q_ragged_host.copy_to_device(ctx)

    freqs_cis_table_dev = _init_device_ndbuffer_from_goldens[
        shape = DimList(max_seq_len, kv_params.head_size)
    ](freqs_cis_table_input[type](), ctx)

    # initialize reference output
    mixed_ce_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    mixed_ce_output_device = mixed_ce_output_host.copy_to_device(ctx)
    true_ce_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))
    true_ce_output_device = true_ce_output_host.copy_to_device(ctx)

    # initialize our KVCache
    true_ce_kv_block_paged_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_layers,
            2,
            num_paged_blocks,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    # don't randomly initialize, we'll copy the output to this buffer after executing.
    mixed_ce_kv_block_paged_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_layers,
            2,
            num_paged_blocks,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    random(true_ce_kv_block_paged_host.tensor)
    true_ce_kv_block_paged_device = true_ce_kv_block_paged_host.copy_to_device(
        ctx
    )
    # intentionally copy from true_ce so the contents of these blocks are consistent.
    mixed_ce_kv_block_paged_device = true_ce_kv_block_paged_host.copy_to_device(
        ctx
    )

    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](
            batch_size,
            ceildiv(true_ce_max_full_context_length, page_size),
        )
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval

    paged_lut_device = paged_lut_host.copy_to_device(ctx)

    var true_ce_k_cache_collection = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        true_ce_kv_block_paged_device.tensor,
        true_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        true_ce_max_prompt_length,
        true_ce_max_cache_length,
    )

    var mixed_ce_k_cache_collection = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        mixed_ce_kv_block_paged_device.tensor,
        mixed_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        mixed_ce_max_prompt_length,
        mixed_ce_max_cache_length,
    )

    # "true CE" execution
    print("true")
    var freqs_cis = rebind[
        NDBuffer[type, 2, shape = freqs_cis_table_dev.shape]
    ](freqs_cis_table_dev.tensor)
    fused_qk_rope_ragged[
        mixed_ce_k_cache_collection.CacheType, interleaved=False, target="gpu"
    ](
        true_ce_q_ragged_device.tensor,
        true_ce_row_offsets_device.tensor,
        true_ce_k_cache_collection,
        freqs_cis,
        layer_idx,
        output=true_ce_output_device.tensor,
        context=ctx,
    )

    # "mixed CE" execution
    print("mixed")
    fused_qk_rope_ragged[
        mixed_ce_k_cache_collection.CacheType, interleaved=False, target="gpu"
    ](
        mixed_ce_q_ragged_device.tensor,
        mixed_ce_row_offsets_device.tensor,
        mixed_ce_k_cache_collection,
        freqs_cis,
        layer_idx,
        output=mixed_ce_output_device.tensor,
        context=ctx,
    )
    ctx.enqueue_copy(
        mixed_ce_output_host.tensor.data, mixed_ce_output_device.buffer
    )
    ctx.enqueue_copy(
        true_ce_output_host.tensor.data, true_ce_output_device.buffer
    )
    ctx.enqueue_copy(
        true_ce_kv_block_paged_host.tensor.data,
        true_ce_kv_block_paged_device.buffer,
    )
    ctx.enqueue_copy(
        mixed_ce_kv_block_paged_host.tensor.data,
        mixed_ce_kv_block_paged_device.buffer,
    )
    ctx.synchronize()

    var true_ce_k_cache_collection_host = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        true_ce_kv_block_paged_host.tensor,
        true_ce_cache_lengths_host.tensor,
        paged_lut_host.tensor,
        true_ce_max_prompt_length,
        true_ce_max_cache_length,
    )
    var true_ce_k_cache = true_ce_k_cache_collection_host.get_key_cache(
        layer_idx
    )

    var mixed_ce_k_cache_collection_host = PagedKVCacheCollection[
        type, kv_params, page_size
    ](
        mixed_ce_kv_block_paged_host.tensor,
        mixed_ce_cache_lengths_host.tensor,
        paged_lut_host.tensor,
        mixed_ce_max_prompt_length,
        mixed_ce_max_cache_length,
    )
    var mixed_ce_k_cache = mixed_ce_k_cache_collection_host.get_key_cache(
        layer_idx
    )
    print("comparing Q")
    for bs_idx in range(batch_size):
        true_ce_batch_start_idx = Int(true_ce_row_offsets_host.tensor[bs_idx])
        mixed_ce_batch_start_idx = Int(mixed_ce_row_offsets_host.tensor[bs_idx])
        mixed_ce_cache_len = Int(mixed_ce_cache_lengths_host.tensor[bs_idx])

        for tok_idx in range(mixed_ce_prompt_lens[bs_idx]):
            for head_idx in range(num_q_heads):
                for head_dim in range(kv_params.head_size):
                    assert_almost_equal(
                        mixed_ce_output_host.tensor[
                            mixed_ce_batch_start_idx + tok_idx,
                            head_idx,
                            head_dim,
                        ],
                        true_ce_output_host.tensor[
                            true_ce_batch_start_idx
                            + mixed_ce_cache_len
                            + tok_idx,
                            head_idx,
                            head_dim,
                        ],
                    )

    print("comparing K")
    for bs_idx in range(batch_size):
        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        for tok_idx in range(mixed_ce_prompt_lens[bs_idx]):
            for head_idx in range(kv_params.num_heads):
                for head_dim in range(kv_params.head_size):
                    assert_almost_equal(
                        true_ce_k_cache.load[width=1](
                            bs_idx,
                            head_idx,
                            mixed_ce_cache_len + tok_idx,
                            head_dim,
                        ),
                        mixed_ce_k_cache.load[width=1](
                            bs_idx,
                            head_idx,
                            mixed_ce_cache_len + tok_idx,
                            head_dim,
                        ),
                    )

    _ = true_ce_row_offsets_host^
    _ = true_ce_row_offsets_device^
    _ = mixed_ce_row_offsets_host^
    _ = mixed_ce_row_offsets_device^
    _ = true_ce_q_ragged_host^
    _ = true_ce_q_ragged_device^
    _ = mixed_ce_q_ragged_host^
    _ = mixed_ce_q_ragged_device^
    _ = true_ce_kv_block_paged_host^
    _ = true_ce_kv_block_paged_device^
    _ = mixed_ce_kv_block_paged_host^
    _ = mixed_ce_kv_block_paged_device^
    _ = paged_lut_host^
    _ = paged_lut_device^
    _ = true_ce_output_host^
    _ = true_ce_output_device^
    _ = mixed_ce_output_host^
    _ = mixed_ce_output_device^


def main():
    with DeviceContext() as ctx:
        execute_fused_qk_rope_ragged(ctx)
