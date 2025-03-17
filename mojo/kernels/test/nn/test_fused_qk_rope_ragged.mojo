# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from collections import Optional
from collections.string import StaticString

from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import HostNDBuffer, assert_almost_equal
from kv_cache.types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
)
from memory import UnsafePointer, memcpy
from nn.fused_qk_rope import fused_qk_rope_ragged
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
)

from utils import IndexList


def test_fused_qk_rope[rope_dim: Int, type: DType]() -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    constrained[type == DType.float32, "goldens only for float32, currently"]()

    # Set up test hyperparameters.
    alias batch_size = 2
    alias start_positions = List[UInt32](0, 5)
    alias seq_len = 3
    alias max_seq_len = 16
    alias num_layers = 1

    fn _max[type: DType, items: List[Scalar[type]]]() -> Scalar[type]:
        constrained[len(items) > 0, "empty list in _max"]()
        max_item = items[0]
        for i in range(1, len(items)):
            if items[i] > max_item:
                max_item = items[i]
        return max_item

    constrained[
        max_seq_len
        > (seq_len + Int(_max[DType.uint32, items=start_positions]())),
        "KV cache size smaller than sum of sequence length and start pos",
    ]()
    alias num_heads = 2
    alias dim = 16
    alias head_dim = dim // num_heads

    # Create aliases for KV cache parameters.
    alias kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_dim
    )
    alias block_shape = IndexList[5](
        num_layers, batch_size, max_seq_len, num_heads, head_dim
    )

    # Construct backing buffer and the KV cache itself.
    k_cache_block_buffer = List[Scalar[type]]()
    k_cache_block_buffer.resize(
        new_size=batch_size * max_seq_len * dim, value=0
    )

    # Initialize KV cache block buffer with golden values.
    k_cache_input_buffer = k_cache_input[type]()
    max_cache_len_in_batch = 0
    for batch_idx in range(batch_size):
        memcpy(
            dest=(
                k_cache_block_buffer.data
                + (batch_idx * max_seq_len * dim)
                + Int(start_positions[batch_idx] * dim)
            ),
            src=k_cache_input_buffer.data + (batch_idx * seq_len * dim),
            count=seq_len * dim,
        )
        max_cache_len_in_batch = max(
            max_cache_len_in_batch, Int(start_positions[batch_idx])
        )

    # Create the actual KV cache type.
    k_cache_block = NDBuffer[type, 5](k_cache_block_buffer.data, block_shape)
    kv_collection = ContiguousKVCacheCollection[type, kv_params](
        key_cache=k_cache_block,
        value_cache=NDBuffer[type, 5](
            UnsafePointer[Scalar[type]](), block_shape
        ),  # passing as a dummy val, this isn't used.
        cache_lengths=NDBuffer[DType.uint32, 1](
            start_positions.data,
            DimList(
                len(start_positions),
            ),
        ),
        is_context_encoding=False,
        num_layers=num_layers,
        batch_size=batch_size,
        max_seq_len_in_batch=seq_len,
        max_cache_len_in_batch=max_cache_len_in_batch,
    )

    # Create and initialize query buffer.
    q_buffer = q_input[type]()
    debug_assert(
        len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"
    )

    # Create query tensor as a view of the query buffer.
    input_row_offsets = HostNDBuffer[DType.uint32, 1]((batch_size + 1,))
    for i in range(batch_size):
        input_row_offsets.tensor[i] = i * seq_len
    input_row_offsets.tensor[batch_size] = batch_size * seq_len

    q = NDBuffer[
        type, rank=3, shape = DimList(batch_size * seq_len, num_heads, head_dim)
    ](q_buffer.data)

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    freqs_cis_table_buffer = freqs_cis_table_input[type]()
    debug_assert(
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim,
        "invalid freqs_cis_table init",
    )
    # Create a view into freqs_cis tensor that only includes the roped dimensions
    freqs_cis_table = NDBuffer[
        type,
        rank=2,
        shape = DimList(max_seq_len, rope_dim),
        strides = DimList(head_dim, 1),
    ](
        freqs_cis_table_buffer.data + (head_dim - rope_dim)
    )  # Offset to last rope_dim elements

    # Create and initialize golden outputs.
    expected_q_out_buffer = q_out_golden[type]()
    debug_assert(
        len(expected_q_out_buffer) == len(q_buffer),
        "invalid expected q out init",
    )
    expected_q_out = NDBuffer[type, rank=3, shape = q.shape](
        expected_q_out_buffer.data
    )
    expected_k_out_buffer = k_out_golden[type]()
    debug_assert(
        len(expected_k_out_buffer) == batch_size * seq_len * dim,
        "invalid expected k out init",
    )

    # Create output buffer.
    q_out_buffer = List[Scalar[type]]()
    q_out_buffer.resize(new_size=len(q_buffer), value=0)
    q_out = NDBuffer[type, rank=3](q_out_buffer.data, q.dynamic_shape)
    fused_qk_rope_ragged[
        kv_collection.CacheType, interleaved=True, target = StaticString("cpu")
    ](
        q_proj=q,
        input_row_offsets=input_row_offsets.tensor,
        kv_collection=kv_collection,
        freqs_cis=freqs_cis_table,
        output=q_out,
        layer_idx=UInt32(0),
        context=Optional[DeviceContext](),
    )

    # Compare output and expected query tensors.
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for head_idx in range(num_heads):
                # Calculate base offset for current head
                base_offset = (
                    batch_idx * seq_len * dim  # batch offset
                    + seq_idx * dim  # sequence offset
                    + head_idx * head_dim  # head offset
                )
                # Verify unroped region: First (head_dim - rope_dim) elements should remain unchanged
                assert_almost_equal(
                    q_out.data + base_offset,
                    q.data + base_offset,
                    head_dim - rope_dim,
                )

                # Verify roped region: Last rope_dim elements should match expected output
                roped_offset = base_offset + (head_dim - rope_dim)
                assert_almost_equal(
                    q_out.data + roped_offset,
                    expected_q_out.data + roped_offset,
                    rope_dim,
                )

    # Compare output and expected key cache buffers.
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for head_idx in range(num_heads):
                # Calculate offsets for current position
                seq_offset = seq_idx * dim + head_idx * head_dim
                cache_offset = (
                    batch_idx * max_seq_len * dim  # batch offset in cache
                    + Int(
                        start_positions[batch_idx] * dim
                    )  # start position offset
                    + seq_offset  # sequence and head offset
                )
                input_offset = batch_idx * seq_len * dim + seq_offset

                # Verify unroped region: Should match original input
                assert_almost_equal(
                    k_cache_block_buffer.data + cache_offset,
                    k_cache_input_buffer.data + input_offset,
                    head_dim - rope_dim,
                )

                # Verify roped region: Should match expected output
                roped_offset = head_dim - rope_dim
                assert_almost_equal(
                    k_cache_block_buffer.data + cache_offset + roped_offset,
                    expected_k_out_buffer.data + input_offset + roped_offset,
                    rope_dim,
                )

    _ = q_out_buffer^
    _ = expected_q_out_buffer^
    _ = freqs_cis_table_buffer^
    _ = q_buffer^
    _ = k_cache_input_buffer^
    _ = k_cache_block_buffer^


def main() -> None:
    # Full head RoPE
    test_fused_qk_rope[8, DType.float32]()
    # Partial RoPE (last 4 elements of each head)
    test_fused_qk_rope[4, DType.float32]()
