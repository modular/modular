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
from internal_utils import assert_almost_equal
from kv_cache.types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
)
from memory import UnsafePointer, memcpy
from nn.fused_qk_rope import fused_qk_rope
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
)

from utils import IndexList


def test_fused_qk_rope[type: DType]() -> None:
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
    var max_cache_len_in_batch = 0
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
    q = NDBuffer[
        type, rank=4, shape = DimList(batch_size, seq_len, num_heads, head_dim)
    ](q_buffer.data)

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    freqs_cis_table_buffer = freqs_cis_table_input[type]()
    debug_assert(
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim,
        "invalid freqs_cis_table init",
    )
    freqs_cis_table = NDBuffer[
        type, rank=2, shape = DimList(max_seq_len, head_dim)
    ](freqs_cis_table_buffer.data)

    # Create and initialize golden outputs.
    expected_q_out_buffer = q_out_golden[type]()
    debug_assert(
        len(expected_q_out_buffer) == len(q_buffer),
        "invalid expected q out init",
    )
    expected_q_out = NDBuffer[type, rank=4, shape = q.shape](
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
    q_out = NDBuffer[type, rank=4](q_out_buffer.data, q.dynamic_shape)
    fused_qk_rope[
        kv_collection.CacheType, interleaved=True, target = StaticString("cpu")
    ](
        q_proj=q,
        kv_collection=kv_collection,
        freqs_cis=freqs_cis_table,
        output=q_out,
        layer_idx=UInt32(0),
        context=Optional[DeviceContext](),
    )

    # Compare output and expected query tensors.
    assert_almost_equal(
        q_out.data, expected_q_out.data, expected_q_out.num_elements()
    )

    # Compare output and expected key cache buffers.
    for batch_idx in range(batch_size):
        assert_almost_equal(
            (
                k_cache_block_buffer.data
                + (batch_idx * max_seq_len * dim)
                # Account for the start_pos (cache_length) for this batch item.
                + Int(start_positions[batch_idx] * dim)
            ),
            expected_k_out_buffer.data + (batch_idx * seq_len * dim),
            # Number of elements in one batch item.
            len(expected_k_out_buffer) // batch_size,
        )

    _ = q_out_buffer^
    _ = expected_q_out_buffer^
    _ = freqs_cis_table_buffer^
    _ = q_buffer^
    _ = k_cache_input_buffer^
    _ = k_cache_block_buffer^


def main() -> None:
    test_fused_qk_rope[DType.float32]()
