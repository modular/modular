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


from buffer import DimList, NDBuffer
from collections import OptionalReg
from gpu.host import DeviceContext
from internal_utils import HostNDBuffer, assert_almost_equal
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import IntTuple
from memory import memcpy
from nn.fused_qk_rope import fused_qk_rope_ragged
from testdata.fused_qk_rope_3d_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
    position_ids_input,
)

from utils import IndexList


def test_fused_qk_rope[rope_dim: Int, dtype: DType]() -> None:
    """Verifies fused_qk_rope_ragged with 3D position_ids and mrope sections
    against golden values computed with PyTorch.
    """
    constrained[dtype is DType.float32, "goldens only for float32, currently"]()

    # Set up test hyperparameters.
    alias batch_size = 2
    alias start_positions = List[UInt32](0, 5)
    alias lookup_table = List[UInt32](0, 1)
    alias seq_len = 3
    alias max_seq_len = 16
    alias num_layers = 1

    fn _max[dtype: DType, items: List[Scalar[dtype]]]() -> Scalar[dtype]:
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
    alias dim = 128
    alias head_dim = dim // num_heads

    # Create aliases for KV cache parameters.
    alias kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_dim
    )
    alias block_shape = IndexList[6](
        batch_size, 2, num_layers, max_seq_len, num_heads, head_dim
    )
    # The testdata is generated with mrope_section = (16, 8, 8),
    # but the expected input for the kernel is the original mrope_section
    # multiplied by 2, and prefix-summed.
    alias mrope_section = IntTuple(32, 48, 64)
    alias mrope_section_size = len(mrope_section)

    # Construct backing buffer and the KV cache itself.
    kv_cache_block_buffer = List[Scalar[dtype]](
        length=block_shape.flattened_length(), fill=0
    )
    kv_cache_block = NDBuffer(kv_cache_block_buffer.unsafe_ptr(), block_shape)

    # Initialize KV cache block buffer with golden values.
    k_cache_input_buffer = k_cache_input[dtype]()
    max_cache_len_in_batch = 0
    for batch_idx in range(batch_size):
        memcpy(
            dest=kv_cache_block._offset(
                IndexList[6](
                    batch_idx, 0, 0, Int(start_positions[batch_idx]), 0, 0
                )
            ),
            src=k_cache_input_buffer.unsafe_ptr() + (batch_idx * seq_len * dim),
            count=seq_len * dim,
        )
        max_cache_len_in_batch = max(
            max_cache_len_in_batch, Int(start_positions[batch_idx])
        )

    # Create the actual KV cache type.
    kv_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        blocks=kv_cache_block,
        cache_lengths=NDBuffer[DType.uint32, 1](
            start_positions.unsafe_ptr(),
            DimList(
                len(start_positions),
            ),
        ),
        lookup_table=NDBuffer[DType.uint32, 1](
            lookup_table.unsafe_ptr(),
            DimList(
                len(lookup_table),
            ),
        ),
        max_seq_length=seq_len,
        max_cache_length=max_cache_len_in_batch,
    )

    # Create and initialize query buffer.
    q_buffer = q_input[dtype]()
    debug_assert(
        len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"
    )

    # Create query tensor as a view of the query buffer.
    input_row_offsets = HostNDBuffer[DType.uint32, 1](DimList(batch_size + 1))
    for i in range(batch_size):
        input_row_offsets.tensor[i] = i * seq_len
    input_row_offsets.tensor[batch_size] = batch_size * seq_len

    # Create position_ids tensor for testing explicit position encoding
    # Total sequence length across all batches
    position_ids_input_buffer = position_ids_input[DType.uint32]()
    position_ids = NDBuffer[
        DType.uint32,
        rank=2,
        shape = DimList(3, batch_size * seq_len),
    ](position_ids_input_buffer.unsafe_ptr())

    q = NDBuffer[
        dtype,
        rank=3,
        shape = DimList(batch_size * seq_len, num_heads, head_dim),
    ](q_buffer.unsafe_ptr())

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    debug_assert(
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim,
        "invalid freqs_cis_table init"
        + String(len(freqs_cis_table_buffer))
        + " != "
        + String(2)
        + " * "
        + String(max_seq_len)
        + " * "
        + String(head_dim),
    )
    # Create a view into freqs_cis tensor that only includes the roped dimensions
    freqs_cis_table = NDBuffer[
        dtype,
        rank=2,
        shape = DimList(max_seq_len, rope_dim),
        strides = DimList(head_dim, 1),
    ](
        freqs_cis_table_buffer.unsafe_ptr() + (head_dim - rope_dim)
    )  # Offset to last rope_dim elements

    # Create and initialize golden outputs.
    expected_q_out_buffer = q_out_golden[dtype]()
    debug_assert(
        len(expected_q_out_buffer) == len(q_buffer),
        "invalid expected q out init",
    )
    expected_q_out = NDBuffer[dtype, rank=3, shape = q.shape](
        expected_q_out_buffer.unsafe_ptr()
    )
    expected_k_out_buffer = k_out_golden[dtype]()
    debug_assert(
        len(expected_k_out_buffer) == batch_size * seq_len * dim,
        "invalid expected k out init",
    )

    print("Created freqs_cis_table_buffer", flush=True)
    # Create output buffer.
    q_out_buffer = List[Scalar[dtype]](length=len(q_buffer), fill=0)
    q_out = NDBuffer[dtype, rank=3](q_out_buffer.unsafe_ptr(), q.dynamic_shape)
    fused_qk_rope_ragged[
        kv_collection.CacheType,
        interleaved=False,
        target = StaticString("cpu"),
        mrope_section=mrope_section,
    ](
        q,
        input_row_offsets.tensor,
        kv_collection,
        freqs_cis_table,
        OptionalReg[NDBuffer[DType.uint32, 2, MutableAnyOrigin]](position_ids),
        UInt32(0),
        q_out,
        Optional[DeviceContext](),
    )

    print("Created freqs_cis_table_buffer", flush=True)
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
                cache_block_ptr = kv_cache_block._offset(
                    IndexList[6](
                        batch_idx,
                        0,
                        0,
                        Int(start_positions[batch_idx]) + seq_idx,
                        head_idx,
                        0,
                    )
                )
                seq_offset = seq_idx * dim + head_idx * head_dim
                input_offset = batch_idx * seq_len * dim + seq_offset

                # Verify unroped region: Should match original input
                assert_almost_equal(
                    cache_block_ptr,
                    k_cache_input_buffer.unsafe_ptr() + input_offset,
                    head_dim - rope_dim,
                )

                # Verify roped region: Should match expected output
                roped_offset = head_dim - rope_dim
                assert_almost_equal(
                    cache_block_ptr + roped_offset,
                    expected_k_out_buffer.unsafe_ptr()
                    + input_offset
                    + roped_offset,
                    rope_dim,
                )

    _ = q_out_buffer^
    _ = expected_q_out_buffer^
    _ = freqs_cis_table_buffer^
    _ = q_buffer^
    _ = k_cache_input_buffer^
    _ = kv_cache_block_buffer^
    _ = position_ids_input_buffer^


def main() -> None:
    # Full head RoPE
    test_fused_qk_rope[64, DType.float32]()
