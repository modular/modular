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


from gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._coord import Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from memory import memcpy
from nn.fused_qk_rope import fused_qk_rope
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
)

from utils import IndexList


def test_fused_qk_rope[dtype: DType]() -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    comptime assert (
        dtype == DType.float32
    ), "goldens only for float32, currently"

    # Set up test hyperparameters.
    comptime batch_size = 2
    comptime start_positions: List[UInt32] = [0, 5]
    comptime seq_len = 3
    comptime max_seq_len = 16
    comptime num_layers = 1
    var lookup_table: List[UInt32] = [0, 1]

    fn _max[dtype: DType, items: List[Scalar[dtype]]]() -> Scalar[dtype]:
        comptime assert len(items) > 0, "empty list in _max"
        items_dyn = materialize[items]()
        max_item = items_dyn[0]
        for i in range(1, len(items_dyn)):
            if items_dyn[i] > max_item:
                max_item = items_dyn[i]
        return max_item

    comptime assert max_seq_len > (
        seq_len + Int(_max[DType.uint32, items=start_positions]())
    ), "KV cache size smaller than sum of sequence length and start pos"
    comptime num_heads = 2
    comptime dim = 16
    comptime head_dim = dim // num_heads

    # Create aliases for KV cache parameters.
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_dim
    )
    comptime block_shape = IndexList[6](
        batch_size, 2, num_layers, max_seq_len, num_heads, head_dim
    )

    # Construct backing buffer and the KV cache itself.
    kv_cache_block_buffer = List[Scalar[dtype]](
        length=block_shape.flattened_length(), fill=0
    )
    kv_cache_block = LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
        kv_cache_block_buffer.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    # Initialize KV cache block buffer with golden values.
    start_positions_dyn = materialize[start_positions]()
    k_cache_input_buffer = k_cache_input[dtype]()
    var max_cache_len_in_batch = 0
    for batch_idx in range(batch_size):
        memcpy(
            dest=kv_cache_block.ptr
            + kv_cache_block._offset(
                IndexList[6](
                    batch_idx, 0, 0, Int(start_positions_dyn[batch_idx]), 0, 0
                )
            ),
            src=k_cache_input_buffer.unsafe_ptr() + (batch_idx * seq_len * dim),
            count=seq_len * dim,
        )
        max_cache_len_in_batch = max(
            max_cache_len_in_batch, Int(start_positions_dyn[batch_idx])
        )

    # Create the actual KV cache type.
    kv_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        blocks=kv_cache_block,
        cache_lengths=LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            start_positions_dyn.unsafe_ptr(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](len(start_positions_dyn))
            ),
        ),
        lookup_table=LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            lookup_table.unsafe_ptr(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](len(lookup_table)),
            ),
        ),
        max_seq_length=seq_len,
        max_cache_length=UInt32(max_cache_len_in_batch),
    )

    # Create and initialize query buffer.
    q_buffer = q_input[dtype]()
    debug_assert(
        len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"
    )

    # Create query tensor as a view of the query buffer.
    q = TileTensor(
        q_buffer, row_major[batch_size, seq_len, num_heads, head_dim]()
    )

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    debug_assert(
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim,
        "invalid freqs_cis_table init",
    )
    freqs_cis_table = TileTensor(
        freqs_cis_table_buffer, row_major[max_seq_len, head_dim]()
    )

    # Create and initialize golden outputs.
    expected_q_out_buffer = q_out_golden[dtype]()
    debug_assert(
        len(expected_q_out_buffer) == len(q_buffer),
        "invalid expected q out init",
    )
    expected_q_out = TileTensor(expected_q_out_buffer, q.layout)
    expected_k_out_buffer = k_out_golden[dtype]()
    debug_assert(
        len(expected_k_out_buffer) == batch_size * seq_len * dim,
        "invalid expected k out init",
    )

    # Create output buffer.
    q_out_buffer = List[Scalar[dtype]](length=len(q_buffer), fill=0)
    q_out = TileTensor(q_out_buffer, q.layout).make_dynamic[DType.int64]()

    # Create valid_lengths buffer - all sequences have full seq_len valid
    var valid_lengths_buffer = List[UInt32](
        length=batch_size, fill=UInt32(seq_len)
    )
    var valid_lengths = TileTensor(
        valid_lengths_buffer,
        row_major(Idx(batch_size)),
    )

    fused_qk_rope[
        kv_collection.CacheType, interleaved=True, target = StaticString("cpu")
    ](
        q_proj=q,
        kv_collection=kv_collection,
        freqs_cis=freqs_cis_table,
        layer_idx=UInt32(0),
        valid_lengths=valid_lengths,
        output=q_out,
        context=Optional[DeviceContext](),
    )

    # Compare output and expected query tensors.
    assert_almost_equal(q_out.ptr, expected_q_out.ptr, expected_q_out.numel())

    # Compare output and expected key cache buffers.
    for batch_idx in range(batch_size):
        assert_almost_equal(
            kv_cache_block.ptr
            + kv_cache_block._offset(
                IndexList[6](
                    batch_idx, 0, 0, Int(start_positions_dyn[batch_idx]), 0, 0
                )
            ),
            expected_k_out_buffer.unsafe_ptr() + (batch_idx * seq_len * dim),
            # Number of elements in one batch item.
            len(expected_k_out_buffer) // batch_size,
        )

    _ = q_out_buffer^
    _ = expected_q_out_buffer^
    _ = freqs_cis_table_buffer^
    _ = q_buffer^
    _ = k_cache_input_buffer^
    _ = kv_cache_block_buffer^
    _ = valid_lengths_buffer^


def main() -> None:
    test_fused_qk_rope[DType.float32]()
