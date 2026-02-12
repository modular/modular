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

from utils import Index, IndexList


def test_fused_qk_rope[dtype: DType](ctx: DeviceContext) -> None:
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

    # Define layouts for LayoutTensor (used for KV cache)
    comptime kv_block_layout = Layout.row_major(
        batch_size, 2, num_layers, max_seq_len, num_heads, head_dim
    )
    comptime cache_lengths_layout = Layout.row_major(UNKNOWN_VALUE)
    comptime lookup_table_layout = Layout.row_major(UNKNOWN_VALUE)

    # Define TileTensor layouts
    comptime q_tile_layout = row_major[
        batch_size, seq_len, num_heads, head_dim
    ]()
    comptime freqs_tile_layout = row_major[max_seq_len, head_dim]()
    comptime valid_lengths_tile_layout = row_major[batch_size]()

    # Create shapes
    var kv_block_shape = IndexList[6](
        batch_size, 2, num_layers, max_seq_len, num_heads, head_dim
    )
    var cache_lengths_shape = Index(batch_size)
    var lookup_table_shape = Index(batch_size)
    var q_shape = IndexList[4](batch_size, seq_len, num_heads, head_dim)
    var freqs_shape = IndexList[2](max_seq_len, head_dim)

    # Create runtime layouts for LayoutTensor
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)
    var lookup_table_runtime_layout = RuntimeLayout[
        lookup_table_layout
    ].row_major(lookup_table_shape)

    # Create device buffers
    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    var q_device = ctx.enqueue_create_buffer[dtype](q_shape.flattened_length())
    var freqs_device = ctx.enqueue_create_buffer[dtype](
        freqs_shape.flattened_length()
    )
    var q_out_device = ctx.enqueue_create_buffer[dtype](
        q_shape.flattened_length()
    )

    start_positions_dyn = materialize[start_positions]()

    # Initialize KV cache block buffer with golden values.
    k_cache_input_buffer = k_cache_input[dtype]()
    with kv_block_device.map_to_host() as kv_block_host:
        var kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
            kv_block_host, kv_block_runtime_layout
        )
        for batch_idx in range(batch_size):
            var dest_offset = (
                batch_idx * 2 * num_layers * max_seq_len * num_heads * head_dim
                + Int(start_positions_dyn[batch_idx]) * num_heads * head_dim
            )
            memcpy(
                dest=kv_block_tensor.ptr + dest_offset,
                src=k_cache_input_buffer.unsafe_ptr()
                + (batch_idx * seq_len * dim),
                count=seq_len * dim,
            )

    # Initialize cache_lengths with start_positions
    with cache_lengths_device.map_to_host() as cache_lengths_host:
        for i in range(batch_size):
            cache_lengths_host[i] = start_positions_dyn[i]

    # Initialize lookup_table
    with lookup_table_device.map_to_host() as lookup_table_host:
        for i in range(batch_size):
            lookup_table_host[i] = lookup_table[i]

    # Initialize query buffer with golden values
    q_input_buffer = q_input[dtype]()
    with q_device.map_to_host() as q_host:
        memcpy(
            dest=q_host.unsafe_ptr(),
            src=q_input_buffer.unsafe_ptr(),
            count=len(q_input_buffer),
        )

    # Initialize freqs_cis_table with golden values
    freqs_input_buffer = freqs_cis_table_input[dtype]()
    with freqs_device.map_to_host() as freqs_host:
        memcpy(
            dest=freqs_host.unsafe_ptr(),
            src=freqs_input_buffer.unsafe_ptr(),
            count=len(freqs_input_buffer),
        )

    # Create the actual KV cache type.
    var max_cache_len_in_batch = 0
    for i in range(batch_size):
        max_cache_len_in_batch = max(
            max_cache_len_in_batch, Int(start_positions_dyn[i])
        )

    # Create layout tensors for KV cache (still uses LayoutTensor)
    var kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_device, kv_block_runtime_layout
    )
    var cache_lengths_tensor = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(cache_lengths_shape),
    )
    var lookup_table_tensor = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(lookup_table_shape),
    )

    # Create TileTensors for q, freqs, and output
    var q_tensor = TileTensor(q_device, q_tile_layout)
    var freqs_tensor = TileTensor(freqs_device, freqs_tile_layout)
    var q_out_tensor = TileTensor(q_out_device, q_tile_layout)

    kv_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        blocks=LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                kv_block_tensor.runtime_layout.shape.value
            ),
        ),
        cache_lengths=cache_lengths_tensor,
        lookup_table=lookup_table_tensor,
        max_seq_length=seq_len,
        max_cache_length=UInt32(max_cache_len_in_batch),
    )

    # Create and initialize golden outputs.
    expected_q_out_buffer = q_out_golden[dtype]()
    debug_assert(
        len(expected_q_out_buffer) == q_shape.flattened_length(),
        "invalid expected q out init",
    )
    expected_k_out_buffer = k_out_golden[dtype]()
    debug_assert(
        len(expected_k_out_buffer) == batch_size * seq_len * dim,
        "invalid expected k out init",
    )

    # Create valid_lengths device buffer - all sequences have full seq_len valid
    var valid_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    with valid_lengths_device.map_to_host() as valid_lengths_host:
        for i in range(batch_size):
            valid_lengths_host[i] = UInt32(seq_len)

    # Create valid_lengths TileTensor with RuntimeInt layout and MutAnyOrigin
    var valid_lengths_static = TileTensor(
        valid_lengths_device, valid_lengths_tile_layout
    )
    var valid_lengths_tensor = TileTensor[
        DType.uint32,
        _,
        MutAnyOrigin,
    ](
        valid_lengths_static.ptr.unsafe_origin_cast[MutAnyOrigin](),
        valid_lengths_static.layout,
    ).make_dynamic[DType.int64]()

    fused_qk_rope[kv_collection.CacheType, interleaved=True, target="gpu"](
        q_proj=q_tensor,
        kv_collection=kv_collection,
        freqs_cis=freqs_tensor,
        layer_idx=UInt32(0),
        valid_lengths=valid_lengths_tensor,
        output=q_out_tensor,
        context=ctx,
    )

    ctx.synchronize()

    # Compare output and expected query tensors.
    with q_out_device.map_to_host() as q_out_host:
        var expected_q_out = TileTensor(
            expected_q_out_buffer.unsafe_ptr(), q_tile_layout
        )
        assert_almost_equal(
            q_out_host.unsafe_ptr(),
            expected_q_out.ptr,
            q_shape.flattened_length(),
        )

    # Compare output and expected key cache buffers.
    with kv_block_device.map_to_host() as kv_block_out_host:
        var kv_block_out_tensor = LayoutTensor[dtype, kv_block_layout](
            kv_block_out_host, kv_block_runtime_layout
        )
        for batch_idx in range(batch_size):
            var src_offset = (
                batch_idx * 2 * num_layers * max_seq_len * num_heads * head_dim
                + Int(start_positions_dyn[batch_idx]) * num_heads * head_dim
            )
            assert_almost_equal(
                kv_block_out_tensor.ptr + src_offset,
                expected_k_out_buffer.unsafe_ptr()
                + (batch_idx * seq_len * dim),
                # Number of elements in one batch item.
                len(expected_k_out_buffer) // batch_size,
            )

    # Explicitly free device buffers to return memory to the buffer cache
    _ = kv_block_device^
    _ = cache_lengths_device^
    _ = lookup_table_device^
    _ = q_device^
    _ = freqs_device^
    _ = q_out_device^
    _ = valid_lengths_device^


def main() -> None:
    with DeviceContext() as ctx:
        test_fused_qk_rope[DType.float32](ctx)
