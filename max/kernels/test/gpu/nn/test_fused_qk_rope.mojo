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

from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
    TileTensor,
)
from std.memory import unsafe_memcpy

from nn.fused_qk_rope import fused_qk_rope
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
)

from std.utils import IndexList


def test_fused_qk_rope[dtype: DType](ctx: DeviceContext) raises -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    comptime assert dtype == .float32, "goldens only for float32, currently"

    # Set up test hyperparameters.
    comptime batch_size = 2
    comptime start_positions: List[UInt32] = [0, 5]
    comptime seq_len = 3
    comptime max_seq_len = 16
    comptime num_layers = 1
    # Small pages so batch 1's [5, 8) window straddles a page boundary.
    comptime page_size = 2
    comptime pages_per_seq = max_seq_len // page_size
    comptime num_paged_blocks = batch_size * pages_per_seq

    # Reverse the page pool so no sequence lands on an identity mapping.
    def _page_of(batch_idx: Int, tok_idx: Int) -> Int:
        return (
            num_paged_blocks
            - 1
            - (batch_idx * pages_per_seq + tok_idx // page_size)
        )

    def _max[dtype: DType, items: List[Scalar[dtype]]]() -> Scalar[dtype]:
        comptime assert len(items) > 0, "empty list in _max"
        var items_dyn = materialize[items]()
        var max_item = items_dyn[0]
        for i in range(1, len(items_dyn)):
            if items_dyn[i] > max_item:
                max_item = items_dyn[i]
        return max_item

    comptime assert max_seq_len > (
        seq_len + Int(_max[.uint32, items=start_positions]())
    ), "KV cache size smaller than sum of sequence length and start pos"
    comptime num_heads = 2
    comptime dim = 16
    comptime head_dim = dim // num_heads

    # Create aliases for KV cache parameters.
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_dim
    )

    # Define TileTensor layouts
    comptime q_tile_layout = row_major[
        batch_size, seq_len, num_heads, head_dim
    ]()
    comptime freqs_tile_layout = row_major[max_seq_len, head_dim]()
    comptime valid_lengths_tile_layout = row_major[batch_size]()

    # Create shapes
    var kv_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, num_heads, head_dim
    )
    var q_shape = IndexList[4](batch_size, seq_len, num_heads, head_dim)
    # The golden freqs table holds 2*max_seq_len rows of head_dim values
    # (positions 0..2*max_seq_len-1); the kernel only reads rows below
    # max_seq_len.
    var freqs_shape = IndexList[2](2 * max_seq_len, head_dim)

    # Create device buffers
    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](
        batch_size * pages_per_seq
    )
    var q_device = ctx.enqueue_create_buffer[dtype](q_shape.flattened_length())
    var freqs_device = ctx.enqueue_create_buffer[dtype](
        freqs_shape.flattened_length()
    )
    var q_out_device = ctx.enqueue_create_buffer[dtype](
        q_shape.flattened_length()
    )

    var start_positions_dyn = materialize[start_positions]()

    # Initialize KV cache block buffer with golden values.
    var k_cache_input_buffer = k_cache_input[dtype]()
    var k_cache_input_buffer_ptr: Pointer[
        k_cache_input_buffer.T, origin_of(k_cache_input_buffer)
    ] = k_cache_input_buffer.unsafe_ptr()
    with kv_block_device.map_to_host() as kv_block_host:
        for batch_idx in range(batch_size):
            var start_pos = Int(start_positions_dyn[batch_idx])
            # Rows are contiguous only within a page, so seed a token at a time.
            for seq_idx in range(seq_len):
                var tok_idx = start_pos + seq_idx
                var dest_offset = (
                    _page_of(batch_idx, tok_idx)
                    * 2
                    * num_layers
                    * page_size
                    * num_heads
                    * head_dim
                    + (tok_idx % page_size) * num_heads * head_dim
                )
                unsafe_memcpy(
                    dest=kv_block_host.unsafe_ptr() + dest_offset,
                    src=k_cache_input_buffer_ptr
                    + ((batch_idx * seq_len + seq_idx) * dim),
                    count=dim,
                )

    # Initialize cache_lengths with start_positions
    with cache_lengths_device.map_to_host() as cache_lengths_host:
        for i in range(batch_size):
            cache_lengths_host[i] = start_positions_dyn[i]

    # Initialize lookup_table
    with lookup_table_device.map_to_host() as lookup_table_host:
        for batch_idx in range(batch_size):
            for page_idx in range(pages_per_seq):
                lookup_table_host[
                    batch_idx * pages_per_seq + page_idx
                ] = UInt32(_page_of(batch_idx, page_idx * page_size))

    # Initialize query buffer with golden values
    var q_input_buffer = q_input[dtype]()
    with q_device.map_to_host() as q_host:
        unsafe_memcpy(
            dest=q_host.unsafe_ptr(),
            src=q_input_buffer.unsafe_ptr(),
            count=len(q_input_buffer),
        )

    # Initialize freqs_cis_table with golden values
    var freqs_input_buffer = freqs_cis_table_input[dtype]()
    with freqs_device.map_to_host() as freqs_host:
        unsafe_memcpy(
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

    var kv_block_tensor = TileTensor(
        kv_block_device,
        row_major(
            Coord(
                Int64(num_paged_blocks),
                Idx[2],
                Int64(num_layers),
                Idx[page_size],
                Idx[num_heads],
                Idx[head_dim],
            )
        ),
    )
    var cache_lengths_tensor = (
        TileTensor(cache_lengths_device, row_major(Coord(Int64(batch_size))))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var lookup_table_tensor = (
        TileTensor(
            lookup_table_device,
            row_major(Coord(Int64(batch_size), Int64(pages_per_seq))),
        )
        .as_imm()
        .as_unsafe_any_origin()
    )

    # Create TileTensors for q, freqs, and output
    var q_tensor = TileTensor(q_device, q_tile_layout)
    var freqs_tensor = TileTensor(freqs_device, freqs_tile_layout)
    var q_out_tensor = TileTensor(q_out_device, q_tile_layout)

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        blocks=rebind[Collection.blocks_tt_type](
            kv_block_tensor.as_unsafe_any_origin()
        ),
        cache_lengths=cache_lengths_tensor,
        lookup_table=lookup_table_tensor,
        max_seq_length=seq_len,
        max_cache_length=UInt32(max_cache_len_in_batch),
    )

    # Create and initialize golden outputs.
    var expected_q_out_buffer = q_out_golden[dtype]()
    assert (
        len(expected_q_out_buffer) == q_shape.flattened_length()
    ), "invalid expected q out init"
    var expected_k_out_buffer = k_out_golden[dtype]()
    assert (
        len(expected_k_out_buffer) == batch_size * seq_len * dim
    ), "invalid expected k out init"
    var expected_k_out_buffer_ptr: Pointer[
        expected_k_out_buffer.T, origin_of(expected_k_out_buffer)
    ] = expected_k_out_buffer.unsafe_ptr()

    # Create valid_lengths device buffer - all sequences have full seq_len valid
    var valid_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with valid_lengths_device.map_to_host() as valid_lengths_host:
        for i in range(batch_size):
            valid_lengths_host[i] = UInt32(seq_len)

    # Create valid_lengths TileTensor with Scalar layout and MutAnyOrigin
    var valid_lengths_static = TileTensor(
        valid_lengths_device, valid_lengths_tile_layout
    )
    var valid_lengths_tensor = TileTensor[
        .uint32, type_of(valid_lengths_static).LayoutType, MutAnyOrigin
    ](
        valid_lengths_static._storage.unsafe_origin_cast[MutAnyOrigin](),
        valid_lengths_static.layout,
    ).make_dynamic[
        DType.int64
    ]()

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
        assert_almost_equal(
            q_out_host.as_span(),
            Span(expected_q_out_buffer)[: q_shape.flattened_length()],
        )

    # Compare output and expected key cache buffers.
    with kv_block_device.map_to_host() as kv_block_out_host:
        for batch_idx in range(batch_size):
            var start_pos = Int(start_positions_dyn[batch_idx])
            for seq_idx in range(seq_len):
                var tok_idx = start_pos + seq_idx
                var src_offset = (
                    _page_of(batch_idx, tok_idx)
                    * 2
                    * num_layers
                    * page_size
                    * num_heads
                    * head_dim
                    + (tok_idx % page_size) * num_heads * head_dim
                )
                assert_almost_equal(
                    kv_block_out_host.unsafe_ptr() + src_offset,
                    expected_k_out_buffer_ptr
                    + ((batch_idx * seq_len + seq_idx) * dim),
                    # Number of elements in one token.
                    dim,
                )

    # Explicitly free device buffers to return memory to the buffer cache
    _ = kv_block_device^
    _ = cache_lengths_device^
    _ = lookup_table_device^
    _ = q_device^
    _ = freqs_device^
    _ = q_out_device^
    _ = valid_lengths_device^


def main() raises -> None:
    with DeviceContext() as ctx:
        test_fused_qk_rope[.float32](ctx)
