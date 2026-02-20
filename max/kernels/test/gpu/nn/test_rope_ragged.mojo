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

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from internal_utils import assert_almost_equal
from layout._coord import Coord, Idx, coord
from layout._layout import Layout, row_major
from layout._tile_tensor import TileTensor
from nn.rope import rope_ragged
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    q_input,
    q_out_golden,
)

from utils import IndexList


def test_rope_ragged_gpu[
    rope_dim: Int, dtype: DType
](ctx: DeviceContext) -> None:
    """Verifies rope_ragged GPU kernel against golden values computed with PyTorch.
    """
    comptime assert (
        dtype == DType.float32
    ), "goldens only for float32, currently"

    # Set up test hyperparameters - same as CPU test
    comptime batch_size = 2
    comptime seq_len = 3
    comptime max_seq_len = 16
    comptime num_heads = 2
    comptime dim = 16
    comptime head_dim = dim // num_heads

    # Define layouts for all tensors
    comptime q_layout = row_major[batch_size * seq_len, num_heads, head_dim]()
    comptime input_row_offsets_layout = row_major[batch_size + 1]()
    comptime start_pos_layout = row_major[batch_size]()
    comptime freqs_cis_layout = row_major[max_seq_len, rope_dim]()

    # ===== Step 1: Create all buffers =====
    # Query tensor buffers
    var q_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    var q_device_buffer = ctx.enqueue_create_buffer[dtype](
        q_layout.static_product
    )

    # Input row offsets buffers
    var input_row_offsets_host_buffer = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](input_row_offsets_layout.static_product)
    var input_row_offsets_device_buffer = ctx.enqueue_create_buffer[
        DType.uint32
    ](input_row_offsets_layout.static_product)

    # Start position buffers
    var start_pos_host_buffer = ctx.enqueue_create_host_buffer[DType.uint32](
        start_pos_layout.static_product
    )
    var start_pos_device_buffer = ctx.enqueue_create_buffer[DType.uint32](
        start_pos_layout.static_product
    )

    # Frequency table buffers
    var freqs_cis_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        freqs_cis_layout.static_product
    )
    var freqs_cis_device_buffer = ctx.enqueue_create_buffer[dtype](
        freqs_cis_layout.static_product
    )

    # Output buffers
    var q_out_device_buffer = ctx.enqueue_create_buffer[dtype](
        q_layout.static_product
    )
    var q_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )

    # Synchronize to ensure all buffers are created
    ctx.synchronize()

    # ===== Step 2: Fill host buffers with data =====
    # Fill query tensor
    q_buffer = q_input[dtype]()
    for i in range(len(q_buffer)):
        q_host_buffer[i] = q_buffer[i]

    # Fill input row offsets: [0, seq_len, 2*seq_len] = [0, 3, 6]
    for i in range(batch_size):
        input_row_offsets_host_buffer[i] = UInt32(i * seq_len)
    input_row_offsets_host_buffer[batch_size] = batch_size * seq_len

    # Fill start positions
    start_pos_host_buffer[0] = 0
    start_pos_host_buffer[1] = 5

    # Fill frequency table
    freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    for seq_idx in range(max_seq_len):
        for rope_idx in range(rope_dim):
            # Offset to last rope_dim elements in the original buffer
            var buffer_offset = (
                seq_idx * head_dim + (head_dim - rope_dim) + rope_idx
            )
            freqs_cis_host_buffer[
                seq_idx * rope_dim + rope_idx
            ] = freqs_cis_table_buffer[buffer_offset]

    # ===== Step 3: Copy all data to device =====
    ctx.enqueue_copy(q_device_buffer, q_host_buffer)
    ctx.enqueue_copy(
        input_row_offsets_device_buffer, input_row_offsets_host_buffer
    )
    ctx.enqueue_copy(start_pos_device_buffer, start_pos_host_buffer)
    ctx.enqueue_copy(freqs_cis_device_buffer, freqs_cis_host_buffer)

    # Synchronize to ensure all copies are complete before kernel execution
    ctx.synchronize()

    # ===== Step 4: Create TileTensor views =====
    var q_device_tensor = TileTensor(q_device_buffer.unsafe_ptr(), q_layout)
    var input_row_offsets_device_tensor = TileTensor(
        input_row_offsets_device_buffer.unsafe_ptr(), input_row_offsets_layout
    )
    var start_pos_device_tensor = TileTensor(
        start_pos_device_buffer.unsafe_ptr(), start_pos_layout
    )
    var freqs_cis_device_tensor = TileTensor(
        freqs_cis_device_buffer.unsafe_ptr(), freqs_cis_layout
    )
    var q_out_device_tensor = TileTensor(
        q_out_device_buffer.unsafe_ptr(), q_layout
    )

    @always_inline
    @__copy_capture(q_out_device_tensor)
    fn output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
        q_out_device_tensor.store[width=width](Coord(idx), val)

    # Execute rope_ragged kernel on GPU
    rope_ragged[
        dtype,
        dtype,
        interleaved=True,
        target = StaticString("gpu"),
        output_fn=output_fn,
    ](
        x=q_device_tensor.as_any_origin(),
        input_row_offsets=input_row_offsets_device_tensor.as_any_origin(),
        start_pos=start_pos_device_tensor.as_any_origin(),
        freqs_cis=freqs_cis_device_tensor.as_any_origin(),
        context=Optional[DeviceContext](ctx),
    )

    # Copy results back to host for validation
    ctx.enqueue_copy(q_out_host_buffer, q_out_device_buffer)
    ctx.synchronize()

    # Create expected output for validation using HostBuffer
    var expected_q_out_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        q_layout.static_product
    )
    ctx.synchronize()
    expected_q_out_buffer = q_out_golden[dtype]()
    for i in range(len(expected_q_out_buffer)):
        expected_q_out_host_buffer[i] = expected_q_out_buffer[i]

    # Validate results - same logic as CPU test
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            for head_idx in range(num_heads):
                # Calculate global token index and offsets
                global_token_idx = batch_idx * seq_len + seq_idx

                # Calculate base offset for current head
                base_offset = (
                    global_token_idx * num_heads * head_dim  # token offset
                    + head_idx * head_dim  # head offset
                )

                comptime if rope_dim == head_dim:
                    # Full RoPE case - compare entire output against golden
                    assert_almost_equal(
                        q_out_host_buffer.unsafe_ptr() + base_offset,
                        expected_q_out_host_buffer.unsafe_ptr() + base_offset,
                        head_dim,
                        atol=1e-4,
                    )
                else:
                    # Partial RoPE case - use same logic as original test
                    # Verify unroped region: Should remain unchanged from input
                    assert_almost_equal(
                        q_out_host_buffer.unsafe_ptr() + base_offset,
                        q_host_buffer.unsafe_ptr() + base_offset,
                        head_dim - rope_dim,
                        atol=1e-4,
                    )

                    # Verify roped region: Should match expected output
                    roped_offset = base_offset + (head_dim - rope_dim)
                    assert_almost_equal(
                        q_out_host_buffer.unsafe_ptr() + roped_offset,
                        expected_q_out_host_buffer.unsafe_ptr() + roped_offset,
                        rope_dim,
                        atol=1e-4,
                    )


def execute_rope_ragged_gpu(ctx: DeviceContext) -> None:
    """Execute GPU RoPE tests with different rope dimensions."""
    # Full head RoPE
    test_rope_ragged_gpu[8, DType.float32](ctx)

    # partial RoPE
    test_rope_ragged_gpu[4, DType.float32](ctx)


def main():
    with DeviceContext() as ctx:
        execute_rope_ragged_gpu(ctx)
