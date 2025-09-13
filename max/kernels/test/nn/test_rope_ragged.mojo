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


from buffer import DimList, NDBuffer, Dim
from gpu.host import DeviceContext
from internal_utils import HostNDBuffer, assert_almost_equal
from layout import *
from layout._utils import ManagedLayoutTensor
from nn.rope import rope_ragged
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    q_input,
    q_out_golden,
)

from utils import IndexList


def test_rope_ragged[rope_dim: Int, dtype: DType]() -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    constrained[dtype is DType.float32, "goldens only for float32, currently"]()

    # Set up test hyperparameters.
    alias batch_size = 2
    var start_positions = List[UInt32](0, 5)
    var lookup_table = List[UInt32](0, 1)
    alias seq_len = 3
    alias max_seq_len = 16
    alias num_layers = 1

    fn _max[dtype: DType](items: List[Scalar[dtype]]) -> Scalar[dtype]:
        debug_assert(len(items) > 0, "empty list in _max")
        var max_item = items[0]

        for i in range(1, len(items)):
            if items[i] > max_item:
                max_item = items[i]
        return max_item

    debug_assert(
        max_seq_len > (seq_len + Int(_max[DType.uint32](start_positions))),
        "KV cache size smaller than sum of sequence length and start pos",
    )
    alias num_heads = 2
    alias dim = 16
    alias head_dim = dim // num_heads

    # Define layouts for all tensors
    alias q_layout = Layout(
        IntTuple(batch_size * seq_len, num_heads, head_dim),
        IntTuple(num_heads * head_dim, head_dim, 1),
    )
    alias input_row_offsets_layout = Layout(
        IntTuple(batch_size + 1), IntTuple(1)
    )
    alias start_pos_layout = Layout(IntTuple(batch_size), IntTuple(1))
    alias freqs_cis_layout = Layout(
        IntTuple(max_seq_len, rope_dim), IntTuple(rope_dim, 1)
    )

    # Create DeviceContext for CPU operations
    var ctx = DeviceContext(api="cpu")

    # Create and initialize query tensor using ManagedLayoutTensor
    var q_managed = ManagedLayoutTensor[dtype, q_layout](ctx)
    q_buffer = q_input[dtype]()
    debug_assert(
        len(q_buffer) == batch_size * seq_len * dim, "invalid q_buffer init"
    )

    # Copy data from golden buffer to managed tensor
    var q_tensor = q_managed.tensor()
    for i in range(len(q_buffer)):
        q_tensor.ptr[i] = q_buffer[i]

    # Create input_row_offsets using ManagedLayoutTensor
    var input_row_offsets_managed = ManagedLayoutTensor[
        DType.uint32, input_row_offsets_layout
    ](ctx)
    var input_row_offsets_tensor = input_row_offsets_managed.tensor()
    for i in range(batch_size):
        input_row_offsets_tensor[i] = i * seq_len
    input_row_offsets_tensor[batch_size] = batch_size * seq_len

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    var freqs_cis_managed = ManagedLayoutTensor[dtype, freqs_cis_layout](ctx)
    freqs_cis_table_buffer = freqs_cis_table_input[dtype]()
    debug_assert(
        len(freqs_cis_table_buffer) == 2 * max_seq_len * head_dim,
        "invalid freqs_cis_table init",
    )

    # Copy the roped dimensions from the buffer to the managed tensor
    var freqs_cis_tensor = freqs_cis_managed.tensor()
    for seq_idx in range(max_seq_len):
        for rope_idx in range(rope_dim):
            # Offset to last rope_dim elements in the original buffer
            var buffer_offset = (
                seq_idx * head_dim + (head_dim - rope_dim) + rope_idx
            )
            freqs_cis_tensor[seq_idx, rope_idx] = freqs_cis_table_buffer[
                buffer_offset
            ]

    # Create and initialize golden outputs using ManagedLayoutTensor
    var expected_q_out_managed = ManagedLayoutTensor[dtype, q_layout](ctx)
    expected_q_out_buffer = q_out_golden[dtype]()
    debug_assert(
        len(expected_q_out_buffer) == len(q_buffer),
        "invalid expected q out init",
    )
    var expected_q_out_tensor = expected_q_out_managed.tensor()
    for i in range(len(expected_q_out_buffer)):
        expected_q_out_tensor.ptr[i] = expected_q_out_buffer[i]

    # Create output tensor using ManagedLayoutTensor
    var q_out_managed = ManagedLayoutTensor[dtype, q_layout](ctx)
    var q_out_tensor = q_out_managed.buffer()
    # Initialize to zero
    for i in range(q_tensor.layout.size()):
        q_out_tensor.data[i] = 0

    # Create start_pos tensor using ManagedLayoutTensor
    var start_pos_managed = ManagedLayoutTensor[DType.uint32, start_pos_layout](
        ctx
    )
    var start_pos_tensor = start_pos_managed.tensor()
    for i in range(len(start_positions)):
        start_pos_tensor[i] = start_positions[i]

    @always_inline
    fn output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
        q_out_tensor.store[width=width](
            rebind[IndexList[q_out_tensor.rank]](idx), val
        )

    rope_ragged[
        dtype,
        q_layout,
        dtype,
        input_row_offsets_layout,
        start_pos_layout,
        freqs_cis_layout,
        interleaved=True,
        target = StaticString("cpu"),
        output_fn=output_fn,
    ](
        x=q_tensor,
        input_row_offsets=input_row_offsets_tensor,
        start_pos=start_pos_tensor,
        freqs_cis=freqs_cis_tensor,
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
                    q_out_tensor.data + base_offset,
                    q_tensor.ptr + base_offset,
                    head_dim - rope_dim,
                )

                # Verify roped region: Last rope_dim elements should match expected output
                roped_offset = base_offset + (head_dim - rope_dim)
                assert_almost_equal(
                    q_out_tensor.data + roped_offset,
                    expected_q_out_tensor.ptr + roped_offset,
                    rope_dim,
                )


def main() -> None:
    # Full head RoPE - this works correctly and is production ready
    print("Full head RoPE")
    test_rope_ragged[8, DType.float32]()

    # TODO: This was failing for some reason, we don't actually need it for now.
    # Circle back and fix this.
    # Partial RoPE (last 4 elements of each head)
    print("Partial RoPE")
    test_rope_ragged[4, DType.float32]()
