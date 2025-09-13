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
from memory import memcpy
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
    constrained[dtype is DType.float32, "goldens only for float32, currently"]()

    # Set up test hyperparameters - same as CPU test
    alias batch_size = 2
    alias seq_len = 3
    alias max_seq_len = 16
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

    # Create input query tensor using ManagedLayoutTensor
    q_buffer = q_input[dtype]()
    var q_managed = ManagedLayoutTensor[dtype, q_layout](ctx)
    var q_tensor = q_managed.tensor()
    # Copy data from golden buffer to managed tensor
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

    # Create start_pos tensor using ManagedLayoutTensor
    var start_pos_managed = ManagedLayoutTensor[DType.uint32, start_pos_layout](
        ctx
    )
    var start_pos_tensor = start_pos_managed.tensor()
    start_pos_tensor[0] = 0
    start_pos_tensor[1] = 5

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x))
    var freqs_cis_managed = ManagedLayoutTensor[dtype, freqs_cis_layout](ctx)
    freqs_cis_table_buffer = freqs_cis_table_input[dtype]()

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

    # Create output tensor using ManagedLayoutTensor
    var q_out_managed = ManagedLayoutTensor[dtype, q_layout](ctx)
    var q_out_tensor = q_out_managed.device_buffer()

    @always_inline
    @__copy_capture(q_out_tensor)
    fn output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
        q_out_tensor.store[width=width](
            rebind[IndexList[q_out_tensor.rank]](idx), val
        )

    # Execute rope_ragged kernel on GPU
    rope_ragged[
        dtype,
        q_layout,
        dtype,
        input_row_offsets_layout,
        start_pos_layout,
        freqs_cis_layout,
        interleaved=True,
        target = StaticString("gpu"),
        output_fn=output_fn,
    ](
        x=q_managed.device_tensor(),
        input_row_offsets=input_row_offsets_managed.device_tensor(),
        start_pos=start_pos_managed.device_tensor(),
        freqs_cis=freqs_cis_managed.device_tensor(),
        context=Optional[DeviceContext](ctx),
    )

    # Copy results back to host for validation
    var q_out_host_tensor = q_out_managed.tensor()

    # Create expected output for validation using ManagedLayoutTensor
    var expected_q_out_managed = ManagedLayoutTensor[dtype, q_layout](ctx)
    expected_q_out_buffer = q_out_golden[dtype]()
    var expected_q_out_tensor = expected_q_out_managed.tensor()
    for i in range(len(expected_q_out_buffer)):
        expected_q_out_tensor.ptr[i] = expected_q_out_buffer[i]

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

                @parameter
                if rope_dim == head_dim:
                    # Full RoPE case - compare entire output against golden
                    assert_almost_equal(
                        q_out_host_tensor.ptr + base_offset,
                        expected_q_out_tensor.ptr + base_offset,
                        head_dim,
                    )
                else:
                    # Partial RoPE case - use same logic as original test
                    # Verify unroped region: Should remain unchanged from input
                    assert_almost_equal(
                        q_out_host_tensor.ptr + base_offset,
                        q_tensor.ptr + base_offset,
                        head_dim - rope_dim,
                    )

                    # Verify roped region: Should match expected output
                    roped_offset = base_offset + (head_dim - rope_dim)
                    assert_almost_equal(
                        q_out_host_tensor.ptr + roped_offset,
                        expected_q_out_tensor.ptr + roped_offset,
                        rope_dim,
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
