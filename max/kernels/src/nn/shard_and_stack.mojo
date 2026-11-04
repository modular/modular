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
"""Implements shard-and-stack: shards a tensor across devices and stacks the shards into a higher-rank output."""

from max.gpu.host import DeviceBuffer, DeviceContextArray
from std.memory import unsafe_memcpy
from extensibility import InputVariadicTensors, OutputVariadicTensors
from std.sys import size_of
from std.utils import product

# Cap on the pinned staging buffer used to coalesce strided H2D uploads in
# `_shard_and_stack_multi_device`.
comptime _STAGING_BYTE_CAP = 32 * 1024 * 1024


def _validate_shard_and_stack[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
) raises:
    """Validate inputs and outputs for shard_and_stack operation.

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
    """
    comptime assert inputs.size > 0, "must have one or more inputs"
    comptime assert 0 <= axis < inputs.rank, "axis must be in [0, inputs.rank)"

    var input_shape = inputs[0].shape()
    var row_major_strides = input_shape.get_row_major_strides()

    # Validate that all inputs must have the same shape and row-major strides
    for i in range(inputs.size):
        if input_shape != inputs._tensors[i].shape():
            raise Error("all input shapes must match")

        if row_major_strides != inputs._tensors[i].strides():
            raise Error("all inputs must have a row-major layout")

    # Validate that input dimension along split axis is divisible by the number
    # of outputs.
    var input_axis_dim = inputs[0].dim_size(axis)
    if input_axis_dim % outputs.size != 0:
        raise Error(
            "input dimension along split axis must be divisible by outputs.size"
        )

    # Validate all outputs have the same expected shape
    for out_idx in range(outputs.size):
        var output_shape = outputs._tensors[out_idx].shape()

        if output_shape[0] != inputs.size:
            raise Error(
                "dimension zero of each output must match number of inputs"
            )

        # Validate that non-axis dimensions match between inputs and outputs
        # Note: output has extra dimension 0, so input dim i maps to output dim i+1
        for i in range(inputs.rank):
            if i != axis:
                if inputs[0].dim_size(i) != outputs._tensors[out_idx].dim_size(
                    i + 1
                ):
                    raise Error(
                        "non-axis dimensions must match between inputs and"
                        " outputs"
                    )

        # Validate that the axis dimension is correctly divided:
        # Output has extra dimension 0, so input axis i → output axis i+1
        comptime output_axis = axis + 1
        if inputs[0].dim_size(axis) // outputs.size != outputs._tensors[
            out_idx
        ].dim_size(output_axis):
            raise Error(
                "axis dimension mismatch: expected input_axis_dim //"
                " outputs.size"
            )


def _shard_and_stack_multi_device[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
    dev_ctxs_input: DeviceContextArray[_, _],
) raises:
    """Multi-device implementation using H2D transfers.

    Each device's slice of an input is uploaded with a single contiguous
    copy: directly when the slice is already contiguous in host memory
    (`outer_dims == 1`), otherwise through a bounded pinned staging buffer
    that gathers the slice's segments in output order.

    Device contexts are extracted from operands/results in order of first
    appearance (deduplicated). For multi-device:
      dev_ctxs_input[0] = CPU (from inputs)
      dev_ctxs_input[1..] = GPUs (from outputs)

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
        dev_ctxs_input: Device contexts for H2D transfers.
    """
    var dyn_inputs = inputs._tensors
    var dyn_outputs = outputs._tensors

    var input_axis = inputs[0].dim_size[axis]()
    var chunk_size = input_axis // outputs.size

    # Calculate dimensional products
    var outer_dims = product(inputs[0].shape(), 0, axis)
    var inner_dims = product(inputs[0].shape(), axis + 1, inputs.rank)

    # Elements in one chunk segment (contiguous in memory)
    var segment_elements = chunk_size * inner_dims

    # Total elements per input in output
    var output_elements_per_input = outer_dims * segment_elements

    # The copies are enqueued from this task, not from parallel CPU sub-tasks:
    # a task that waits on sub-tasks runs other ready tasks inline while it
    # waits, and a graph with one of these per MoE layer, each frame holding
    # every expert's tensor, nests deep enough to overflow the worker stack.
    # The graph already runs different layers' copies concurrently.
    for tp_index in range(outputs.size):
        # Device context for this output (index 0 is CPU, so +1)
        var gpu_ctx = dev_ctxs_input[tp_index + 1]
        var output_tensor = dyn_outputs[tp_index]

        if outer_dims == 1:
            # Each device's slice of an input is one contiguous host range,
            # so one H2D copy per (device, input) needs no staging.
            for input_idx in range(inputs.size):
                var input_tensor = dyn_inputs[input_idx]

                for outer_idx in range(outer_dims):
                    # Calculate source offset in input tensor
                    var src_offset = (
                        outer_idx * input_axis * inner_dims
                        + tp_index * segment_elements
                    )

                    # Calculate dest offset in output tensor
                    var dst_offset = (
                        input_idx * output_elements_per_input
                        + outer_idx * segment_elements
                    )

                    # H2D copy: source is host pointer, dest is device buffer
                    var src_ptr = input_tensor.unsafe_ptr().unsafe_offset(
                        src_offset
                    )
                    var dst_ptr = output_tensor.unsafe_ptr().unsafe_offset(
                        dst_offset
                    )

                    gpu_ctx.enqueue_copy(
                        DeviceBuffer(
                            gpu_ctx,
                            dst_ptr,
                            segment_elements,
                            owning=False,
                        ),
                        src_ptr,
                    )
        else:
            # The slice is strided in host memory: `outer_dims` segments of
            # `segment_elements` elements. Copying it segment by segment costs
            # one H2D call per outer row, which for a last-axis split is one
            # call per row of the input. Instead gather the segments into a
            # pinned staging buffer in output order -- which is also the
            # destination layout -- and upload each input with a single
            # contiguous copy.
            comptime elem_size = size_of[outputs.dtype]()
            var inputs_per_batch = max(
                1, _STAGING_BYTE_CAP // (output_elements_per_input * elem_size)
            )
            var staging = gpu_ctx.enqueue_create_host_buffer[outputs.dtype](
                min(inputs_per_batch, inputs.size) * output_elements_per_input
            )
            var staging_ptr = staging.unsafe_ptr()

            for batch_start in range(0, inputs.size, inputs_per_batch):
                var batch_end = min(batch_start + inputs_per_batch, inputs.size)

                for input_idx in range(batch_start, batch_end):
                    var input_tensor = dyn_inputs[input_idx]
                    var slot_ptr = staging_ptr.unsafe_offset(
                        (input_idx - batch_start) * output_elements_per_input
                    )

                    # Gather this device's segments of the input into its
                    # staging slot, in output order.
                    for outer_idx in range(outer_dims):
                        var src_offset = (
                            outer_idx * input_axis * inner_dims
                            + tp_index * segment_elements
                        )
                        unsafe_memcpy(
                            dest=slot_ptr.unsafe_offset(
                                outer_idx * segment_elements
                            ),
                            src=input_tensor.unsafe_ptr().unsafe_offset(
                                src_offset
                            ),
                            count=segment_elements,
                        )

                    # H2D copy of the input's whole output region, which is
                    # contiguous: source is the staging slot, dest is the
                    # device buffer.
                    var dst_ptr = output_tensor.unsafe_ptr().unsafe_offset(
                        input_idx * output_elements_per_input
                    )
                    gpu_ctx.enqueue_copy(
                        DeviceBuffer(
                            gpu_ctx,
                            dst_ptr,
                            output_elements_per_input,
                            owning=False,
                        ),
                        slot_ptr,
                    )

                # Keep the pinned buffer alive past the copies just enqueued:
                # without a use after them, Mojo could free it at the last
                # `.unsafe_ptr()` above, before the copies run.
                _ = staging
                if batch_end < inputs.size:
                    # The staging buffer is refilled for the next batch, so
                    # wait for this batch's copies to complete first.
                    gpu_ctx.synchronize()


def _shard_and_stack_single_device[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
) raises:
    """Single-device implementation using CPU unsafe_memcpy.

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
    """
    var dyn_inputs = inputs._tensors
    var dyn_outputs = outputs._tensors

    var input_axis = inputs[0].dim_size[axis]()
    var chunk_size = input_axis // outputs.size

    # Calculate dimensional products
    var outer_dims = product(inputs[0].shape(), 0, axis)
    var inner_dims = product(inputs[0].shape(), axis + 1, inputs.rank)

    # Elements in one chunk segment (contiguous in memory)
    var segment_elements = chunk_size * inner_dims

    # Total elements per input in output
    var output_elements_per_input = outer_dims * segment_elements

    # Serial for the same reason as the multi-device copies.
    for input_idx in range(inputs.size):
        var input_tensor = dyn_inputs[input_idx]

        for tp_index in range(outputs.size):
            var output_tensor = dyn_outputs[tp_index]

            for outer_idx in range(outer_dims):
                # Calculate source offset in input tensor
                # Layout: [outer_dims][full_axis_dim][inner_dims]
                var src_offset = (
                    outer_idx
                    * input_axis
                    * inner_dims  # Skip to outer position
                    + tp_index * segment_elements  # Skip to tp_index chunk
                )

                # Calculate dest offset in output tensor
                # Layout: [num_inputs][outer_dims][chunk_size][inner_dims]
                var dst_offset = (
                    input_idx
                    * output_elements_per_input  # Skip to input position
                    + outer_idx * segment_elements  # Skip to outer position
                )

                # Copy contiguous segment
                var src_ptr = input_tensor.unsafe_ptr().unsafe_offset(
                    src_offset
                )
                var dst_ptr = output_tensor.unsafe_ptr().unsafe_offset(
                    dst_offset
                )

                unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=segment_elements)


def shard_and_stack[
    axis: Int,
](
    outputs: OutputVariadicTensors,
    inputs: InputVariadicTensors[
        dtype=outputs.dtype,
        rank=outputs.rank - 1,
        ...,
    ],
    dev_ctxs_input: DeviceContextArray[_, _],
) raises:
    """Shard weight tensors across multiple devices for tensor parallelism.

    This operation takes multiple input tensors with identical shapes and shards
    them along a specified axis, distributing the shards to different devices
    (typically GPUs for tensor parallel inference).

    Parameters:
        axis: The dimension along which to shard the weights.

    Args:
        outputs: Output tensors, one per device/shard.
        inputs: Input tensors to be sharded, all with identical shapes.
        dev_ctxs_input: Device contexts for multi-device transfers.
    """
    _validate_shard_and_stack[axis](outputs, inputs)

    # Check if outputs are on different devices than inputs (multi-device mode).
    comptime is_multi_device = dev_ctxs_input.length > 1

    comptime if is_multi_device:
        _shard_and_stack_multi_device[axis](outputs, inputs, dev_ctxs_input)
    else:
        _shard_and_stack_single_device[axis](outputs, inputs)
