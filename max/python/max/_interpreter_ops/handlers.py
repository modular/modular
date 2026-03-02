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

"""Op handlers for the MO graph interpreter.

This module contains the operation handlers that implement the actual
computation for MO operations. Each handler takes the interpreter instance,
the operation, and input buffers, and returns output buffers.

Handlers are registered using the @register_op_handler decorator.
"""

from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import max._interpreter_ops as ops
import numpy as np
from max import _core, graph
from max._core.dialects import mo, mosh
from max.driver import CPU, Buffer, Device
from max.dtype import DType

# Type alias for op handlers
# Signature: (op, input_buffers) -> output_buffers
OpHandler = Callable[
    [Any, Sequence[Buffer | None]],
    Sequence[Buffer | None],
]

# Op handler registries
# Maps operation types to handler functions (for isinstance checks)
_MO_OP_HANDLERS: dict[type[_core.Operation], OpHandler] = {}
# Maps operation names to handler functions (for name-based lookup fallback)
_MO_OP_NAME_HANDLERS: dict[str, OpHandler] = {}


def register_op_handler(
    op_type: type[_core.Operation],
) -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an MO op handler.

    Args:
        op_type: The MO operation class to handle (e.g., mo.AddOp).

    Returns:
        Decorator function that registers the handler.

    Example:
        @register_op_handler(mo.AddOp)
        def _handle_add(op, inputs):
            # Implementation
            return [output_buffer]
    """

    def decorator(fn: OpHandler) -> OpHandler:
        _MO_OP_HANDLERS[op_type] = fn
        # Also register by name for fallback lookup
        # Register both the direct name (e.g., "ExpOp") and with "Mo" prefix
        # (e.g., "MoExpOp") since nanobind may use either convention
        name = op_type.__name__
        _MO_OP_NAME_HANDLERS[name] = fn
        # Also register with "Mo" prefix for runtime compatibility
        if not name.startswith("Mo"):
            _MO_OP_NAME_HANDLERS[f"Mo{name}"] = fn
        return fn

    return decorator


def lookup_handler(op: _core.Operation) -> OpHandler | None:
    """Look up the handler for an operation.

    First tries type-based lookup, then falls back to name-based lookup
    to handle cases where nanobind creates different class objects.

    Args:
        op: The operation to look up.

    Returns:
        The handler function, or None if no handler exists.
    """
    # Try type-based lookup first
    if type(op) in _MO_OP_HANDLERS:
        return _MO_OP_HANDLERS[type(op)]

    # Fallback: try name-based lookup
    op_class_name = type(op).__name__
    if op_class_name in _MO_OP_NAME_HANDLERS:
        return _MO_OP_NAME_HANDLERS[op_class_name]

    return None


def _check_cpu_only(op: _core.Operation, target_device: Device) -> None:
    """Check that operation is running on CPU (host device).

    Args:
        op: The operation being executed.
        target_device: The target device for execution.

    Raises:
        NotImplementedError: If target device is not CPU.
    """
    if not target_device.is_host:
        raise NotImplementedError(
            f"GPU execution not supported for {type(op).__name__} "
            "in MO interpreter"
        )


def _get_target_device(op: _core.Operation) -> Device:
    """Get the target device from an op's first result type.

    Accesses the device_ref directly from the MLIR type to avoid
    Shape.from_mlir() crashes on parametric shapes (ParamDeclRefAttr).

    Args:
        op: The operation whose result device to extract.

    Returns:
        The target device for the operation's result.
    """
    result_mlir_type: mo.TensorType = list(op.results)[0].type  # type: ignore[assignment]
    return graph.DeviceRef.from_mlir(result_mlir_type.device_ref).to_device()


# Constant operations


@register_op_handler(mo.ConstantOp)
def _handle_constant(
    op: mo.ConstantOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.constant by materializing its value via C++ binding.

    Constants are mo.constant ops with embedded #M.dense_array values in the
    'value' attribute. Supported attribute types:
    - ArrayElementsAttr (#M.dense_array)
    - DenseResourceElementsAttr (external blob)
    - AlignedBytesAttr (#M.aligned_bytes)

    This implementation always copies data from the MLIR attribute into a new
    Buffer on CPU first, then transfers to the target device if needed.
    For splat constants (1 element in source, many in output), the single
    value is replicated on CPU before transfer.

    Args:
        op: The constant operation.
        inputs: Input buffers (empty for constants).

    Returns:
        List containing the materialized constant buffer.
    """
    # Extract the result type to get dtype and shape info
    result_type = graph.Type.from_mlir(op.results[0].type)
    assert isinstance(result_type, graph.TensorType)
    dtype = result_type.dtype
    shape = result_type.shape

    if not graph.Shape.is_static(shape):
        raise ValueError("Dynamic shapes not supported for constants")

    target_device = result_type.device.to_device()

    # Always create buffer on CPU first (C++ binding uses memcpy which
    # requires host memory). Splatting also happens on CPU.
    cpu_buffer = _core.graph._buffer_from_constant_attr(
        op.value, dtype, graph.Shape(shape).static_dims, CPU()
    )

    # Transfer to target device if not CPU
    if not target_device.is_host:
        return [cpu_buffer.to(target_device)]

    return [cpu_buffer]


# Mutable load operations


@register_op_handler(mo.MutableLoadOp)
def _handle_mutable_load(
    op: mo.MutableLoadOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.mutable.load by passing through the input buffer.

    mo.mutable.load reads from a buffer input. The handler receives the
    buffer as the first input (already resolved from slots by the dispatcher).
    The second input is the chain (None since chains are skipped).

    Args:
        op: The mutable load operation (unused).
        inputs: Input buffers - first is the buffer to load, second is the chain
            (None).

    Returns:
        List containing the loaded tensor buffer and None for the chain.
    """
    # MutableLoadOp produces (tensor, chain)
    # The interpreter executes sequentially, so chains are not needed.
    # Use None to avoid unnecessary buffer allocation.
    return [inputs[0], None]


# Transfer operations


@register_op_handler(mo.TransferOp)
def _handle_transfer(
    op: mo.TransferOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.transfer by transferring buffer between devices.

    TransferOp transfers tensor contents between devices (e.g. CPU<->GPU).
    When source and destination devices match and alwaysElideSameDeviceCopy is
    True, the result aliases the input. When the flag is False, a copy is made.

    Args:
        op: The transfer operation.
        inputs: Input buffers - first is the tensor to transfer, second is the
            chain (None).

    Returns:
        List containing the transferred tensor buffer and None for the chain.
    """
    assert isinstance(inputs[0], Buffer)
    input_buffer = inputs[0]
    target_device = _get_target_device(op)

    if input_buffer.device == target_device:
        if op.always_elide_same_device_copy:
            # Alias: return the input buffer directly (no copy).
            return [input_buffer, None]
        # Flag is False: copy on the same device via broadcast to same shape.
        output = Buffer(
            shape=input_buffer.shape,
            dtype=input_buffer.dtype,
            device=target_device,
        )
        ops.data_movement_ops.StaticBroadcastTo(
            output,
            input_buffer,
            list(input_buffer.shape),
            target_device._device_context_ptr(),
        )
        return [output, None]

    # Cross-device transfer
    # TransferOp produces (tensor, chain)
    return [input_buffer.to(target_device), None]


# Shape operations


@register_op_handler(mo.RebindOp)
def _handle_rebind(
    op: mo.RebindOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.rebind by passing through the input buffer.

    Rebind is a shape assertion that doesn't change the underlying data.

    Args:
        op: The rebind operation (unused).
        inputs: Input buffers - contains the tensor to rebind.

    Returns:
        List containing the input buffer unchanged.
    """
    return [inputs[0]]


@register_op_handler(mo.StaticBroadcastToOp)
def _handle_static_broadcast_to(
    op: mo.StaticBroadcastToOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.static.broadcast_to using Mojo kernel.

    Args:
        op: The static broadcast operation.
        inputs: Input buffers - contains the tensor to broadcast.

    Returns:
        List containing the broadcast tensor buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_buffers_on_device(inputs, target_device)

    assert isinstance(inputs[0], Buffer)

    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )
    target_shape = graph.Shape(shape).static_dims

    # Allocate output buffer
    output = Buffer(
        shape=target_shape,
        dtype=inputs[0].dtype,
        device=target_device,
    )

    # Call Mojo kernel
    ops.data_movement_ops.StaticBroadcastTo(
        output, inputs[0], target_shape, target_device._device_context_ptr()
    )

    return [output]


@register_op_handler(mo.BroadcastToOp)
def _handle_broadcast_to(
    op: mo.BroadcastToOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.broadcast_to using Mojo kernel.

    Supports both CPU and GPU tensors via the StaticBroadcastTo kernel.

    Args:
        op: The broadcast operation.
        inputs: Input buffers - first is the tensor to broadcast,
            second (optional) is the target shape tensor.

    Returns:
        List containing the broadcast tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)

    # Try to get static shape from result type, fall through to dynamic
    # shape from the second input if the shape is parametric.
    target_shape = None
    result_mlir_type: mo.TensorType = list(op.results)[0].type  # type: ignore[assignment]
    shape_attr = result_mlir_type.shape_attr
    if isinstance(shape_attr, mosh.ShapeAttr):
        shape = graph.Shape.from_mlir(shape_attr)
        if graph.Shape.is_static(shape):
            target_shape = graph.Shape(shape).static_dims

    if target_shape is None and len(inputs) > 1:
        # For dynamic/parametric shapes, get from the shape operand
        assert isinstance(inputs[1], Buffer)
        target_shape = inputs[1].to_numpy().tolist()

    if target_shape is None:
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )

    # Allocate output buffer on target device
    output = Buffer(
        shape=target_shape,
        dtype=inputs[0].dtype,
        device=target_device,
    )

    # Call Mojo kernel (supports both CPU and GPU)
    ops.data_movement_ops.StaticBroadcastTo(
        output, inputs[0], target_shape, target_device._device_context_ptr()
    )

    return [output]


# Helper for device validation


def _check_buffers_on_device(
    buffers: Sequence[Buffer | None], target_device: Device
) -> None:
    """Check that all non-None buffers are on the target device.

    Args:
        buffers: Sequence of buffers to check (None entries are skipped).
        target_device: The expected device for all buffers.

    Raises:
        ValueError: If any buffer is not on the target device.
    """
    for i, buf in enumerate(buffers):
        if buf is not None and buf.device != target_device:
            raise ValueError(
                f"Input buffer {i} is on {buf.device}, "
                f"but expected {target_device}."
            )


# Binary elementwise operations


def binary_elementwise_handler(op_type: type) -> OpHandler:
    op_binding = ops.BINARY_ELEMENTWISE[op_type]

    def handler(
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        assert isinstance(inputs[0], Buffer)
        assert isinstance(inputs[1], Buffer)

        target_device = _get_target_device(op)
        _check_buffers_on_device(inputs, target_device)

        output = Buffer(
            shape=inputs[0].shape,
            dtype=inputs[0].dtype,
            device=target_device,
        )

        op_binding(
            output, inputs[0], inputs[1], target_device._device_context_ptr()
        )

        return [output]

    return handler


for op_type in ops.BINARY_ELEMENTWISE:
    register_op_handler(op_type)(binary_elementwise_handler(op_type))


def binary_comparison_handler(op_type: type) -> OpHandler:
    op_binding = ops.BINARY_ELEMENTWISE_COMPARISON[op_type]

    def handler(
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        assert isinstance(inputs[0], Buffer)
        assert isinstance(inputs[1], Buffer)

        target_device = _get_target_device(op)
        _check_buffers_on_device(inputs, target_device)

        output = Buffer(
            shape=inputs[0].shape,
            dtype=DType.bool,
            device=target_device,
        )

        op_binding(
            output, inputs[0], inputs[1], target_device._device_context_ptr()
        )

        return [output]

    return handler


for op_type in ops.BINARY_ELEMENTWISE_COMPARISON:
    register_op_handler(op_type)(binary_comparison_handler(op_type))


# Unary elementwise operations


def unary_elementwise_handler(op_type: type) -> OpHandler:
    op_binding = ops.UNARY_ELEMENTWISE[op_type]

    def handler(
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        assert isinstance(inputs[0], Buffer)

        target_device = _get_target_device(op)
        _check_buffers_on_device(inputs, target_device)

        output = Buffer(
            shape=inputs[0].shape,
            dtype=inputs[0].dtype,
            device=target_device,
        )

        op_binding(output, inputs[0], target_device._device_context_ptr())

        return [output]

    return handler


for op_type in ops.UNARY_ELEMENTWISE:
    register_op_handler(op_type)(unary_elementwise_handler(op_type))


# Unary mixed-dtype operations (cast, is_nan, is_inf)


def unary_mixed_handler(op_type: type) -> OpHandler:
    op_binding = ops.UNARY_MIXED[op_type]

    def handler(
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        assert isinstance(inputs[0], Buffer)

        result_type = graph.Type.from_mlir(list(op.results)[0].type)
        assert isinstance(result_type, graph.TensorType)
        target_device = result_type.device.to_device()
        _check_buffers_on_device(inputs, target_device)

        # Output dtype comes from the MLIR result type (not the input dtype).
        # For IsNan/IsInf: result_type.dtype is DType.bool
        # For Cast: result_type.dtype is the target cast dtype
        output = Buffer(
            shape=inputs[0].shape,
            dtype=result_type.dtype,
            device=target_device,
        )

        op_binding(output, inputs[0], target_device._device_context_ptr())

        return [output]

    return handler


for op_type in ops.UNARY_MIXED:
    register_op_handler(op_type)(unary_mixed_handler(op_type))

# Matrix operations


@register_op_handler(mo.MatmulOp)
def _handle_matmul(
    op: mo.MatmulOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.matmul by dispatching to Mojo matmul kernel."""
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_buffers_on_device(inputs, target_device)

    lhs = inputs[0]
    rhs = inputs[1]
    assert isinstance(lhs, Buffer)
    assert isinstance(rhs, Buffer)

    # Calculate output shape: (M, K) @ (K, N) -> (M, N)
    m = lhs.shape[0]
    n = rhs.shape[1]

    output = Buffer(shape=(m, n), dtype=lhs.dtype, device=target_device)

    ops.matmul_ops.Matmul(output, lhs, rhs, target_device._device_context_ptr())
    return [output]


@register_op_handler(mo.BatchMatmulOp)
def _handle_batch_matmul(
    op: mo.BatchMatmulOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.batch_matmul by dispatching to Mojo batched matmul kernel."""
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_buffers_on_device(inputs, target_device)

    lhs = inputs[0]
    rhs = inputs[1]
    assert isinstance(lhs, Buffer)
    assert isinstance(rhs, Buffer)

    # Compute output shape - try static first, fall back to Mojo shape fn
    shape = result_type.shape
    if graph.Shape.is_static(shape):
        output_shape = graph.Shape(shape).static_dims
    else:
        shape_result = ops.matmul_ops.BatchMatmulShape(lhs, rhs)
        output_shape = [int(shape_result[i]) for i in range(len(shape_result))]

    output = Buffer(shape=output_shape, dtype=lhs.dtype, device=target_device)

    ops.matmul_ops.BatchMatmul(
        output, lhs, rhs, target_device._device_context_ptr()
    )
    return [output]


# Shape manipulation operations


def _reshape_common(
    op: _core.Operation,
    inputs: Sequence[Buffer | None],
    op_name: str,
) -> Sequence[Buffer]:
    """Common implementation for reshape operations.

    Uses Buffer.view() to create a reshaped view sharing the underlying
    memory, supporting both CPU and GPU tensors without data movement.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_buffers_on_device(inputs, target_device)

    assert isinstance(inputs[0], Buffer)

    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(f"Dynamic shapes not supported for {op_name}")
    target_shape = graph.Shape(shape).static_dims

    return [inputs[0].view(inputs[0].dtype, tuple(target_shape))]


@register_op_handler(mo.ReshapeOp)
def _handle_reshape(
    op: mo.ReshapeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reshape."""
    return _reshape_common(op, inputs, "reshape")


@register_op_handler(mo.StaticReshapeOp)
def _handle_static_reshape(
    op: mo.StaticReshapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.static.reshape - reshape without inferred dimensions."""
    return _reshape_common(op, inputs, "static reshape")


@register_op_handler(mo.SqueezeShapeOp)
def _handle_squeeze_shape(
    op: mo.SqueezeShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.squeeze_shape - computes shape with specified dimensions removed.

    This is a CPU-side shape metadata operation. Given an input shape vector
    and a list of indices, returns a new shape vector with the indicated
    dimensions removed. The indicated dimensions must have size 1.

    Args:
        op: The squeeze shape operation.
        inputs: Input buffers - first is the shape vector, second is the
            indices tensor specifying which dimensions to remove.

    Returns:
        List containing the new shape vector as a 1D si64 buffer.
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_shape = inputs[0].to_numpy().tolist()
    remove_indices = inputs[1].to_numpy().tolist()

    rank = len(input_shape)
    # Normalize negative indices
    normalized = set()
    for idx in remove_indices:
        idx = int(idx)
        if idx < 0:
            idx += rank
        normalized.add(idx)

    # Build output shape by removing indicated dimensions
    result_shape = [
        dim for i, dim in enumerate(input_shape) if i not in normalized
    ]
    result_np = np.array(result_shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.UnsqueezeShapeOp)
def _handle_unsqueeze_shape(
    op: mo.UnsqueezeShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.unsqueeze_shape - computes shape with size-1 dimensions inserted.

    This is a CPU-side shape metadata operation. Given an input shape vector
    of rank N and a list of M indices, returns a new shape vector of rank N+M
    where the indicated positions are filled with 1 and the original dimensions
    fill the remaining positions.

    Args:
        op: The unsqueeze shape operation.
        inputs: Input buffers - first is the shape vector, second is the
            padding indices tensor specifying where to insert size-1 dims.

    Returns:
        List containing the new shape vector as a 1D si64 buffer.
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_shape = inputs[0].to_numpy().tolist()
    padding_indices = inputs[1].to_numpy().tolist()

    new_rank = len(input_shape) + len(padding_indices)
    # Normalize negative indices relative to the new rank
    normalized = set()
    for idx in padding_indices:
        idx = int(idx)
        if idx < 0:
            idx += new_rank
        normalized.add(idx)

    # Build output shape: insert 1s at indicated positions, fill rest from input
    result_shape = []
    input_idx = 0
    for i in range(new_rank):
        if i in normalized:
            result_shape.append(1)
        else:
            result_shape.append(int(input_shape[input_idx]))
            input_idx += 1

    result_np = np.array(result_shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.AddSingletonDimOp)
def _handle_add_singleton_dim(
    op: mo.AddSingletonDimOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.add_singleton_dim - adds a dimension of size 1 at the given axis.

    This is a shape-change op that does not copy data. It uses numpy.reshape
    with the target shape from the MLIR result type.

    Args:
        op: The add singleton dim operation.
        inputs: Input buffers - contains the tensor to reshape.

    Returns:
        List containing the reshaped tensor buffer.
    """
    return _reshape_common(op, inputs, "add_singleton_dim")


@register_op_handler(mo.SplitDimOp)
def _handle_split_dim(
    op: mo.SplitDimOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.split_dim - splits one dimension into two dimensions.

    E.g., a tensor of shape [N, K] with axis=0 becomes [S1, S2, K] where
    S1 * S2 = N. The target shape comes from the MLIR result type.

    Args:
        op: The split dim operation.
        inputs: Input buffers - contains the tensor to reshape.

    Returns:
        List containing the reshaped tensor buffer.
    """
    return _reshape_common(op, inputs, "split_dim")


@register_op_handler(mo.MergeDimOp)
def _handle_merge_dim(
    op: mo.MergeDimOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.merge_dim - merges two adjacent dimensions into one.

    E.g., a tensor of shape [A, B, C, D] with axis=1 becomes [A, B*C, D].
    The target shape comes from the MLIR result type.

    Args:
        op: The merge dim operation.
        inputs: Input buffers - contains the tensor to reshape.

    Returns:
        List containing the reshaped tensor buffer.
    """
    return _reshape_common(op, inputs, "merge_dim")


@register_op_handler(mo.TransposeOp)
def _handle_transpose(
    op: mo.TransposeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.transpose using Mojo kernel.

    Supports both CPU and GPU tensors via the Transpose kernel.

    Args:
        op: The transpose operation.
        inputs: Input buffers - first is the tensor to transpose,
            second is the permutation tensor (int64 on CPU).

    Returns:
        List containing the transposed tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    # Read permutation from the second input (int64 constant on CPU)
    perm = inputs[1].to_numpy().tolist()
    perm = [int(p) for p in perm]

    # Compute output shape by applying permutation to input shape
    in_shape = list(inputs[0].shape)
    out_shape = [in_shape[p] for p in perm]

    # Allocate output buffer on target device
    output = Buffer(
        shape=out_shape,
        dtype=inputs[0].dtype,
        device=target_device,
    )

    # Call Mojo kernel (supports both CPU and GPU)
    ops.data_movement_ops.Transpose(
        output,
        inputs[0],
        perm,
        in_shape,
        out_shape,
        target_device._device_context_ptr(),
    )

    return [output]


@register_op_handler(mo.SliceOp)
def _handle_slice(
    op: mo.SliceOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.slice by dispatching to Mojo slice kernel.

    Args:
        op: The slice operation.
        inputs: Input buffers - (input, starts, stops, steps) where
            starts/stops/steps are 1D tensors with one element per dimension.

    Returns:
        List containing the sliced tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)
    assert isinstance(inputs[3], Buffer)

    input_buffer = inputs[0]
    starts_buffer = inputs[1]
    stops_buffer = inputs[2]
    steps_buffer = inputs[3]

    # Read starts/stops/steps to compute output shape
    # .to_numpy() handles GPU→CPU transfer transparently
    start_np = starts_buffer.to_numpy().astype(np.int64)
    stop_np = stops_buffer.to_numpy().astype(np.int64)
    step_np = steps_buffer.to_numpy().astype(np.int64)

    rank = len(start_np)
    output_shape = tuple(
        int(max(0, int(np.ceil((stop_np[i] - start_np[i]) / step_np[i]))))
        for i in range(rank)
    )

    # Allocate output buffer on target device
    output = Buffer(
        shape=output_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    # Pad starts/stops/steps to MAX_RANK=5 for Mojo kernel
    max_rank = 5
    pad_count = max_rank - rank
    padded_starts = np.zeros(max_rank, dtype=np.int64)
    padded_stops = np.ones(max_rank, dtype=np.int64)
    padded_steps = np.ones(max_rank, dtype=np.int64)
    padded_starts[pad_count:] = start_np
    padded_stops[pad_count:] = stop_np
    padded_steps[pad_count:] = step_np

    # Call Mojo kernel
    ops.data_movement_ops.Slice(
        output,
        input_buffer,
        Buffer.from_numpy(padded_starts),
        Buffer.from_numpy(padded_stops),
        Buffer.from_numpy(padded_steps),
        target_device._device_context_ptr(),
    )

    return [output]


# Shape/parameter operations


@register_op_handler(mo.ShapeOfOp)
def _handle_shape_of(
    op: mo.ShapeOfOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.shape_of - returns the shape of a tensor as a 1D si64 tensor.

    This is a CPU-side metadata operation. The result is always a CPU buffer
    regardless of the input tensor's device, since shape metadata is always
    host-accessible.
    """
    assert isinstance(inputs[0], Buffer)
    shape = inputs[0].shape
    result_np = np.array(shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.BroadcastShapeOp)
def _handle_broadcast_shape(
    op: mo.BroadcastShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.broadcast_shape - compute broadcast shape of two shapes.

    This is a CPU-side metadata operation. The result is always a CPU buffer
    since it computes shape information from small integer tensors.
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    shape_x = tuple(inputs[0].to_numpy().tolist())
    shape_y = tuple(inputs[1].to_numpy().tolist())
    result_shape = np.broadcast_shapes(shape_x, shape_y)
    result_np = np.array(result_shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.ShapeToTensorOp)
def _handle_shape_to_tensor(
    op: mo.ShapeToTensorOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.shape.to_tensor - converts shape value to tensor.

    The input is a !mosh.ape shape value (already a buffer from ParamToValueOp).
    This op just passes through the buffer since ParamToValueOp already
    created a tensor representation.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    # The input should already be a buffer containing the shape values
    # Just pass it through
    assert isinstance(inputs[0], Buffer)
    return [inputs[0]]


@register_op_handler(mosh.ParamToValueOp)
def _handle_param_to_value(
    op: mosh.ParamToValueOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mosh.param.to_value - materializes parameter values.

    This op takes a compile-time parameter expression and produces an SSA value.
    For static shapes like <[0, 0]> or <[1, 3]>, we extract the values and
    create a buffer.
    """
    # Get the value attribute which contains the parameter expression
    value_attr = op.value

    # Get the result type to understand what we're producing
    result = list(op.results)[0]
    result_type = result.type

    # Handle !mosh.ape (shape type) - produces a tensor of indices
    if isinstance(result_type, mosh.ShapeType):
        # value_attr should be a ShapeAttr with values
        if isinstance(value_attr, mosh.ShapeAttr):
            shape_values = []
            for dim_attr in value_attr.values:
                if hasattr(dim_attr, "value"):
                    val = dim_attr.value
                    if isinstance(val, int):
                        shape_values.append(val)
                    else:
                        raise NotImplementedError(
                            f"Dynamic dimension in param.to_value: {dim_attr}"
                        )
                else:
                    raise NotImplementedError(
                        f"Unsupported dimension attr in param.to_value: {dim_attr}"
                    )
            # Create a 1D tensor of si64 values
            result_np = np.array(shape_values, dtype=np.int64)
            output = Buffer.from_numpy(result_np)
            return [output]
        else:
            raise NotImplementedError(
                f"Unsupported value attr type for shape: {type(value_attr)}"
            )

    # Handle index type (single integer value)
    # Check if it's an index/integer type by looking at the attribute
    if hasattr(value_attr, "value"):
        val = value_attr.value
        if isinstance(val, int):
            result_np = np.array([val], dtype=np.int64)
            output = Buffer.from_numpy(result_np)
            return [output]

    raise NotImplementedError(
        f"Unsupported param.to_value result type: {result_type}, attr: {value_attr}"
    )


# Reduce operations


def reduce_handler(op_type: type) -> OpHandler:
    op_binding = ops.REDUCE[op_type]

    def handler(
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        result_type = graph.Type.from_mlir(list(op.results)[0].type)
        assert isinstance(result_type, graph.TensorType)
        target_device = result_type.device.to_device()

        assert isinstance(inputs[0], Buffer)
        assert isinstance(inputs[1], Buffer)

        input_buffer = inputs[0]
        axis_buffer = inputs[1]

        # Extract axis value from the axis tensor (scalar si64)
        axis_np = axis_buffer.to_numpy()
        axis = int(axis_np.item())

        # Calculate output shape (same as input with reduced axis dim = 1)
        output_shape = list(input_buffer.shape)
        output_shape[axis] = 1

        output = Buffer(
            shape=output_shape,
            dtype=input_buffer.dtype,
            device=target_device,
        )

        op_binding(
            output, input_buffer, axis, target_device._device_context_ptr()
        )

        return [output]

    return handler


for op_type in ops.REDUCE:
    register_op_handler(op_type)(reduce_handler(op_type))


# Softmax operations


def softmax_handler(op_type: type) -> OpHandler:
    op_binding = ops.SOFTMAX[op_type]

    def handler(
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        result_type = graph.Type.from_mlir(list(op.results)[0].type)
        assert isinstance(result_type, graph.TensorType)
        target_device = result_type.device.to_device()

        assert isinstance(inputs[0], Buffer)
        assert isinstance(inputs[1], Buffer)

        input_buffer = inputs[0]
        axis_buffer = inputs[1]

        # Extract axis value from the axis tensor (scalar si64)
        axis = int(axis_buffer.to_numpy().item())

        # Output shape is the same as input (not reduced)
        output = Buffer(
            shape=input_buffer.shape,
            dtype=input_buffer.dtype,
            device=target_device,
        )

        op_binding(
            output, input_buffer, axis, target_device._device_context_ptr()
        )

        return [output]

    return handler


for op_type in ops.SOFTMAX:
    register_op_handler(op_type)(softmax_handler(op_type))


# Cumsum operation


@register_op_handler(mo.CumsumOp)
def _handle_cumsum(
    op: mo.CumsumOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.cumsum by dispatching to Mojo cumsum kernel.

    Args:
        op: The cumsum operation.
        inputs: Input buffers - first is the input tensor, second is the
            axis tensor (scalar int64).

    Returns:
        List containing the cumsum tensor buffer.
    """
    assert isinstance(inputs[0], Buffer)  # input tensor
    assert isinstance(inputs[1], Buffer)  # axis (scalar int64)

    input_buffer = inputs[0]
    axis_buffer = inputs[1]

    # Extract axis value from the axis tensor (scalar si64)
    axis = int(axis_buffer.to_numpy().item())

    # Extract exclusive and reverse from op attributes
    exclusive = op.exclusive
    reverse = op.reverse

    # Output shape is the same as input shape (cumsum preserves shape)
    output = Buffer(
        shape=input_buffer.shape,
        dtype=input_buffer.dtype,
        device=input_buffer.device,
    )

    ops.misc_ops.CumSum(
        output,
        input_buffer,
        axis,
        exclusive,
        reverse,
    )

    return [output]


# Layer norm operations


@register_op_handler(mo.LayerNormOp)
def _handle_layer_norm(
    op: mo.LayerNormOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.layer_norm by dispatching to Mojo layer_norm kernel.

    Args:
        op: The layer_norm operation.
        inputs: Input buffers - input tensor, gamma, beta, epsilon.
            Note: epsilon is always on CPU (MO_SingleDeviceWithHostOperands).

    Returns:
        List containing the normalized tensor buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # gamma
    assert isinstance(inputs[2], Buffer)  # beta
    assert isinstance(inputs[3], Buffer)  # epsilon (always CPU)

    # Output shape = input shape (trivial, no Mojo shape delegation)
    output = Buffer(
        shape=inputs[0].shape,
        dtype=inputs[0].dtype,
        device=target_device,
    )

    ops.layer_norm_ops.LayerNorm(
        output,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        target_device._device_context_ptr(),
    )

    return [output]


# Range operations


@register_op_handler(mo.RangeOp)
def _handle_range(
    op: mo.RangeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.range by dispatching to Mojo range kernel.

    Args:
        op: The range operation.
        inputs: Input buffers - start, limit, step (all scalar tensors on CPU).

    Returns:
        List containing the range tensor buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)

    start_buffer = inputs[0]
    stop_buffer = inputs[1]
    step_buffer = inputs[2]

    # Compute output size from inputs
    shape = result_type.shape
    if graph.Shape.is_static(shape):
        output_shape = graph.Shape(shape).static_dims
    else:
        size = int(
            ops.misc_ops.RangeShape(start_buffer, stop_buffer, step_buffer)
        )
        output_shape = [size]

    # Allocate output buffer
    output = Buffer(
        shape=output_shape, dtype=result_type.dtype, device=target_device
    )

    # Call Mojo kernel
    ops.misc_ops.Range(
        output,
        start_buffer,
        stop_buffer,
        step_buffer,
        target_device._device_context_ptr(),
    )

    return [output]


# Random operations


@register_op_handler(mo.RandomNormalOp)
def _handle_random_normal(
    op: mo.RandomNormalOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.random.normal by dispatching to Mojo random normal kernel.

    Args:
        op: The random normal operation.
        inputs: Input buffers - shape, mean, variance (std), seed
            (all scalar/1D tensors on CPU per MO_SingleDeviceWithHostOperands).

    Returns:
        List containing the random normal tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # shape
    assert isinstance(inputs[1], Buffer)  # mean
    assert isinstance(inputs[2], Buffer)  # variance (std)
    assert isinstance(inputs[3], Buffer)  # seed

    # Extract output shape from shape tensor (on CPU)
    output_shape = inputs[0].to_numpy().tolist()

    # Extract scalar params from CPU buffers
    mean_val = float(inputs[1].to_numpy().item())
    variance_val = float(inputs[2].to_numpy().item())
    seed_val = int(inputs[3].to_numpy().item())

    # Get dtype from MLIR type directly (safe with parametric shapes)
    result_mlir_type: mo.TensorType = list(op.results)[0].type  # type: ignore[assignment]
    output_dtype = result_mlir_type.dtype

    # Allocate output buffer on target device
    output = Buffer(
        shape=output_shape,
        dtype=output_dtype,
        device=target_device,
    )

    ops.misc_ops.RandomNormal(
        output,
        mean_val,
        variance_val,
        seed_val,
        target_device._device_context_ptr(),
    )
    return [output]


@register_op_handler(mo.RandomUniformOp)
def _handle_random_uniform(
    op: mo.RandomUniformOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.random.uniform by dispatching to Mojo random uniform kernel.

    Args:
        op: The random uniform operation.
        inputs: Input buffers - shape, lower_bound, upper_bound, seed
            (all scalar/1D tensors on CPU per MO_SingleDeviceWithHostOperands).

    Returns:
        List containing the random uniform tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # shape
    assert isinstance(inputs[1], Buffer)  # lower_bound
    assert isinstance(inputs[2], Buffer)  # upper_bound
    assert isinstance(inputs[3], Buffer)  # seed

    # Extract output shape from shape tensor (on CPU)
    output_shape = inputs[0].to_numpy().tolist()

    # Extract scalar params from CPU buffers
    lower_val = float(inputs[1].to_numpy().item())
    upper_val = float(inputs[2].to_numpy().item())
    seed_val = int(inputs[3].to_numpy().item())

    # Get dtype from MLIR type directly (safe with parametric shapes)
    result_mlir_type: mo.TensorType = list(op.results)[0].type  # type: ignore[assignment]
    output_dtype = result_mlir_type.dtype

    # Allocate output buffer on target device
    output = Buffer(
        shape=output_shape,
        dtype=output_dtype,
        device=target_device,
    )

    ops.misc_ops.RandomUniform(
        output,
        lower_val,
        upper_val,
        seed_val,
        target_device._device_context_ptr(),
    )
    return [output]


# Select operations


@register_op_handler(mo.SelectOp)
def _handle_select(
    op: mo.SelectOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.select by dispatching to Mojo select kernel.

    Performs element-wise selection: result = cond ? x : y.

    Args:
        op: The select operation.
        inputs: Input buffers - cond (bool tensor), x (true values),
            y (false values).

    Returns:
        List containing the selected tensor buffer.
    """
    assert isinstance(inputs[0], Buffer)  # cond
    assert isinstance(inputs[1], Buffer)  # x (true values)
    assert isinstance(inputs[2], Buffer)  # y (false values)

    target_device = _get_target_device(op)
    _check_buffers_on_device(inputs, target_device)

    # Output dtype matches x/y dtype (not cond dtype which is bool)
    output = Buffer(
        shape=inputs[1].shape,
        dtype=inputs[1].dtype,
        device=target_device,
    )

    ops.elementwise_comparison_ops.Select(
        output,
        inputs[0],
        inputs[1],
        inputs[2],
        target_device._device_context_ptr(),
    )

    return [output]


# Concat operations


@register_op_handler(mo.ConcatOp)
def _handle_concat(
    op: mo.ConcatOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.concat by concatenating input tensors along a given axis.

    Uses a Mojo memcpy kernel to copy contiguous slices from each input into
    the output buffer, supporting both CPU and GPU.

    The axis operand is the first input (a scalar tensor on CPU), followed
    by the variadic tensor inputs to concatenate.

    Args:
        op: The concat operation.
        inputs: Input buffers - first is the axis tensor (scalar si64 on CPU),
            remaining are the tensors to concatenate.

    Returns:
        List containing the concatenated tensor buffer.
    """
    target_device = _get_target_device(op)

    # First operand is the axis (scalar tensor on CPU)
    assert isinstance(inputs[0], Buffer)
    axis = int(inputs[0].to_numpy().item())

    # Remaining operands are the tensors to concatenate
    tensor_inputs: list[Buffer] = []
    for buf in inputs[1:]:
        assert isinstance(buf, Buffer)
        tensor_inputs.append(buf)
    assert len(tensor_inputs) >= 1, (
        "ConcatOp requires at least one input tensor"
    )
    _check_buffers_on_device(tensor_inputs, target_device)

    # Normalize negative axis
    ndim = len(tensor_inputs[0].shape)
    if axis < 0:
        axis += ndim

    # Compute output shape
    output_shape = list(tensor_inputs[0].shape)
    output_shape[axis] = sum(inp.shape[axis] for inp in tensor_inputs)

    output = Buffer(
        shape=output_shape, dtype=tensor_inputs[0].dtype, device=target_device
    )
    ctx_ptr = target_device._device_context_ptr()

    # Decompose into contiguous memcpy calls.
    # For axis=0, outer_size=1 so we get one call per input (optimal).
    outer_size = prod(output_shape[:axis]) if axis > 0 else 1
    suffix_size = prod(output_shape[axis + 1 :]) if axis < ndim - 1 else 1
    out_axis_stride = output_shape[axis] * suffix_size

    dst_axis_offset = 0
    for inp in tensor_inputs:
        inner_count = inp.shape[axis] * suffix_size
        inp_stride = inner_count
        for outer_idx in range(outer_size):
            ops.data_movement_ops.Memcpy(
                output,
                inp,
                outer_idx * out_axis_stride + dst_axis_offset * suffix_size,
                outer_idx * inp_stride,
                inner_count,
                ctx_ptr,
            )
        dst_axis_offset += inp.shape[axis]

    return [output]
