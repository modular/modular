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
    op: mo.StaticBroadcastToOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.static.broadcast_to by broadcasting to the target shape.

    Args:
        op: The static broadcast operation.
        inputs: Input buffers - contains the tensor to broadcast.

    Returns:
        List containing the broadcast tensor buffer.
    """
    # Get target shape from result type
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()

    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )
    target_shape = graph.Shape(shape).static_dims

    # Perform broadcast using numpy
    broadcast_np = np.broadcast_to(input_np, target_shape)
    # broadcast_to returns a view, make a copy
    output_np = broadcast_np.copy()
    return [Buffer.from_numpy(output_np)]


@register_op_handler(mo.BroadcastToOp)
def _handle_broadcast_to(
    op: mo.BroadcastToOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.broadcast_to by broadcasting to the target shape.

    Args:
        op: The broadcast operation.
        inputs: Input buffers - first is the tensor to broadcast,
            second (optional) is the target shape tensor.

    Returns:
        List containing the broadcast tensor buffer.
    """
    # Get target device from result type and check CPU-only
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()

    shape = result_type.shape
    if graph.Shape.is_static(shape):
        target_shape = graph.Shape(shape).static_dims
    elif len(inputs) > 1:
        # For dynamic shapes, get from the new_shape operand
        assert isinstance(inputs[1], Buffer)
        target_shape = inputs[1].to_numpy().tolist()
    else:
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )

    # Perform broadcast using numpy
    broadcast_np = np.broadcast_to(input_np, target_shape)
    # broadcast_to returns a view, make a copy
    output_np = broadcast_np.copy()
    return [Buffer.from_numpy(output_np)]


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

        result_type = graph.Type.from_mlir(list(op.results)[0].type)
        assert isinstance(result_type, graph.TensorType)
        target_device = result_type.device.to_device()
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

        result_type = graph.Type.from_mlir(list(op.results)[0].type)
        assert isinstance(result_type, graph.TensorType)
        target_device = result_type.device.to_device()
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

        result_type = graph.Type.from_mlir(list(op.results)[0].type)
        assert isinstance(result_type, graph.TensorType)
        target_device = result_type.device.to_device()
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

# Matrix operations


@register_op_handler(mo.MatmulOp)
def _handle_matmul(
    op: mo.MatmulOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.matmul by dispatching to Mojo matmul kernel."""
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    lhs = inputs[0]
    rhs = inputs[1]
    assert isinstance(lhs, Buffer)
    assert isinstance(rhs, Buffer)

    # Calculate output shape: (M, K) @ (K, N) -> (M, N)
    m = lhs.shape[0]
    n = rhs.shape[1]

    output = Buffer(shape=(m, n), dtype=lhs.dtype, device=target_device)

    ops.mojo_ops.Matmul(output, lhs, rhs)
    return [output]


# Shape manipulation operations


def _reshape_common(
    op: _core.Operation,
    inputs: Sequence[Buffer | None],
    op_name: str,
) -> Sequence[Buffer]:
    """Common implementation for reshape operations."""
    # Get target device from result type and check CPU-only
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()

    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(f"Dynamic shapes not supported for {op_name}")
    target_shape = graph.Shape(shape).static_dims

    result_np = input_np.reshape(target_shape)
    return [Buffer.from_numpy(result_np)]


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


@register_op_handler(mo.TransposeOp)
def _handle_transpose(
    op: mo.TransposeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.transpose."""
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()
    # TransposeOp should have a permutation attribute
    # For now, use default transpose (reverse axes)
    if hasattr(op, "permutation"):
        perm = list(op.permutation)
        result_np = np.transpose(input_np, axes=perm)
    else:
        result_np = np.transpose(input_np)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.SliceOp)
def _handle_slice(
    op: mo.SliceOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.slice - tensor slicing with start/stop/step.

    The op takes (input, start, stop, step) tensors where start/stop/step
    are 1D tensors with one element per dimension of the input.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)
    assert isinstance(inputs[3], Buffer)
    input_np = inputs[0].to_numpy()
    start_np = inputs[1].to_numpy().astype(np.int64)
    stop_np = inputs[2].to_numpy().astype(np.int64)
    step_np = inputs[3].to_numpy().astype(np.int64)

    # Build slice objects for each dimension
    slices = []
    for i in range(len(start_np)):
        start_i = int(start_np[i])
        stop_i = int(stop_np[i])
        step_i = int(step_np[i])
        slices.append(slice(start_i, stop_i, step_i))

    result_np = input_np[tuple(slices)]
    # Ensure we have a contiguous array
    result_np = np.ascontiguousarray(result_np)
    return [Buffer.from_numpy(result_np)]


# Shape/parameter operations


@register_op_handler(mo.ShapeOfOp)
def _handle_shape_of(
    op: mo.ShapeOfOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.shape_of - returns the shape of a tensor as a 1D si64 tensor."""
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    assert isinstance(inputs[0], Buffer)
    shape = inputs[0].shape
    result_np = np.array(shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.BroadcastShapeOp)
def _handle_broadcast_shape(
    op: mo.BroadcastShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.broadcast_shape - compute broadcast shape of two shapes."""
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

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
