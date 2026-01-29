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
from max import _core
from max._core.dialects import mo, mosh
from max._core.dialects.m import ArrayElementsAttr
from max.driver import Buffer, Device
from max.dtype import DType


def _elements_attr_to_numpy(attr: Any, dtype: DType, shape: list[int]) -> Any:
    """Convert an ElementsAttr to a numpy array.

    Args:
        attr: The elements attribute (dense array).
        dtype: The target dtype.
        shape: The target shape.

    Returns:
        A numpy array with the constant values.
    """

    # Handle M::ArrayElementsAttr - has a data property with PrimitiveArrayAttr
    if isinstance(attr, ArrayElementsAttr):
        # ArrayElementsAttr.data is a PrimitiveArrayAttr
        # PrimitiveArrayAttr.data returns Sequence[int] (raw bytes)
        primitive_attr = attr.data
        raw_bytes = bytes(primitive_attr.data)
        np_dtype = dtype.to_numpy()
        return np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

    # ElementsAttr should have methods to extract values
    # Try various approaches based on the attribute type
    if hasattr(attr, "to_numpy"):
        return attr.to_numpy()

    if hasattr(attr, "__iter__"):
        # Try to iterate over values
        values = list(attr)
        np_dtype = dtype.to_numpy()
        return np.array(values, dtype=np_dtype).reshape(shape)

    # Last resort: try to get raw data if available
    if hasattr(attr, "raw_data"):
        np_dtype = dtype.to_numpy()
        return np.frombuffer(attr.raw_data, dtype=np_dtype).reshape(shape)

    raise ValueError(f"Cannot convert attribute to numpy: {type(attr)}")


# Type alias for op handlers
# Signature: (interpreter, op, input_buffers) -> output_buffers
OpHandler = Callable[
    [list[Device], Any, Sequence[Buffer | None]],
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
        def _handle_add(devices, op, inputs):
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


# Helper functions for type conversion and device handling


def _find_device(devices: list[Device], label: str, device_id: int) -> Device:
    """Find a device matching the given label and ID.

    Args:
        devices: List of available devices.
        label: Device label ("cpu" or "gpu").
        device_id: Device ID.

    Returns:
        The matching device, or the first host device as fallback.
    """
    # Normalize label for comparison
    is_host = label.lower() == "cpu"

    for device in devices:
        if is_host and device.is_host:
            return device
        elif not is_host and not device.is_host:
            # For GPU, we could also check device ID if needed
            # For now, return first non-host device
            return device

    # Fallback: return first device that matches host requirement
    for device in devices:
        if device.is_host:
            return device

    # Last resort: return first device
    return devices[0]


def _get_output_device(op: _core.Operation, devices: list[Device]) -> Device:
    """Get the target device for an operation's output.

    Extracts the device from the operation's first result type. For MO tensor
    types, the device is encoded in the device_ref attribute.

    Args:
        op: The operation to get the output device for.
        devices: List of available devices to match against.

    Returns:
        The target device for the operation's output.
    """
    # Get the result type
    results = list(op.results)
    if not results:
        # Operations without results - use CPU as default
        return _find_device(devices, "cpu", 0)

    result_type = results[0].type
    if isinstance(result_type, mo.TensorType):
        device_ref = result_type.device_ref
        return _find_device(devices, device_ref.label, device_ref.id)

    # Fallback to CPU for non-tensor types
    return _find_device(devices, "cpu", 0)


def _get_operand_value(operand: Any) -> Any:
    """Get the Value from an operand, handling both OpOperand and Value types.

    op.operands can return either OpOperand objects (which have a .value property)
    or Value objects directly. This helper handles both cases.

    Args:
        operand: Either an OpOperand or a Value.

    Returns:
        The underlying Value object.
    """
    if hasattr(operand, "value"):
        return operand.value
    return operand


def _extract_static_shape(shape_attr: Any) -> list[int] | None:
    """Extract static shape from a shape attribute.

    Returns None if the shape contains dynamic dimensions.

    The shape_attr is typically a MOSH::ShapeAttr with a `values` property
    containing TypedAttr elements. For static shapes, these are IntegerAttrs.
    """
    try:
        # Handle MOSH::ShapeAttr - has a values property
        if isinstance(shape_attr, mosh.ShapeAttr):
            shape = []
            for dim_attr in shape_attr.values:
                # Check if it's an IntegerAttr (has .value property with int)
                if hasattr(dim_attr, "value"):
                    val = dim_attr.value
                    if isinstance(val, int):
                        shape.append(val)
                    else:
                        # Could be a symbolic dimension
                        return None
                else:
                    # Dynamic or symbolic dimension
                    return None
            return shape

        # Fallback: check if it's directly iterable with value properties
        if hasattr(shape_attr, "__iter__"):
            shape = []
            for dim in shape_attr:
                if hasattr(dim, "value"):
                    # It's an IntegerAttr or similar
                    shape.append(int(dim.value))
                elif isinstance(dim, int):
                    shape.append(dim)
                else:
                    # Dynamic dimension
                    return None
            return shape

        return None
    except (TypeError, AttributeError):
        return None


# Constant operations


@register_op_handler(mo.ConstantOp)
def _handle_constant(
    devices: list[Device], op: mo.ConstantOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.constant by materializing its value.

    Constants are mo.constant ops with embedded #M.dense_array values in the
    'value' attribute.

    Args:
        devices: List of available devices.
        op: The constant operation.
        inputs: Input buffers (empty for constants).

    Returns:
        List containing the materialized constant buffer.
    """
    # Extract the result type to get dtype and shape info
    result = list(op.results)[0]
    result_type = result.type

    if isinstance(result_type, mo.TensorType):
        dtype = DType(result_type.dtype.value)
        # Extract shape from shape_attr
        # For now, handle static shapes only
        shape = _extract_static_shape(result_type.shape_attr)
        if shape is None:
            raise NotImplementedError(
                "Dynamic shapes not yet supported for constants"
            )
    else:
        raise NotImplementedError(
            f"Constant with non-tensor type: {result_type}"
        )

    # Extract the value attribute
    # The value is an ElementsAttr (dense array)
    value_attr = op.value
    # Convert to numpy and create buffer
    # ElementsAttr should support conversion to numpy
    try:
        # TODO(EMF-95): Do not convert to intermediate numpy array here.
        numpy_array = _elements_attr_to_numpy(value_attr, dtype, shape)
        buffer = Buffer.from_numpy(numpy_array)
        # Move to the appropriate device specified by the tensor type
        target_device = _get_output_device(op, devices)
        if not target_device.is_host:
            buffer = buffer.to(target_device)
        return [buffer]
    except Exception as e:
        raise NotImplementedError(f"Failed to materialize constant: {e}") from e


# Mutable load operations


@register_op_handler(mo.MutableLoadOp)
def _handle_mutable_load(
    devices: list[Device], op: mo.MutableLoadOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.mutable.load by passing through the input buffer.

    mo.mutable.load reads from a buffer input. The handler receives the
    buffer as the first input (already resolved from slots by the dispatcher).
    The second input is the chain (None since chains are skipped).

    Args:
        devices: List of available devices (unused).
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
    devices: list[Device], op: mo.RebindOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.rebind by passing through the input buffer.

    Rebind is a shape assertion that doesn't change the underlying data.

    Args:
        devices: List of available devices (unused).
        op: The rebind operation (unused).
        inputs: Input buffers - contains the tensor to rebind.

    Returns:
        List containing the input buffer unchanged.
    """
    return [inputs[0]]


@register_op_handler(mo.StaticBroadcastToOp)
def _handle_static_broadcast_to(
    devices: list[Device],
    op: mo.StaticBroadcastToOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.static.broadcast_to by broadcasting to the target shape.

    Args:
        devices: List of available devices.
        op: The static broadcast operation.
        inputs: Input buffers - contains the tensor to broadcast.

    Returns:
        List containing the broadcast tensor buffer.
    """
    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()

    # Get target shape from result type
    result = list(op.results)[0]
    result_type = result.type
    target_shape: tuple[int, ...] | None = None
    if isinstance(result_type, mo.TensorType):
        static_shape = _extract_static_shape(result_type.shape_attr)
        if static_shape is not None:
            target_shape = tuple(static_shape)

    if target_shape is None:
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )

    # Perform broadcast using numpy
    broadcast_np = np.broadcast_to(input_np, target_shape)
    # broadcast_to returns a view, make a copy
    output_np = broadcast_np.copy()
    output_buffer = Buffer.from_numpy(output_np)
    # Move to target device specified by result type
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output_buffer = output_buffer.to(target_device)
    return [output_buffer]


@register_op_handler(mo.BroadcastToOp)
def _handle_broadcast_to(
    devices: list[Device], op: mo.BroadcastToOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.broadcast_to by broadcasting to the target shape.

    Args:
        devices: List of available devices.
        op: The broadcast operation.
        inputs: Input buffers - first is the tensor to broadcast,
            second (optional) is the target shape tensor.

    Returns:
        List containing the broadcast tensor buffer.
    """
    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()

    # Try to get target shape from result type first (static case)
    result = list(op.results)[0]
    result_type = result.type
    target_shape: tuple[int, ...] | None = None
    if isinstance(result_type, mo.TensorType):
        static_shape = _extract_static_shape(result_type.shape_attr)
        if static_shape is not None:
            target_shape = tuple(static_shape)

    # For dynamic shapes, get from the new_shape operand
    if target_shape is None and len(inputs) > 1:
        assert isinstance(inputs[1], Buffer)
        target_shape = tuple(inputs[1].to_numpy().tolist())

    if target_shape is None:
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )

    # Perform broadcast using numpy
    broadcast_np = np.broadcast_to(input_np, target_shape)
    # broadcast_to returns a view, make a copy
    output_np = broadcast_np.copy()
    output_buffer = Buffer.from_numpy(output_np)
    # Move to target device specified by result type
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output_buffer = output_buffer.to(target_device)
    return [output_buffer]


# Binary elementwise operations


def binary_elementwise_handler(op_type: type) -> OpHandler:
    op_binding = ops.BINARY_ELEMENTWISE[op_type]

    def handler(
        devices: list[Device],
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        assert isinstance(inputs[0], Buffer)
        assert isinstance(inputs[1], Buffer)
        input_dtype: DType = inputs[0].dtype
        output = Buffer(
            shape=inputs[0].shape,
            dtype=input_dtype,
            device=_get_output_device(op, devices),
        )
        op_binding(output, inputs[0], inputs[1])
        return [output]

    return handler


for op_type in ops.BINARY_ELEMENTWISE:
    register_op_handler(op_type)(binary_elementwise_handler(op_type))


def binary_comparison_handler(op_type: type) -> OpHandler:
    op_binding = ops.BINARY_ELEMENTWISE_COMPARISON[op_type]

    def handler(
        devices: list[Device],
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        assert isinstance(inputs[0], Buffer)
        assert isinstance(inputs[1], Buffer)
        output = Buffer(
            shape=inputs[0].shape,
            dtype=DType.bool,
            device=_get_output_device(op, devices),
        )
        op_binding(output, inputs[0], inputs[1])
        return [output]

    return handler


for op_type in ops.BINARY_ELEMENTWISE_COMPARISON:
    register_op_handler(op_type)(binary_comparison_handler(op_type))


# Unary elementwise operations


def unary_elementwise_handler(op_type: type) -> OpHandler:
    op_binding = ops.UNARY_ELEMENTWISE[op_type]

    def handler(
        devices: list[Device],
        op: _core.Operation,
        inputs: Sequence[Buffer | None],
    ) -> Sequence[Buffer]:
        assert isinstance(inputs[0], Buffer)
        input_dtype: DType = inputs[0].dtype
        output = Buffer(
            shape=inputs[0].shape,
            dtype=input_dtype,
            device=_get_output_device(op, devices),
        )
        op_binding(output, inputs[0])
        return [output]

    return handler


for op_type in ops.UNARY_ELEMENTWISE:
    register_op_handler(op_type)(unary_elementwise_handler(op_type))

# Matrix operations


@register_op_handler(mo.MatmulOp)
def _handle_matmul(
    devices: list[Device], op: mo.MatmulOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.matmul by dispatching to matmul kernel."""
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    lhs_np = inputs[0].to_numpy()
    rhs_np = inputs[1].to_numpy()
    result_np = np.matmul(lhs_np, rhs_np)
    output = Buffer.from_numpy(result_np)
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output = output.to(target_device)
    return [output]


# Shape manipulation operations


def _reshape_common(
    devices: list[Device],
    op: _core.Operation,
    inputs: Sequence[Buffer | None],
    op_name: str,
) -> Sequence[Buffer]:
    """Common implementation for reshape operations."""
    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()
    # Get target shape from result type
    result = list(op.results)[0]
    result_type = result.type
    if isinstance(result_type, mo.TensorType):
        target_shape = _extract_static_shape(result_type.shape_attr)
        if target_shape is None:
            raise NotImplementedError(
                f"Dynamic shapes not supported for {op_name}"
            )
    else:
        raise NotImplementedError(
            f"{op_name} with non-tensor result: {result_type}"
        )

    result_np = input_np.reshape(target_shape)
    output = Buffer.from_numpy(result_np)
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output = output.to(target_device)
    return [output]


@register_op_handler(mo.ReshapeOp)
def _handle_reshape(
    devices: list[Device], op: mo.ReshapeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reshape."""
    return _reshape_common(devices, op, inputs, "reshape")


@register_op_handler(mo.StaticReshapeOp)
def _handle_static_reshape(
    devices: list[Device],
    op: mo.StaticReshapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.static.reshape - reshape without inferred dimensions."""
    return _reshape_common(devices, op, inputs, "static reshape")


@register_op_handler(mo.TransposeOp)
def _handle_transpose(
    devices: list[Device], op: mo.TransposeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.transpose."""
    assert isinstance(inputs[0], Buffer)
    input_np = inputs[0].to_numpy()
    # TransposeOp should have a permutation attribute
    # For now, use default transpose (reverse axes)
    if hasattr(op, "permutation"):
        perm = list(op.permutation)
        result_np = np.transpose(input_np, axes=perm)
    else:
        result_np = np.transpose(input_np)
    output = Buffer.from_numpy(result_np)
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output = output.to(target_device)
    return [output]


@register_op_handler(mo.SliceOp)
def _handle_slice(
    devices: list[Device], op: mo.SliceOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.slice - tensor slicing with start/stop/step.

    The op takes (input, start, stop, step) tensors where start/stop/step
    are 1D tensors with one element per dimension of the input.
    """
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
    output = Buffer.from_numpy(result_np)
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output = output.to(target_device)
    return [output]


# Shape/parameter operations


@register_op_handler(mo.ShapeOfOp)
def _handle_shape_of(
    devices: list[Device], op: mo.ShapeOfOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.shape_of - returns the shape of a tensor as a 1D si64 tensor."""
    assert isinstance(inputs[0], Buffer)
    shape = inputs[0].shape
    result_np = np.array(shape, dtype=np.int64)
    output = Buffer.from_numpy(result_np)
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output = output.to(target_device)
    return [output]


@register_op_handler(mo.BroadcastShapeOp)
def _handle_broadcast_shape(
    devices: list[Device],
    op: mo.BroadcastShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.broadcast_shape - compute broadcast shape of two shapes."""
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    shape_x = tuple(inputs[0].to_numpy().tolist())
    shape_y = tuple(inputs[1].to_numpy().tolist())
    result_shape = np.broadcast_shapes(shape_x, shape_y)
    result_np = np.array(result_shape, dtype=np.int64)
    output = Buffer.from_numpy(result_np)
    target_device = _get_output_device(op, devices)
    if not target_device.is_host:
        output = output.to(target_device)
    return [output]


@register_op_handler(mo.ShapeToTensorOp)
def _handle_shape_to_tensor(
    devices: list[Device],
    op: mo.ShapeToTensorOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.shape.to_tensor - converts shape value to tensor.

    The input is a !mosh.ape shape value (already a buffer from ParamToValueOp).
    This op just passes through the buffer since ParamToValueOp already
    created a tensor representation.
    """
    # The input should already be a buffer containing the shape values
    # Just pass it through
    assert isinstance(inputs[0], Buffer)
    return [inputs[0]]


@register_op_handler(mosh.ParamToValueOp)
def _handle_param_to_value(
    devices: list[Device],
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
