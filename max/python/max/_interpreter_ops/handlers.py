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

from collections.abc import Callable, Mapping, Sequence
from math import ceil, prod
from typing import Any

import max._interpreter_ops as ops
import numpy as np
from max import _core, graph
from max._core.dialects import builtin, mo, mosh
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


@register_op_handler(mo.ConstantScalarOp)
def _handle_constant_scalar(
    op: mo.ConstantScalarOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.constant.scalar by extracting the scalar value attribute.

    Scalar constants have a ``value`` attribute that is an ``IntegerAttr``,
    ``FloatAttr``, or ``BoolAttr``.  The result type is ``!mo.scalar<dtype>``
    which we materialise as a rank-0 ``Buffer`` on CPU.

    Args:
        op: The constant scalar operation.
        inputs: Input buffers (empty for constants).

    Returns:
        List containing a rank-0 Buffer with the scalar value.
    """
    result_type: mo.ScalarType = op.results[0].type  # type: ignore[assignment]
    dtype = DType(result_type.dtype)

    attr = op.value
    value: bool | int | float
    if isinstance(attr, builtin.BoolAttr):
        value = attr.value
    elif isinstance(attr, builtin.IntegerAttr):
        value = attr.value
    elif isinstance(attr, builtin.FloatAttr):
        value = attr.value
    else:
        raise ValueError(
            f"Unsupported scalar attribute type: {type(attr).__name__}"
        )

    np_val = np.array(value, dtype=dtype.to_numpy())
    # Rank-0 bool arrays are not supported by Buffer.from_dlpack;
    # wrap as int8 (same underlying representation).
    if np_val.dtype == np.bool_:
        np_val = np_val.view(np.int8)
    return [Buffer.from_numpy(np_val)]


# Module-level weights registry set by MOInterpreter during execution.
_weights_registry: Mapping[str, Buffer] | None = None


@register_op_handler(mo.ConstantExternalOp)
def _handle_constant_external(
    op: mo.ConstantExternalOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.constant.external by looking up the named weight.

    External constants reference named weights whose backing data is provided
    at runtime via the weights registry.  The interpreter stashes the registry
    in the module-level ``_weights_registry`` for the duration of execution.

    Args:
        op: The constant external operation.
        inputs: Input buffers (empty for constants).

    Returns:
        List containing the looked-up weight buffer.

    Raises:
        RuntimeError: If no weights registry is available or the name is
            not found.
    """
    name = op.name
    if _weights_registry is None:
        raise RuntimeError(
            f"No weights registry provided to interpreter, cannot resolve "
            f"external constant '{name}'"
        )
    if name not in _weights_registry:
        raise RuntimeError(
            f"Weight '{name}' not found in weights registry. "
            f"Available: {list(_weights_registry.keys())}"
        )
    return [_weights_registry[name]]


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


# Mutable store operations


@register_op_handler(mo.MutableStoreOp)
def _handle_mutable_store(
    op: mo.MutableStoreOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.mutable.store by copying the tensor into the buffer.

    ``mo.mutable.store`` writes a full tensor value into a mutable tensor
    slot. Operand order is ``(in_buffer, in_tensor, in_chain)`` and the sole
    result is ``out_chain``. The interpreter represents chains as ``None``.

    Args:
        op: The mutable store operation (unused).
        inputs: Input buffers - ``(in_buffer, in_tensor, in_chain)``.
            ``in_chain`` is ``None`` since chain values are skipped.

    Returns:
        List containing ``None`` for the out_chain.
    """
    in_buffer = inputs[0]
    in_tensor = inputs[1]
    assert isinstance(in_buffer, Buffer)
    assert isinstance(in_tensor, Buffer)
    in_buffer.inplace_copy_from(in_tensor)
    return [None]


@register_op_handler(mo.MutableStoreSliceOp)
def _handle_mutable_store_slice(
    op: mo.MutableStoreSliceOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.mutable.store.slice via MOGG's MutableStoreSlice kernel.

    Operand order: ``(in_buffer, slice, start, stop, step, in_chain)``;
    result is ``out_chain``. Numpy slice semantics apply to start/stop/step.

    Args:
        op: The mutable store slice operation (unused).
        inputs: ``(in_buffer, slice, start, stop, step, in_chain)``.

    Returns:
        List containing ``None`` for the out_chain.
    """
    in_buffer = inputs[0]
    slice_tensor = inputs[1]
    start_buf = inputs[2]
    stop_buf = inputs[3]
    step_buf = inputs[4]
    assert isinstance(in_buffer, Buffer)
    assert isinstance(slice_tensor, Buffer)
    assert isinstance(start_buf, Buffer)
    assert isinstance(stop_buf, Buffer)
    assert isinstance(step_buf, Buffer)

    # fp4 needs sub-byte addressing the kernel doesn't do yet.
    dtype = in_buffer.dtype
    if dtype is DType.float4_e2m1fn:
        raise NotImplementedError(
            f"mo.mutable.store.slice interpreter handler does not yet "
            f"support dtype {dtype}"
        )

    # fp8 isn't in dispatch_dtype; reinterpret as uint8 (same storage size,
    # pure byte copy).
    dst = in_buffer
    src = slice_tensor
    if dtype.is_float8():
        dst = in_buffer.view(DType.uint8)
        src = slice_tensor.view(DType.uint8)

    starts_list = [int(s) for s in start_buf.to_numpy().flatten()]
    stops_list = [int(s) for s in stop_buf.to_numpy().flatten()]
    steps_list = [int(s) for s in step_buf.to_numpy().flatten()]

    ops.data_movement_ops.MutableStoreSlice(
        dst,
        src,
        starts_list,
        stops_list,
        steps_list,
        in_buffer.device._device_context_ptr(),
    )
    return [None]


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


# Buffer operations


@register_op_handler(mo.BufferCreateOp)
def _handle_buffer_create(
    op: mo.BufferCreateOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.buffer.create by allocating a zero-filled buffer.

    ``BufferCreateOp`` has no operands and a single ``!mo.buffer<shape, dtype,
    device>`` result.  Shape, dtype, and target device are extracted from the
    result type.  The interpreter allocates a zeroed buffer so that downstream
    ops (e.g. ``buffer.transfer``) have valid storage to write into.
    """
    result_type = graph.BufferType.from_mlir(
        list(op.results)[0].type  # type: ignore[arg-type]
    )
    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(
            "Dynamic shapes not supported for buffer.create in interpreter"
        )
    target_device = result_type.device.to_device()
    buf = Buffer(
        dtype=result_type.dtype,
        shape=graph.Shape(shape).static_dims,
        device=target_device,
    )
    return [buf]


@register_op_handler(mo.BufferTransferOp)
def _handle_buffer_transfer(
    op: mo.BufferTransferOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.buffer.transfer by copying src contents into dst.

    Operand order: ``(src, dst, inChain)``.  The operation copies data from
    ``src`` into ``dst`` (both must have matching shape and dtype).  The sole
    result is an ``outChain`` which the interpreter represents as ``None``.
    """
    src = inputs[0]
    dst = inputs[1]
    assert isinstance(src, Buffer)
    assert isinstance(dst, Buffer)
    dst.inplace_copy_from(src)
    return [None]


# Debug operations


@register_op_handler(mo.DebugPrintOp)
def _handle_debug_print(
    op: mo.DebugPrintOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.debug.print by printing the string value.

    DebugPrintOp has operands (inChain) and attributes (value, label).
    It produces (outChain). The interpreter prints the string to stdout.

    Args:
        op: The debug print operation.
        inputs: Input buffers - first is the chain (None).

    Returns:
        List containing None for the output chain.
    """
    label = op.label
    value = op.value
    if label:
        print(f"[{label}] {value}")
    else:
        print(value)
    return [None]


@register_op_handler(mo.DebugTensorPrintOp)
def _handle_debug_tensor_print(
    op: mo.DebugTensorPrintOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.debug.tensor.print by printing tensor data.

    DebugTensorPrintOp has operands (inChain, input_tensor) and attribute
    (label). It produces (outChain). The interpreter converts the tensor
    buffer to numpy and prints it to stdout.

    Args:
        op: The debug tensor print operation.
        inputs: Input buffers - first is the chain (None), second is the
            tensor Buffer.

    Returns:
        List containing None for the output chain.
    """
    tensor_buf = inputs[1]
    label = op.label
    if tensor_buf is not None:
        np_array = tensor_buf.to_numpy()
        if label:
            print(f"[{label}] {np_array}")
        else:
            print(np_array)
    else:
        tag = f"[{label}] " if label else ""
        print(f"{tag}<no tensor data>")
    return [None]


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


# Shared shape/stride helpers


def _row_major_strides(shape: list[int]) -> tuple[int, ...]:
    """Compute row-major (C-order) strides for the given shape.

    Args:
        shape: Tensor dimensions in order from outermost to innermost.

    Returns:
        Tuple of strides with the same length as ``shape``.
    """
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


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
    # .to_numpy() handles GPU->CPU transfer transparently
    start_np = starts_buffer.to_numpy().astype(np.int64)
    stop_np = stops_buffer.to_numpy().astype(np.int64)
    step_np = steps_buffer.to_numpy().astype(np.int64)

    # Normalize negative starts/stops relative to input dims (NumPy
    # convention: -1 means last element, etc.).
    input_shape_np = np.array(input_buffer.shape, dtype=np.int64)
    start_np = np.where(start_np < 0, start_np + input_shape_np, start_np)
    stop_np = np.where(stop_np < 0, stop_np + input_shape_np, stop_np)

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


@register_op_handler(mo.ShapeFromTensorOp)
def _handle_shape_from_tensor(
    op: mo.ShapeFromTensorOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.shape.from_tensor - converts a tensor to a shape value.

    The input is a rank-1 integer tensor containing shape dimension values.
    The output is a !mosh.ape shape value.  In the interpreter both are
    represented as 1-D int64 Buffers, so this is a pass-through (symmetric
    with ``_handle_shape_to_tensor``).
    """
    assert isinstance(inputs[0], Buffer)
    return [inputs[0]]


@register_op_handler(mo.IndexToTensorOp)
def _handle_index_to_tensor(
    op: mo.IndexToTensorOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.index.to_tensor - wraps an SI64 scalar into a rank-0 tensor.

    The input is a scalar int64 value (stored by the interpreter as a
    1-element Buffer from ``ParamToValueOp`` or similar).  The result is
    a rank-0 ``!mo.tensor<[], si64>`` scalar tensor.
    """
    assert isinstance(inputs[0], Buffer)
    val = int(inputs[0].to_numpy().item())
    result_np = np.array(val, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


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


# ArgMax / ArgMin operations


def _argmax_min_handler(
    op: _core.Operation,
    inputs: Sequence[Buffer | None],
    kernel_fn: Any,
) -> Sequence[Buffer]:
    """Shared implementation for ArgMaxOp and ArgMinOp.

    Both ops reduce along an axis and return int64 indices. The axis operand
    is always on host (MO_SingleDeviceWithHostOperands<["axis"]>).

    Args:
        op: The argmax/argmin operation.
        inputs: Input buffers - input tensor and axis (scalar si64 on CPU).
        kernel_fn: The Mojo kernel to call (ArgMax or ArgMin).

    Returns:
        List containing the output int64 index tensor.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_buffer = inputs[0]
    axis_buffer = inputs[1]

    axis = int(axis_buffer.to_numpy().item())
    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)

    if axis < 0:
        axis += ndim

    # Normalize to 3D: [dim0, dim1, dim2]
    dim0 = prod(in_shape[:axis]) if axis > 0 else 1
    dim1 = in_shape[axis]
    dim2 = prod(in_shape[axis + 1 :]) if axis < ndim - 1 else 1

    # Output shape: same as input with shape[axis] = 1, dtype always int64
    output_shape = list(in_shape)
    output_shape[axis] = 1
    output = Buffer(shape=output_shape, dtype=DType.int64, device=target_device)

    kernel_fn(
        output,
        input_buffer,
        (dim0, dim1, dim2),
        target_device._device_context_ptr(),
    )

    return [output]


@register_op_handler(mo.ReduceArgMaxOp)
def _handle_argmax(
    op: mo.ReduceArgMaxOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reduce.arg_max by dispatching to Mojo argmax kernel."""
    return _argmax_min_handler(op, inputs, ops.argmax_ops.ArgMax)


@register_op_handler(mo.ReduceArgMinOp)
def _handle_argmin(
    op: mo.ReduceArgMinOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reduce.arg_min by dispatching to Mojo argmin kernel."""
    return _argmax_min_handler(op, inputs, ops.argmax_ops.ArgMin)


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


@register_op_handler(mo.ReduceLayerNormOp)
def _handle_layer_norm(
    op: mo.ReduceLayerNormOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reduce.layer_norm by dispatching to Mojo layer_norm kernel.

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


# RMS norm operations


@register_op_handler(mo.ReduceRmsNormOp)
def _handle_rms_norm(
    op: mo.ReduceRmsNormOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reduce.rms_norm by dispatching to Mojo rms_norm kernel.

    Args:
        op: The rms_norm operation.
        inputs: Input buffers - input tensor, weight, epsilon, weight_offset.
            Epsilon and weight_offset are always on CPU
            (MO_SingleDeviceWithHostOperands).

    Returns:
        List containing the normalized tensor buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # weight
    assert isinstance(inputs[2], Buffer)  # epsilon (always CPU)
    assert isinstance(inputs[3], Buffer)  # weight_offset (always CPU)

    output = Buffer(
        shape=inputs[0].shape,
        dtype=inputs[0].dtype,
        device=target_device,
    )

    multiply_before_cast = int(bool(op.multiply_before_cast))

    ops.rms_norm_ops.RmsNorm(
        output,
        inputs[0],
        inputs[1],
        inputs[2],
        (inputs[3], multiply_before_cast),
        target_device._device_context_ptr(),
    )

    return [output]


# Group norm operations


@register_op_handler(mo.ReduceGroupNormOp)
def _handle_group_norm(
    op: mo.ReduceGroupNormOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reduce.group_norm by dispatching to Mojo group_norm kernel.

    Args:
        op: The group_norm operation.
        inputs: Input buffers - input tensor, gamma, beta, epsilon,
            num_groups. Epsilon and num_groups are always on CPU
            (MO_SingleDeviceWithHostOperands).

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
    assert isinstance(inputs[4], Buffer)  # num_groups (always CPU, int32)

    output = Buffer(
        shape=inputs[0].shape,
        dtype=inputs[0].dtype,
        device=target_device,
    )

    ops.group_norm_ops.GroupNorm(
        output,
        inputs[0],
        inputs[1],
        inputs[2],
        (inputs[3], inputs[4]),
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

    Args:
        op: The concat operation.
        inputs: Input buffers to concatenate.

    Returns:
        List containing the concatenated tensor buffer.
    """
    target_device = _get_target_device(op)

    axis = op.axis

    tensor_inputs: list[Buffer] = []
    for buf in inputs:
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


# Gather operations


@register_op_handler(mo.GatherOp)
def _handle_gather(
    op: mo.GatherOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.gather by dispatching to Mojo gather kernel.

    Operands: input (tensor), indices (tensor), axis (scalar int64 on CPU).
    Output shape: input[:axis] + indices.shape + input[axis+1:]

    Args:
        op: The gather operation.
        inputs: Input buffers - input tensor, indices tensor, axis scalar.

    Returns:
        List containing the gathered tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # indices
    assert isinstance(inputs[2], Buffer)  # axis (scalar int64, always CPU)

    input_buffer = inputs[0]
    indices_buffer = inputs[1]

    axis = int(inputs[2].to_numpy().item())
    if axis < 0:
        axis += len(input_buffer.shape)

    in_shape = list(input_buffer.shape)
    idx_shape = list(indices_buffer.shape)

    outer_size = prod(in_shape[:axis]) if axis > 0 else 1
    axis_size = in_shape[axis]
    inner_size = prod(in_shape[axis + 1 :]) if axis < len(in_shape) - 1 else 1
    num_indices = prod(idx_shape) if idx_shape else 1

    output_shape = in_shape[:axis] + idx_shape + in_shape[axis + 1 :]

    output = Buffer(
        shape=output_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    ops.gather_scatter_ops.Gather(
        output,
        input_buffer,
        indices_buffer,
        (outer_size, axis_size, inner_size, num_indices),
        target_device._device_context_ptr(),
    )

    return [output]


@register_op_handler(mo.GatherSumOp)
def _handle_gather_sum(
    op: mo.GatherSumOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.gather_sum via NumPy gather-then-sum.

    This is a fused composite op used by DLRM-style multi-hot embeddings:
    gather along axis 0, then reduce-add along axis 1.

    ``output[i, k] = sum_j(input[indices[i, j], k])``

    Operands: input (2-D+ tensor), indices (index tensor).
    """
    target_device = _get_target_device(op)
    _check_cpu_only(op, target_device)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_np = inputs[0].to_numpy()
    indices_np = inputs[1].to_numpy().astype(np.intp)

    gathered = np.take(input_np, indices_np, axis=0)
    result = gathered.sum(axis=1, keepdims=True)

    return [Buffer.from_numpy(np.ascontiguousarray(result))]


@register_op_handler(mo.GatherNdOp)
def _handle_gather_nd(
    op: mo.GatherNdOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.gather_nd by dispatching to Mojo gather_nd kernel.

    Operands: input (tensor), indices (tensor).
    Attribute: batch_dims (int).
    Output shape: input[:batch_dims] + indices[batch_dims:-1]
                  + input[batch_dims + index_depth:]

    Args:
        op: The gather_nd operation.
        inputs: Input buffers - input tensor, indices tensor.

    Returns:
        List containing the gathered tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # indices

    input_buffer = inputs[0]
    indices_buffer = inputs[1]
    batch_dims = op.batch_dims

    in_shape = list(input_buffer.shape)
    idx_shape = list(indices_buffer.shape)
    index_depth = idx_shape[-1]

    # Compute flattened parameters for the Mojo kernel
    batch_size = prod(in_shape[:batch_dims]) if batch_dims > 0 else 1
    indices_outer_size = (
        prod(idx_shape[batch_dims:-1])
        if len(idx_shape[batch_dims:-1]) > 0
        else 1
    )
    suffix_size = (
        prod(in_shape[batch_dims + index_depth :])
        if batch_dims + index_depth < len(in_shape)
        else 1
    )
    input_inner_shape = in_shape[batch_dims:]
    input_data_stride = prod(input_inner_shape) if input_inner_shape else 1

    output_shape = (
        in_shape[:batch_dims]
        + idx_shape[batch_dims:-1]
        + in_shape[batch_dims + index_depth :]
    )

    output = Buffer(
        shape=output_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    ops.gather_scatter_ops.GatherNd(
        output,
        input_buffer,
        indices_buffer,
        (
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            input_inner_shape,
        ),
        target_device._device_context_ptr(),
    )

    return [output]


@register_op_handler(mo.ScatterNdOp)
def _handle_scatter_nd(
    op: mo.ScatterNdOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter_nd by copying input then scattering via Mojo (GPU-capable).

    Operands: input, updates, indices, outputParamDecls.
    Output: same shape/dtype as input; ``output[indices[i, :k]] = updates[i]``
    with last write winning for duplicate index vectors.

    Shape math (batch_dims == 0):
    - ``batch_size = 1``
    - ``indices_outer_size = prod(indices.shape[:-1])``
    - ``index_depth = indices.shape[-1]``
    - ``suffix_size = prod(input.shape[index_depth:])``
    - ``input_inner_shape = input.shape``

    Args:
        op: The scatter_nd operation.
        inputs: Input buffers - input, updates, indices, outputParamDecls.

    Returns:
        List containing the scatter_nd result buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # updates
    assert isinstance(inputs[2], Buffer)  # indices

    input_buffer = inputs[0]
    updates_buffer = inputs[1]
    indices_buffer = inputs[2]

    in_shape = list(input_buffer.shape)
    idx_shape = list(indices_buffer.shape)
    index_depth = idx_shape[-1]

    batch_size = 1
    indices_outer_size = prod(idx_shape[:-1]) if len(idx_shape) > 1 else 1
    suffix_size = (
        prod(in_shape[index_depth:]) if index_depth < len(in_shape) else 1
    )
    input_data_stride = prod(in_shape) if in_shape else 1
    input_inner_shape = in_shape

    output = Buffer(
        shape=in_shape, dtype=input_buffer.dtype, device=target_device
    )
    total_elements = prod(in_shape) if in_shape else 1
    ctx_ptr = target_device._device_context_ptr()

    ops.data_movement_ops.Memcpy(
        output, input_buffer, 0, 0, total_elements, ctx_ptr
    )

    if indices_outer_size > 0:
        ops.gather_scatter_ops.ScatterNd(
            output,
            updates_buffer,
            indices_buffer,
            (
                batch_size,
                indices_outer_size,
                index_depth,
                suffix_size,
                input_data_stride,
                input_inner_shape,
            ),
            ctx_ptr,
        )

    return [output]


@register_op_handler(mo.ScatterNdAddOp)
def _handle_scatter_nd_add(
    op: mo.ScatterNdAddOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter_nd.add by copying input then accumulating via Mojo (CPU-only).

    Operands: input, updates, indices, outputParamDecls.
    Output: same shape/dtype as input; ``output[indices[i, :k]] += updates[i]``
    with duplicate index vectors summed.

    Args:
        op: The scatter_nd.add operation.
        inputs: Input buffers - input, updates, indices, outputParamDecls.

    Returns:
        List containing the scatter_nd_add result buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # updates
    assert isinstance(inputs[2], Buffer)  # indices

    input_buffer = inputs[0]
    updates_buffer = inputs[1]
    indices_buffer = inputs[2]

    in_shape = list(input_buffer.shape)
    idx_shape = list(indices_buffer.shape)
    index_depth = idx_shape[-1]

    batch_size = 1
    indices_outer_size = prod(idx_shape[:-1]) if len(idx_shape) > 1 else 1
    suffix_size = (
        prod(in_shape[index_depth:]) if index_depth < len(in_shape) else 1
    )
    input_data_stride = prod(in_shape) if in_shape else 1
    input_inner_shape = in_shape

    output = Buffer(
        shape=in_shape, dtype=input_buffer.dtype, device=target_device
    )
    total_elements = prod(in_shape) if in_shape else 1
    ctx_ptr = target_device._device_context_ptr()

    ops.data_movement_ops.Memcpy(
        output, input_buffer, 0, 0, total_elements, ctx_ptr
    )

    if indices_outer_size > 0:
        ops.gather_scatter_ops.ScatterNdAdd(
            output,
            updates_buffer,
            indices_buffer,
            (
                batch_size,
                indices_outer_size,
                index_depth,
                suffix_size,
                input_data_stride,
                input_inner_shape,
            ),
            ctx_ptr,
        )

    return [output]


def _scatter_nd_reduction_common(
    op: mo.ScatterNdMaxOp | mo.ScatterNdMinOp | mo.ScatterNdMulOp,
    inputs: Sequence[Buffer | None],
    mojo_fn_name: str,
) -> Sequence[Buffer]:
    """Shared logic for scatter_nd_max/min/mul handlers.

    Copies input to output, then applies the named Mojo scatter-nd kernel.

    Args:
        op: The scatter_nd reduction operation.
        inputs: Input buffers - input, updates, indices, outputParamDecls.
        mojo_fn_name: Name of the Mojo dispatcher function on
            ``gather_scatter_ops`` (e.g. ``"ScatterNdMax"``).

    Returns:
        List containing the scatter_nd result buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # updates
    assert isinstance(inputs[2], Buffer)  # indices

    input_buffer = inputs[0]
    updates_buffer = inputs[1]
    indices_buffer = inputs[2]

    in_shape = list(input_buffer.shape)
    idx_shape = list(indices_buffer.shape)
    index_depth = idx_shape[-1]

    batch_size = 1
    indices_outer_size = prod(idx_shape[:-1]) if len(idx_shape) > 1 else 1
    suffix_size = (
        prod(in_shape[index_depth:]) if index_depth < len(in_shape) else 1
    )
    input_data_stride = prod(in_shape) if in_shape else 1
    input_inner_shape = in_shape

    output = Buffer(
        shape=in_shape, dtype=input_buffer.dtype, device=target_device
    )
    total_elements = prod(in_shape) if in_shape else 1
    ctx_ptr = target_device._device_context_ptr()

    ops.data_movement_ops.Memcpy(
        output, input_buffer, 0, 0, total_elements, ctx_ptr
    )

    if indices_outer_size > 0:
        mojo_fn = getattr(ops.gather_scatter_ops, mojo_fn_name)
        mojo_fn(
            output,
            updates_buffer,
            indices_buffer,
            (
                batch_size,
                indices_outer_size,
                index_depth,
                suffix_size,
                input_data_stride,
                input_inner_shape,
            ),
            ctx_ptr,
        )

    return [output]


@register_op_handler(mo.ScatterNdMaxOp)
def _handle_scatter_nd_max(
    op: mo.ScatterNdMaxOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter_nd.max via the shared scatter-nd reduction helper."""
    return _scatter_nd_reduction_common(op, inputs, "ScatterNdMax")


@register_op_handler(mo.ScatterNdMinOp)
def _handle_scatter_nd_min(
    op: mo.ScatterNdMinOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter_nd.min via the shared scatter-nd reduction helper."""
    return _scatter_nd_reduction_common(op, inputs, "ScatterNdMin")


@register_op_handler(mo.ScatterNdMulOp)
def _handle_scatter_nd_mul(
    op: mo.ScatterNdMulOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter_nd.mul via the shared scatter-nd reduction helper."""
    return _scatter_nd_reduction_common(op, inputs, "ScatterNdMul")


# Split operations


@register_op_handler(mo.SplitOp)
def _handle_split(
    op: mo.SplitOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.split by copying each chunk via the Mojo split kernel.

    Operands: input (device tensor), splitSizes (host int64 rank-1),
    axis (host scalar int64).
    Returns N output buffers where N = len(splitSizes).
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # splitSizes (host)
    assert isinstance(inputs[2], Buffer)  # axis (host)

    input_buffer = inputs[0]
    split_sizes = [int(s) for s in inputs[1].to_numpy().flatten()]
    axis = int(inputs[2].to_numpy().item())

    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)

    if axis < 0:
        axis += ndim

    dim0 = prod(in_shape[:axis]) if axis > 0 else 1
    in_dim1 = in_shape[axis]
    dim2 = prod(in_shape[axis + 1 :]) if axis < ndim - 1 else 1

    ctx_ptr = target_device._device_context_ptr()
    outputs: list[Buffer] = []
    axis_offset = 0

    for chunk_size in split_sizes:
        out_shape = list(in_shape)
        out_shape[axis] = chunk_size

        output = Buffer(
            shape=out_shape,
            dtype=input_buffer.dtype,
            device=target_device,
        )

        ops.split_ops.SplitCopy(
            output,
            input_buffer,
            (dim0, chunk_size, dim2, axis_offset, in_dim1),
            ctx_ptr,
        )

        outputs.append(output)
        axis_offset += chunk_size

    return outputs


# Scatter operations


@register_op_handler(mo.ScatterOp)
def _handle_scatter(
    op: mo.ScatterOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter by copying input then scattering updates via Mojo.

    Operands: input, updates, indices, axis (scalar int64 on CPU),
    outputParamDecls.
    Output: same shape/dtype as input with updates scattered along axis.

    Args:
        op: The scatter operation.
        inputs: Input buffers - input, updates, indices, axis.

    Returns:
        List containing the scattered tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # updates
    assert isinstance(inputs[2], Buffer)  # indices
    assert isinstance(inputs[3], Buffer)  # axis (scalar int64, always CPU)

    input_buffer = inputs[0]
    updates_buffer = inputs[1]
    indices_buffer = inputs[2]

    axis = int(inputs[3].to_numpy().item())
    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)
    if axis < 0:
        axis += ndim

    inner_size = prod(in_shape[axis + 1 :]) if axis < ndim - 1 else 1
    outer_size = prod(in_shape[:axis]) if axis > 0 else 1
    axis_size = in_shape[axis]
    upd_shape = list(updates_buffer.shape)
    num_updates_axis = upd_shape[axis]

    output = Buffer(
        shape=in_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    total_elements = prod(in_shape)
    ctx_ptr = target_device._device_context_ptr()

    ops.data_movement_ops.Memcpy(
        output, input_buffer, 0, 0, total_elements, ctx_ptr
    )

    ops.gather_scatter_ops.Scatter(
        output,
        updates_buffer,
        indices_buffer,
        (outer_size, axis_size, inner_size, num_updates_axis),
        ctx_ptr,
    )

    return [output]


@register_op_handler(mo.ScatterAddOp)
def _handle_scatter_add(
    op: mo.ScatterAddOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter.add by copying input then accumulating updates via Mojo.

    Operands: input, updates, indices, axis (scalar int64 on CPU),
    outputParamDecls.
    Output: same shape/dtype as input; ``output[...][indices[...]] += updates[...]``
    with duplicate indices summed.

    Args:
        op: The scatter-add operation.
        inputs: Input buffers - input, updates, indices, axis.

    Returns:
        List containing the scatter-accumulated tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # updates
    assert isinstance(inputs[2], Buffer)  # indices
    assert isinstance(inputs[3], Buffer)  # axis (scalar int64, always CPU)

    input_buffer = inputs[0]
    updates_buffer = inputs[1]
    indices_buffer = inputs[2]

    axis = int(inputs[3].to_numpy().item())
    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)
    if axis < 0:
        axis += ndim

    inner_size = prod(in_shape[axis + 1 :]) if axis < ndim - 1 else 1
    outer_size = prod(in_shape[:axis]) if axis > 0 else 1
    axis_size = in_shape[axis]
    upd_shape = list(updates_buffer.shape)
    num_updates_axis = upd_shape[axis]

    output = Buffer(
        shape=in_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    total_elements = prod(in_shape)
    ctx_ptr = target_device._device_context_ptr()

    ops.data_movement_ops.Memcpy(
        output, input_buffer, 0, 0, total_elements, ctx_ptr
    )

    ops.gather_scatter_ops.ScatterAdd(
        output,
        updates_buffer,
        indices_buffer,
        (outer_size, axis_size, inner_size, num_updates_axis),
        ctx_ptr,
    )

    return [output]


def _scatter_reduction_common(
    op: mo.ScatterMaxOp | mo.ScatterMinOp | mo.ScatterMulOp,
    inputs: Sequence[Buffer | None],
    mojo_fn_name: str,
) -> Sequence[Buffer]:
    """Shared logic for scatter_max, scatter_min, scatter_mul handlers.

    All three reductions share the same operand layout (input, updates,
    indices, axis) and the same pre-/post-processing: copy input to output,
    then apply the Mojo reduction kernel.

    Args:
        op: The scatter reduction MO operation.
        inputs: Input buffers - input, updates, indices, axis.
        mojo_fn_name: Name of the Mojo kernel function to call
            (``"ScatterMax"``, ``"ScatterMin"``, or ``"ScatterMul"``).

    Returns:
        List containing the scatter-reduced tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)
    assert isinstance(inputs[3], Buffer)

    input_buffer = inputs[0]
    updates_buffer = inputs[1]
    indices_buffer = inputs[2]

    axis = int(inputs[3].to_numpy().item())
    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)
    if axis < 0:
        axis += ndim

    inner_size = prod(in_shape[axis + 1 :]) if axis < ndim - 1 else 1
    outer_size = prod(in_shape[:axis]) if axis > 0 else 1
    axis_size = in_shape[axis]
    upd_shape = list(updates_buffer.shape)
    num_updates_axis = upd_shape[axis]

    output = Buffer(
        shape=in_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    total_elements = prod(in_shape)
    ctx_ptr = target_device._device_context_ptr()

    ops.data_movement_ops.Memcpy(
        output, input_buffer, 0, 0, total_elements, ctx_ptr
    )

    mojo_fn = getattr(ops.gather_scatter_ops, mojo_fn_name)
    mojo_fn(
        output,
        updates_buffer,
        indices_buffer,
        (outer_size, axis_size, inner_size, num_updates_axis),
        ctx_ptr,
    )

    return [output]


@register_op_handler(mo.ScatterMaxOp)
def _handle_scatter_max(
    op: mo.ScatterMaxOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter.max by copying input then applying max reduction."""
    return _scatter_reduction_common(op, inputs, "ScatterMax")


@register_op_handler(mo.ScatterMinOp)
def _handle_scatter_min(
    op: mo.ScatterMinOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter.min by copying input then applying min reduction."""
    return _scatter_reduction_common(op, inputs, "ScatterMin")


@register_op_handler(mo.ScatterMulOp)
def _handle_scatter_mul(
    op: mo.ScatterMulOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.scatter.mul by copying input then applying mul reduction."""
    return _scatter_reduction_common(op, inputs, "ScatterMul")


def _conv_out_dim(
    in_dim: int, k: int, dilation: int, stride: int, pad_total: int
) -> int:
    """Compute conv output dimension (floor mode)."""
    return 1 + (in_dim + pad_total - (1 + dilation * (k - 1))) // stride


@register_op_handler(mo.ConvOp)
def _handle_conv(
    op: mo.ConvOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.conv (2D forward convolution, NHWC + RSCF).

    Operands: input, filter, strides, dilations, paddings, num_groups.
    All shape params are host int64 tensors.

    Args:
        op: The conv operation.
        inputs: Input buffers.

    Returns:
        List containing the convolution output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input (NHWC)
    assert isinstance(inputs[1], Buffer)  # filter (RSCF)
    assert isinstance(inputs[2], Buffer)  # strides
    assert isinstance(inputs[3], Buffer)  # dilations
    assert isinstance(inputs[4], Buffer)  # paddings
    assert isinstance(inputs[5], Buffer)  # num_groups

    input_buffer = inputs[0]
    filter_buffer = inputs[1]

    strides = [int(s) for s in inputs[2].to_numpy().flatten()]
    dilations = [int(d) for d in inputs[3].to_numpy().flatten()]
    paddings = [int(p) for p in inputs[4].to_numpy().flatten()]
    groups = int(inputs[5].to_numpy().item())

    in_shape = list(input_buffer.shape)
    filt_shape = list(filter_buffer.shape)

    if len(in_shape) != 4:
        raise ValueError(
            f"conv2d expects rank-4 input, got rank {len(in_shape)}"
        )

    batch, in_h, in_w, in_c = in_shape
    kh, kw = filt_shape[0], filt_shape[1]
    out_c = filt_shape[-1]

    stride_h, stride_w = strides[0], strides[1]
    dil_h, dil_w = dilations[0], dilations[1]
    pad_h_before, pad_h_after = paddings[0], paddings[1]
    pad_w_before, pad_w_after = paddings[2], paddings[3]

    out_h = _conv_out_dim(in_h, kh, dil_h, stride_h, pad_h_before + pad_h_after)
    out_w = _conv_out_dim(in_w, kw, dil_w, stride_w, pad_w_before + pad_w_after)

    output = Buffer(
        shape=[batch, out_h, out_w, out_c],
        dtype=input_buffer.dtype,
        device=target_device,
    )
    ctx_ptr = target_device._device_context_ptr()

    ops.conv_ops.Conv2d(
        output,
        input_buffer,
        filter_buffer,
        (
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            pad_h_before,
            pad_w_before,
            groups,
            out_h,
            out_w,
        ),
        ctx_ptr,
    )

    return [output]


@register_op_handler(mo.ConvTransposeOp)
def _handle_conv_transpose(
    op: mo.ConvTransposeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.conv_transpose (2D transposed convolution, NHWC + RSCF).

    Operands: input, filter, strides, dilations, paddings, output_paddings.
    All shape params are host int64 tensors.

    Args:
        op: The conv transpose operation.
        inputs: Input buffers.

    Returns:
        List containing the transposed convolution output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input (NHWC)
    assert isinstance(inputs[1], Buffer)  # filter (RSCF)
    assert isinstance(inputs[2], Buffer)  # strides
    assert isinstance(inputs[3], Buffer)  # dilations
    assert isinstance(inputs[4], Buffer)  # paddings
    assert isinstance(inputs[5], Buffer)  # output_paddings

    input_buffer = inputs[0]
    filter_buffer = inputs[1]

    strides = [int(s) for s in inputs[2].to_numpy().flatten()]
    dilations = [int(d) for d in inputs[3].to_numpy().flatten()]
    paddings = [int(p) for p in inputs[4].to_numpy().flatten()]
    output_pads = [int(p) for p in inputs[5].to_numpy().flatten()]

    in_shape = list(input_buffer.shape)
    filt_shape = list(filter_buffer.shape)

    if len(in_shape) != 4:
        raise ValueError(
            f"conv_transpose2d expects rank-4 input, got rank {len(in_shape)}"
        )

    batch, in_h, in_w, in_c = in_shape
    kh, kw = filt_shape[0], filt_shape[1]
    out_c = filt_shape[2]

    stride_h, stride_w = strides[0], strides[1]
    dil_h, dil_w = dilations[0], dilations[1]
    pad_h_before, pad_h_after = paddings[0], paddings[1]
    pad_w_before, pad_w_after = paddings[2], paddings[3]
    opad_h = output_pads[0] if len(output_pads) > 0 else 0
    opad_w = output_pads[1] if len(output_pads) > 1 else 0

    out_h = (
        (in_h - 1) * stride_h
        - pad_h_before
        - pad_h_after
        + dil_h * (kh - 1)
        + 1
        + opad_h
    )
    out_w = (
        (in_w - 1) * stride_w
        - pad_w_before
        - pad_w_after
        + dil_w * (kw - 1)
        + 1
        + opad_w
    )

    output = Buffer(
        shape=[batch, out_h, out_w, out_c],
        dtype=input_buffer.dtype,
        device=target_device,
    )
    ctx_ptr = target_device._device_context_ptr()

    ops.conv_ops.ConvTranspose2d(
        output,
        input_buffer,
        filter_buffer,
        (
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            pad_h_before,
            pad_w_before,
            out_h,
            out_w,
        ),
        ctx_ptr,
    )

    return [output]


# Pooling operations


def _compute_pool_out_dim(
    in_dim: int,
    filter_dim: int,
    stride: int,
    dilation: int,
    pad: int,
    ceil_mode: bool,
) -> int:
    """Compute output spatial dim for a sliding window operation."""
    numerator = in_dim + pad - (dilation * (filter_dim - 1) + 1)
    if ceil_mode:
        return 1 + -(-numerator // stride)  # ceildiv
    return 1 + numerator // stride


def _handle_max_pool_impl(
    op: mo.MaxPoolOp | mo.MaxPoolCeilModeTrueOp,
    inputs: Sequence[Buffer | None],
    ceil_mode: bool,
) -> Sequence[Buffer]:
    """Shared implementation for MaxPoolOp and MaxPoolCeilModeTrueOp."""
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input (NHWC)
    assert isinstance(inputs[1], Buffer)  # filter_shape
    assert isinstance(inputs[2], Buffer)  # strides
    assert isinstance(inputs[3], Buffer)  # dilations
    assert isinstance(inputs[4], Buffer)  # paddings

    input_buffer = inputs[0]
    filter_shape = [int(x) for x in inputs[1].to_numpy().flatten()]
    strides = [int(x) for x in inputs[2].to_numpy().flatten()]
    dilations = [int(x) for x in inputs[3].to_numpy().flatten()]
    paddings = [int(x) for x in inputs[4].to_numpy().flatten()]

    in_shape = list(input_buffer.shape)
    batch = in_shape[0]
    in_h = in_shape[1]
    in_w = in_shape[2]
    channels = in_shape[3]

    filter_h, filter_w = filter_shape[0], filter_shape[1]
    stride_h, stride_w = strides[0], strides[1]
    dilation_h, dilation_w = dilations[0], dilations[1]
    pad_h_before, pad_h_after = paddings[0], paddings[1]
    pad_w_before, pad_w_after = paddings[2], paddings[3]

    out_h = _compute_pool_out_dim(
        in_h,
        filter_h,
        stride_h,
        dilation_h,
        pad_h_before + pad_h_after,
        ceil_mode,
    )
    out_w = _compute_pool_out_dim(
        in_w,
        filter_w,
        stride_w,
        dilation_w,
        pad_w_before + pad_w_after,
        ceil_mode,
    )

    output = Buffer(
        shape=[batch, out_h, out_w, channels],
        dtype=input_buffer.dtype,
        device=target_device,
    )

    ctx_ptr = target_device._device_context_ptr()
    kernel_fn = (
        ops.pooling_ops.MaxPoolCeil if ceil_mode else ops.pooling_ops.MaxPool
    )
    kernel_fn(
        output,
        input_buffer,
        (
            batch,
            in_h,
            in_w,
            channels,
            out_h,
            out_w,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_h_before,
            pad_w_before,
        ),
        ctx_ptr,
    )

    return [output]


@register_op_handler(mo.MaxPoolOp)
def _handle_max_pool(
    op: mo.MaxPoolOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.max_pool via Mojo max pooling kernel (floor mode)."""
    return _handle_max_pool_impl(op, inputs, ceil_mode=False)


@register_op_handler(mo.MaxPoolCeilModeTrueOp)
def _handle_max_pool_ceil(
    op: mo.MaxPoolCeilModeTrueOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.max_pool_ceil_mode_true via Mojo max pooling kernel."""
    return _handle_max_pool_impl(op, inputs, ceil_mode=True)


@register_op_handler(mo.TileOp)
def _handle_tile(
    op: mo.TileOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.tile by repeating the input along each dimension.

    Operands: input (device tensor), repeats (host int64 rank-1).
    Output shape[i] = input shape[i] * repeats[i].
    CPU-only (mo.tile is MO_HostOnly).

    Args:
        op: The tile operation.
        inputs: Input buffers - [input_tensor, repeats_tensor].

    Returns:
        List containing the tiled output tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # repeats (host int64)

    input_buffer = inputs[0]
    repeats = [int(r) for r in inputs[1].to_numpy().flatten()]

    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    out_shape = [in_shape[i] * repeats[i] for i in range(rank)]

    in_strides = _row_major_strides(in_shape)
    out_strides = _row_major_strides(out_shape)

    output = Buffer(
        shape=out_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    ctx_ptr = target_device._device_context_ptr()

    ops.tile_ops.Tile(
        output,
        input_buffer,
        (tuple(in_shape), out_strides, in_strides, rank),
        ctx_ptr,
    )

    return [output]


@register_op_handler(mo.LinalgBandPartOp)
def _handle_band_part(
    op: mo.LinalgBandPartOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.linalg.band_part (matrix band part masking).

    Operands: input (device), num_lower (host int64 scalar),
    num_upper (host int64 scalar), exclude (host bool scalar).

    Args:
        op: The band_part operation.
        inputs: Input buffers.

    Returns:
        List containing the masked output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # num_lower
    assert isinstance(inputs[2], Buffer)  # num_upper
    assert isinstance(inputs[3], Buffer)  # exclude

    input_buffer = inputs[0]
    num_lower = int(inputs[1].to_numpy().item())
    num_upper = int(inputs[2].to_numpy().item())
    exclude = int(inputs[3].to_numpy().item())

    in_shape = list(input_buffer.shape)
    if len(in_shape) < 2:
        raise ValueError(
            f"band_part expects rank >= 2 input, got rank {len(in_shape)}"
        )

    M = in_shape[-2]
    N = in_shape[-1]
    total = prod(in_shape)

    output = Buffer(
        shape=in_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )
    ctx_ptr = target_device._device_context_ptr()

    ops.band_part_ops.BandPart(
        output,
        input_buffer,
        (total, M, N, num_lower, num_upper, exclude),
        ctx_ptr,
    )

    return [output]


# Average pooling


def _avg_pool_common(
    op: mo.AvgPoolOp | mo.AvgPoolCeilModeTrueOp,
    inputs: Sequence[Buffer | None],
    ceil_mode: bool,
) -> Sequence[Buffer]:
    """Shared logic for avg_pool (floor) and avg_pool_ceil_mode_true.

    Args:
        op: The avg_pool operation.
        inputs: Input buffers [input, filter_shape, strides, dilations, paddings].
        ceil_mode: Whether to use ceiling mode for output shape.

    Returns:
        List containing the pooled output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # filter_shape
    assert isinstance(inputs[2], Buffer)  # strides
    assert isinstance(inputs[3], Buffer)  # dilations
    assert isinstance(inputs[4], Buffer)  # paddings

    input_buffer = inputs[0]
    filter_np = inputs[1].to_numpy().flatten()
    strides_np = inputs[2].to_numpy().flatten()
    dilations_np = inputs[3].to_numpy().flatten()
    paddings_np = inputs[4].to_numpy().flatten()

    in_shape = list(input_buffer.shape)
    if len(in_shape) != 4:
        raise ValueError(
            f"avg_pool2d expects rank-4 NHWC input, got rank {len(in_shape)}"
        )

    batch, in_h, in_w, channels = in_shape
    kH, kW = int(filter_np[0]), int(filter_np[1])
    stride_h, stride_w = int(strides_np[0]), int(strides_np[1])
    dil_h, dil_w = int(dilations_np[0]), int(dilations_np[1])
    pad_h_before = int(paddings_np[0])
    pad_h_after = int(paddings_np[1])
    pad_w_before = int(paddings_np[2])
    pad_w_after = int(paddings_np[3])

    count_boundary = bool(op.count_boundary)

    eff_kH = dil_h * (kH - 1) + 1
    eff_kW = dil_w * (kW - 1) + 1
    if ceil_mode:
        out_h = ceil(
            (in_h + pad_h_before + pad_h_after - eff_kH + 1) / stride_h
        )
        out_w = ceil(
            (in_w + pad_w_before + pad_w_after - eff_kW + 1) / stride_w
        )
    else:
        out_h = (in_h + pad_h_before + pad_h_after - eff_kH) // stride_h + 1
        out_w = (in_w + pad_w_before + pad_w_after - eff_kW) // stride_w + 1

    output = Buffer(
        shape=[batch, out_h, out_w, channels],
        dtype=input_buffer.dtype,
        device=target_device,
    )
    ctx_ptr = target_device._device_context_ptr()

    ops.avg_pool_ops.AvgPool2d(
        output,
        input_buffer,
        (
            batch,
            in_h,
            in_w,
            channels,
            out_h,
            out_w,
            kH,
            kW,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            pad_h_before,
            pad_w_before,
            int(count_boundary),
        ),
        ctx_ptr,
    )

    return [output]


@register_op_handler(mo.AvgPoolOp)
def _handle_avg_pool(
    op: mo.AvgPoolOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.avg_pool (floor-mode 2D average pooling)."""
    return _avg_pool_common(op, inputs, ceil_mode=False)


@register_op_handler(mo.AvgPoolCeilModeTrueOp)
def _handle_avg_pool_ceil(
    op: mo.AvgPoolCeilModeTrueOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.avg_pool_ceil_mode_true (ceil-mode 2D average pooling)."""
    return _avg_pool_common(op, inputs, ceil_mode=True)


# ROI Align operation


@register_op_handler(mo.RoiAlignOp)
def _handle_roi_align(
    op: mo.RoiAlignOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.roi_align (ROI Align pooling, NHWC).

    Operands: input (4D), rois (2D), output_height, output_width,
    spatial_scale, sampling_ratio.
    Attributes: aligned (bool), mode (string: "AVG" or "MAX").

    Args:
        op: The roi_align operation.
        inputs: Input buffers.

    Returns:
        List containing the ROI-aligned output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input [N, H, W, C]
    assert isinstance(inputs[1], Buffer)  # rois [M, 5]
    assert isinstance(inputs[2], Buffer)  # output_height (scalar)
    assert isinstance(inputs[3], Buffer)  # output_width (scalar)
    assert isinstance(inputs[4], Buffer)  # spatial_scale (scalar)
    assert isinstance(inputs[5], Buffer)  # sampling_ratio (scalar)

    input_buffer = inputs[0]
    rois_buffer = inputs[1]

    in_shape = list(input_buffer.shape)
    if len(in_shape) != 4:
        raise ValueError(
            f"roi_align expects rank-4 NHWC input, got rank {len(in_shape)}"
        )

    rois_shape = list(rois_buffer.shape)
    if len(rois_shape) != 2 or rois_shape[1] != 5:
        raise ValueError(
            f"roi_align expects [M, 5] rois, got shape {rois_shape}"
        )

    out_h = int(inputs[2].to_numpy().item())
    out_w = int(inputs[3].to_numpy().item())
    spatial_scale = float(inputs[4].to_numpy().item())
    sampling_ratio = float(inputs[5].to_numpy().item())

    n_regions = rois_shape[0]
    height = in_shape[1]
    width = in_shape[2]
    channels = in_shape[3]

    aligned = bool(op.aligned)
    mode_str = str(op.mode.value)
    if mode_str not in ("AVG", "MAX"):
        raise ValueError(
            f"roi_align mode must be 'AVG' or 'MAX', got '{mode_str}'"
        )
    mode_flag = 0 if mode_str == "AVG" else 1
    aligned_flag = 1 if aligned else 0

    output = Buffer(
        shape=[n_regions, out_h, out_w, channels],
        dtype=input_buffer.dtype,
        device=target_device,
    )
    ctx_ptr = target_device._device_context_ptr()

    ops.roi_align_ops.RoiAlign(
        output,
        input_buffer,
        rois_buffer,
        (
            n_regions,
            height,
            width,
            channels,
            out_h,
            out_w,
            spatial_scale,
            sampling_ratio,
            aligned_flag,
            mode_flag,
        ),
        ctx_ptr,
    )

    return [output]


# Top-K operation


@register_op_handler(mo.TopKOp)
def _handle_top_k(
    op: mo.TopKOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.top_k by dispatching to Mojo top-k kernel.

    Operands (per MO_SingleDeviceWithHostOperands<["k", "axis", "sorted"]>):
      inputs[0]: input tensor (device)
      inputs[1]: k scalar (host, int64)
      inputs[2]: axis scalar (host, int64)
      inputs[3]: sorted scalar (host, bool)

    Returns two buffers: values (same dtype as input) and indices (int64),
    both of shape input_shape with shape[axis] replaced by k.

    Note: values are always returned in descending order; the ``sorted``
    flag is accepted but currently ignored since the implementation always
    sorts.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)
    assert isinstance(inputs[3], Buffer)

    input_buffer = inputs[0]
    k = int(inputs[1].to_numpy().item())
    axis = int(inputs[2].to_numpy().item())

    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)
    if axis < 0:
        axis += ndim

    out_shape = list(in_shape)
    out_shape[axis] = k

    dim0 = prod(in_shape[:axis]) if axis > 0 else 1
    dim1 = in_shape[axis]
    dim2 = prod(in_shape[axis + 1 :]) if axis < ndim - 1 else 1

    out_vals = Buffer(
        shape=out_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )
    out_idxs = Buffer(
        shape=out_shape,
        dtype=DType.int64,
        device=target_device,
    )

    ctx_ptr = target_device._device_context_ptr()
    ops.topk_ops.TopK(
        out_vals,
        out_idxs,
        input_buffer,
        (dim0, dim1, dim2, k),
        ctx_ptr,
    )

    return [out_vals, out_idxs]


# Bottom-K operation


@register_op_handler(mo.BottomKOp)
def _handle_bottom_k(
    op: mo.BottomKOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.bottom_k by dispatching to Mojo bottom-k kernel.

    Operands (per MO_SingleDeviceWithHostOperands<["k", "axis", "sorted"]>):
      inputs[0]: input tensor (device)
      inputs[1]: k scalar (host, int64)
      inputs[2]: axis scalar (host, int64)
      inputs[3]: sorted scalar (host, bool)

    Returns two buffers: values (same dtype as input) and indices (int64),
    both of shape input_shape with shape[axis] replaced by k.

    Note: values are always returned in ascending order; the ``sorted``
    flag is accepted but currently ignored since the implementation always
    sorts.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)
    assert isinstance(inputs[3], Buffer)

    input_buffer = inputs[0]
    k = int(inputs[1].to_numpy().item())
    axis = int(inputs[2].to_numpy().item())

    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)
    if axis < 0:
        axis += ndim

    out_shape = list(in_shape)
    out_shape[axis] = k

    dim0 = prod(in_shape[:axis]) if axis > 0 else 1
    dim1 = in_shape[axis]
    dim2 = prod(in_shape[axis + 1 :]) if axis < ndim - 1 else 1

    out_vals = Buffer(
        shape=out_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )
    out_idxs = Buffer(
        shape=out_shape,
        dtype=DType.int64,
        device=target_device,
    )

    ctx_ptr = target_device._device_context_ptr()
    ops.bottomk_ops.BottomK(
        out_vals,
        out_idxs,
        input_buffer,
        (dim0, dim1, dim2, k),
        ctx_ptr,
    )

    return [out_vals, out_idxs]


# Arg-NonZero operation


@register_op_handler(mo.ArgNonzeroOp)
def _handle_arg_nonzero(
    op: mo.ArgNonzeroOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.arg_nonzero with a two-pass count-then-fill strategy.

    Operand: input (host tensor, CPU-only via MO_HostOnly trait).

    The output shape ``[nnz, rank]`` is data-dependent, so we:

    1. Run ``ArgNonZeroCount`` to determine ``nnz``.
    2. Allocate a ``[nnz, rank]`` int64 output buffer on CPU.
    3. If ``nnz > 0``, run ``ArgNonZeroFill`` to write row-major coordinates.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    input_buffer = inputs[0]

    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    numel = prod(in_shape)

    ctx_ptr = target_device._device_context_ptr()

    # Pass 1: count nonzero elements.
    count_buf = Buffer(shape=[1], dtype=DType.int64, device=CPU())
    ops.argnonzero_ops.ArgNonZeroCount(
        count_buf,
        input_buffer,
        numel,
        ctx_ptr,
    )
    nnz = int(count_buf.to_numpy().item())

    # Pass 2: fill coordinates.
    out_buf = Buffer(shape=[nnz, rank], dtype=DType.int64, device=CPU())
    if nnz > 0:
        ops.argnonzero_ops.ArgNonZeroFill(
            out_buf,
            input_buffer,
            in_shape,
            rank,
            ctx_ptr,
        )

    return [out_buf]


# Padding operations


def _pad_common(
    op: _core.Operation,
    input_buffer: Buffer,
    paddings: list[int],
    target_device: Device,
) -> tuple[list[int], list[int], tuple[int, ...], tuple[int, ...], int]:
    """Compute output shape, strides, and total elements for a pad op.

    Args:
        op: The pad operation (used only to name the caller in errors).
        input_buffer: The input tensor buffer.
        paddings: Flat list [pre_0, post_0, pre_1, post_1, ...].
        target_device: The target execution device.

    Returns:
        Tuple of (out_shape, in_shape, out_strides, in_strides, total).
    """
    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    out_shape = [
        in_shape[d] + paddings[2 * d] + paddings[2 * d + 1] for d in range(rank)
    ]
    in_strides = _row_major_strides(in_shape)
    out_strides = _row_major_strides(out_shape)
    total = prod(out_shape)
    return out_shape, in_shape, out_strides, in_strides, total


@register_op_handler(mo.PadConstantOp)
def _handle_pad_constant(
    op: mo.PadConstantOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.pad.constant via Mojo pad kernel (CPU and GPU).

    Operands (MO_SingleDeviceWithHostOperands<["paddings", "constant"]>):
      inputs[0]: input tensor (device)
      inputs[1]: paddings (host, int32|int64, shape [2*rank])
      inputs[2]: constant scalar (host, same dtype as input)

    The padded region is filled with the scalar constant; the content
    region copies from the input.

    Args:
        op: The pad_constant operation.
        inputs: Input buffers.

    Returns:
        List containing the padded output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)

    input_buffer = inputs[0]
    paddings = [int(p) for p in inputs[1].to_numpy().flatten()]
    const_addr = int(inputs[2]._data_ptr())

    out_shape, in_shape, out_strides, in_strides, total = _pad_common(
        op, input_buffer, paddings, target_device
    )

    output = Buffer(
        shape=out_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    ctx_ptr = target_device._device_context_ptr()
    ops.pad_ops.PadConstant(
        output,
        input_buffer,
        (
            paddings,
            tuple(out_shape),
            tuple(in_shape),
            out_strides,
            in_strides,
            len(in_shape),
            total,
            const_addr,
        ),
        ctx_ptr,
    )

    return [output]


@register_op_handler(mo.PadReflectOp)
def _handle_pad_reflect(
    op: mo.PadReflectOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.pad.reflect via Mojo pad kernel (CPU-only).

    Operands (MO_HostOnly):
      inputs[0]: input tensor
      inputs[1]: paddings (host, int32|int64, shape [2*rank])

    Padded cells mirror values from the content region using a periodic
    reflection with period 2*(input_dim-1) per axis.

    Note: values are always computed; the op has no ``sorted`` flag.

    Args:
        op: The pad_reflect operation.
        inputs: Input buffers.

    Returns:
        List containing the reflected-padded output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_buffer = inputs[0]
    paddings = [int(p) for p in inputs[1].to_numpy().flatten()]

    out_shape, in_shape, out_strides, in_strides, total = _pad_common(
        op, input_buffer, paddings, target_device
    )

    output = Buffer(
        shape=out_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    ops.pad_ops.PadReflect(
        output,
        input_buffer,
        (
            paddings,
            tuple(out_shape),
            tuple(in_shape),
            out_strides,
            in_strides,
            len(in_shape),
            total,
        ),
        None,
    )

    return [output]


@register_op_handler(mo.PadRepeatOp)
def _handle_pad_repeat(
    op: mo.PadRepeatOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.pad.repeat (edge pad) via Mojo pad kernel (CPU-only).

    Operands (MO_HostOnly):
      inputs[0]: input tensor
      inputs[1]: paddings (host, int32|int64, shape [2*rank])

    Padded cells are filled by clamping the output coordinate to the
    nearest valid input index per axis (nearest-edge / repeat semantics).

    Args:
        op: The pad_repeat operation.
        inputs: Input buffers.

    Returns:
        List containing the edge-padded output buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_buffer = inputs[0]
    paddings = [int(p) for p in inputs[1].to_numpy().flatten()]

    out_shape, in_shape, out_strides, in_strides, total = _pad_common(
        op, input_buffer, paddings, target_device
    )

    output = Buffer(
        shape=out_shape,
        dtype=input_buffer.dtype,
        device=target_device,
    )

    ops.pad_ops.PadRepeat(
        output,
        input_buffer,
        (
            paddings,
            tuple(out_shape),
            tuple(in_shape),
            out_strides,
            in_strides,
            len(in_shape),
            total,
        ),
        None,
    )

    return [output]


@register_op_handler(mo.ResizeLinearOp)
def _handle_resize_linear(
    op: mo.ResizeLinearOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.resize.linear via Mojo separable linear-filter resize (CPU-only).

    Operands (MO_HostOnly):
      inputs[0]: input data tensor (host)
      inputs[1]: size -- 1-D int64 tensor whose values give the full output
                 shape (one value per input rank dimension).

    Attributes on ``op``:
      ``coordinate_transform_mode`` -- int 0-3 (half_pixel / align_corners /
          asymmetric / half_pixel_1D).
      ``antialias`` -- bool; widens the tent-filter kernel when downscaling.

    Args:
        op: The resize-linear operation.
        inputs: Two buffers -- input data and size.

    Returns:
        List containing a single output buffer with shape given by ``size``
        and the same dtype as ``input``.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # size (int64 output-shape vector)

    input_buffer = inputs[0]
    size_buffer = inputs[1]

    coord_mode = int(op.coordinate_transform_mode.value)
    antialias = bool(op.antialias)

    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    out_shape = size_buffer.to_numpy().astype(int).flatten().tolist()

    assert len(out_shape) == rank, (
        f"resize_linear: size rank {len(out_shape)} != input rank {rank}"
    )

    output = Buffer(shape=out_shape, dtype=input_buffer.dtype, device=CPU())
    ops.resize_ops.ResizeLinear(
        output,
        input_buffer,
        (coord_mode, antialias, rank, in_shape, out_shape),
        target_device._device_context_ptr(),
    )
    return [output]


@register_op_handler(mo.ResizeNearestOp)
def _handle_resize_nearest(
    op: mo.ResizeNearestOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.resize.nearest via Mojo nearest-neighbor resize (CPU-only).

    Operands (MO_HostOnly):
      inputs[0]: input data tensor (host)
      inputs[1]: size -- 1-D int64 tensor whose values give the full output
                 shape (one value per input rank dimension).

    Attributes on ``op``:
      ``coordinate_transform_mode`` -- int 0-3 (half_pixel / align_corners /
          asymmetric / half_pixel_1D).
      ``round_mode`` -- int 0-3 (HalfDown / HalfUp / Floor / Ceil).

    Args:
        op: The resize-nearest operation.
        inputs: Two buffers -- input data and size.

    Returns:
        List containing a single output buffer with shape given by ``size``
        and the same dtype as ``input``.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # size (int64 output-shape vector)

    input_buffer = inputs[0]
    size_buffer = inputs[1]

    coord_mode = int(op.coordinate_transform_mode.value)
    round_mode = int(op.round_mode)

    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    out_shape = size_buffer.to_numpy().astype(int).flatten().tolist()

    assert len(out_shape) == rank, (
        f"resize_nearest: size rank {len(out_shape)} != input rank {rank}"
    )

    output = Buffer(shape=out_shape, dtype=input_buffer.dtype, device=CPU())
    ops.resize_ops.ResizeNearest(
        output,
        input_buffer,
        (coord_mode, round_mode, rank, in_shape, out_shape),
        target_device._device_context_ptr(),
    )
    return [output]


@register_op_handler(mo.ResizeBicubicOp)
def _handle_resize_bicubic(
    op: mo.ResizeBicubicOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.resize.bicubic via Mojo CPU bicubic kernel.

    Operands:
      inputs[0]: input data tensor (rank-4 NCHW).
      inputs[1]: size -- 1-D int64 tensor whose values give the full output
                 shape (4 values: N, C, H, W).

    The kernel uses hardcoded half_pixel coordinate mapping and
    a=-0.75 Catmull-Rom cubic filter.  No configurable attributes.

    Args:
        op: The resize-bicubic operation.
        inputs: Two buffers -- input data and size.

    Returns:
        List containing a single output buffer with shape given by ``size``
        and the same dtype as ``input``.
    """
    assert isinstance(inputs[0], Buffer)  # input
    assert isinstance(inputs[1], Buffer)  # size (int64 output-shape vector)

    input_buffer = inputs[0]
    size_buffer = inputs[1]

    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    out_shape = size_buffer.to_numpy().astype(int).flatten().tolist()

    assert rank == 4, (
        f"resize_bicubic: input must be rank 4 (NCHW), got rank {rank}"
    )
    assert len(out_shape) == 4, (
        f"resize_bicubic: size must have 4 elements, got {len(out_shape)}"
    )

    target_device = _get_target_device(op)
    output = Buffer(shape=out_shape, dtype=input_buffer.dtype, device=CPU())
    ops.resize_ops.ResizeBicubic(
        output,
        input_buffer,
        (in_shape, out_shape),
        target_device._device_context_ptr(),
    )
    return [output]


# Distributed operations


@register_op_handler(mo.DistributedAllreduceSumOp)
def _handle_distributed_allreduce_sum(
    op: mo.DistributedAllreduceSumOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.allreduce.sum by summing tensors across devices.

    Operands (flat): N input tensors, N signal buffers, 1 input chain.
    Results: N output tensors (one per device, all holding the sum), 1 output chain.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.  Each input tensor is transferred to the CPU,
    summed via NumPy, and the result is placed back on each output device.

    Args:
        op: The allreduce sum operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers (one per device) followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"allreduce input {i} is not a Buffer"
        bufs.append(b)

    # Sum all inputs on the CPU via NumPy.
    total = bufs[0].to(CPU()).to_numpy().copy()
    for buf in bufs[1:]:
        total += buf.to(CPU()).to_numpy()

    # Place the sum on each output device.
    results = list(op.results)
    output_buffers: list[Buffer | None] = []
    for result in results[:-1]:
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(Buffer.from_numpy(total).to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers


@register_op_handler(mo.DistributedAllgatherOp)
def _handle_distributed_allgather(
    op: mo.DistributedAllgatherOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.allgather by copying each input to every device.

    Operands (flat): N input tensors, N signal buffers, 1 input chain.
    Results: N*N output tensors, 1 output chain.

    The MO-level allgather produces N*N raw outputs: for each device d
    and each input i, ``results[d*N + i]`` is a copy of ``input[i]`` on
    ``device[d]``.  The Graph API wraps these with separate ``ConcatOp``
    calls to produce the final N gathered tensors.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.

    Args:
        op: The allgather operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N*N output buffers followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"allgather input {i} is not a Buffer"
        bufs.append(b)

    results = list(op.results)
    output_buffers: list[Buffer | None] = []
    for idx, result in enumerate(results[:-1]):
        input_idx = idx % num_inputs
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(bufs[input_idx].to(device))

    output_buffers.append(None)
    return output_buffers


@register_op_handler(mo.DistributedScatterOp)
def _handle_distributed_scatter(
    op: mo.DistributedScatterOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.scatter by distributing root inputs to devices.

    Operands (flat): N input tensors (all on root), N signal buffers, 1 input chain.
    Results: N output tensors (one per device), 1 output chain.

    Each input[i] is copied to the device indicated by result[i]'s type.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.

    Args:
        op: The scatter operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"scatter input {i} is not a Buffer"
        bufs.append(b)

    # All inputs should reside on the root device.
    if bufs:
        root_device = bufs[0].device
        for i, buf in enumerate(bufs[1:], 1):
            assert buf.device == root_device, (
                f"scatter expects all inputs on root device {root_device}, "
                f"but input {i} is on {buf.device}"
            )

    results = list(op.results)
    num_outputs = len(results) - 1  # exclude trailing chain
    assert num_outputs == num_inputs, (
        f"scatter expects N inputs and N outputs, "
        f"got {num_inputs} inputs and {num_outputs} outputs"
    )

    output_buffers: list[Buffer | None] = []
    for idx, result in enumerate(results[:-1]):
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(bufs[idx].to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers


@register_op_handler(mo.DistributedBroadcastOp)
def _handle_distributed_broadcast(
    op: mo.DistributedBroadcastOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.broadcast by replicating input to all devices.

    Operands (flat): 1 input tensor, N signal buffers, 1 input chain.
    Results: N output tensors (one per device, all copies of the input),
    1 output chain.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.

    Args:
        op: The broadcast operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers followed by None for the chain.
    """
    # op.root identifies the source device, but in the flat operand layout
    # inputs[0] is always the root tensor; the interpreter simply copies it
    # to every output device regardless of root index.
    input_buf = inputs[0]
    assert isinstance(input_buf, Buffer), "broadcast input is not a Buffer"

    num_signal_bufs = len(op.signal_buffers)
    results = list(op.results)
    num_outputs = len(results) - 1  # exclude trailing chain
    assert num_outputs == num_signal_bufs, (
        f"broadcast expects one output per signal buffer, "
        f"got {num_outputs} outputs and {num_signal_bufs} signal buffers"
    )

    output_buffers: list[Buffer | None] = []
    for result in results[:-1]:
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(input_buf.to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers


# Non-maximum suppression


@register_op_handler(mo.NonMaximumSuppressionOp)
def _handle_non_maximum_suppression(
    op: mo.NonMaximumSuppressionOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.non_maximum_suppression via single-pass Mojo NMS kernel.

    Operands (MO_HostOnly):
      inputs[0]: boxes -- [batch, num_boxes, 4] float
      inputs[1]: scores -- [batch, num_classes, num_boxes] float
      inputs[2]: max_output_boxes_per_class -- scalar int64
      inputs[3]: iou_threshold -- scalar float
      inputs[4]: score_threshold -- scalar float

    The output shape [num_selected, 3] is data-dependent.  We allocate an
    upper-bound buffer (batch * classes * max_output_per_class), run NMS
    once, then truncate to the actual result size.

    Args:
        op: The non-maximum suppression operation.
        inputs: Five buffers as described above.

    Returns:
        List containing a single [num_selected, 3] int64 output buffer
        where each row is [batch_index, class_index, box_index].
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)  # boxes
    assert isinstance(inputs[1], Buffer)  # scores
    assert isinstance(inputs[2], Buffer)  # max_output_boxes_per_class
    assert isinstance(inputs[3], Buffer)  # iou_threshold
    assert isinstance(inputs[4], Buffer)  # score_threshold

    boxes_buffer = inputs[0]
    scores_buffer = inputs[1]
    max_output_boxes = int(inputs[2].to_numpy().item())
    iou_threshold = float(inputs[3].to_numpy().item())
    score_threshold = float(inputs[4].to_numpy().item())

    boxes_shape = list(boxes_buffer.shape)
    scores_shape = list(scores_buffer.shape)

    batch_size = boxes_shape[0]
    num_boxes = boxes_shape[1]
    num_classes = scores_shape[1]

    upper_bound = batch_size * num_classes * max_output_boxes
    if upper_bound == 0:
        return [Buffer(shape=[0, 3], dtype=DType.int64, device=CPU())]

    ctx_ptr = target_device._device_context_ptr()

    params = (
        batch_size,
        num_classes,
        num_boxes,
        max_output_boxes,
        iou_threshold,
        score_threshold,
    )

    # Single pass: run NMS into an upper-bound buffer.
    count_buf = Buffer(shape=[1], dtype=DType.int64, device=CPU())
    work_buf = Buffer(shape=[upper_bound, 3], dtype=DType.int64, device=CPU())
    ops.nms_ops.NmsRun(
        count_buf,
        work_buf,
        boxes_buffer,
        scores_buffer,
        params,
        ctx_ptr,
    )
    num_selected = int(count_buf.to_numpy().item())

    if num_selected == 0:
        return [Buffer(shape=[0, 3], dtype=DType.int64, device=CPU())]

    # Truncate upper-bound buffer to actual result size.
    return [Buffer.from_numpy(work_buf.to_numpy()[:num_selected].copy())]


@register_op_handler(mo.DistributedReducescatterSumOp)
def _handle_distributed_reducescatter_sum(
    op: mo.DistributedReducescatterSumOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.reducescatter.sum by summing then splitting.

    Operands (flat): N input tensors, N signal buffers, 1 input chain.
    Results: N output tensors (one per device, each a chunk of the sum),
    1 output chain.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.  Each input tensor is transferred to the CPU,
    summed via NumPy, and the result is split along the scatter axis so
    that each device receives its disjoint chunk.

    Args:
        op: The reduce-scatter sum operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"reducescatter input {i} is not a Buffer"
        bufs.append(b)

    axis = op.axis

    # Sum all inputs on the CPU via NumPy.
    total = bufs[0].to(CPU()).to_numpy().copy()
    for buf in bufs[1:]:
        total += buf.to(CPU()).to_numpy()

    # Split the summed result along the scatter axis using ragged binning
    # (same formula as ops/reducescatter.py).
    dim = total.shape[axis]
    chunk_sizes = [
        (dim + (num_inputs - i - 1)) // num_inputs for i in range(num_inputs)
    ]
    chunks = np.split(total, np.cumsum(chunk_sizes[:-1]), axis=axis)

    results = list(op.results)
    num_outputs = len(results) - 1  # exclude trailing chain
    assert num_outputs == num_inputs, (
        f"reducescatter expects N inputs and N outputs, "
        f"got {num_inputs} inputs and {num_outputs} outputs"
    )

    output_buffers: list[Buffer | None] = []
    for idx, result in enumerate(results[:-1]):
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(Buffer.from_numpy(chunks[idx]).to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers
