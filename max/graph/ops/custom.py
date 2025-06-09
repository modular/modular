# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Operations for invoking user-defined operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from max import mlir
from max._core import graph as _graph
from max._core.dialects import builtin
from max.dtype import DType
from max.mlir import BoolAttr, DictAttr, IndexType, IntegerAttr, StringAttr
from max.mlir.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, Type, _ChainType
from ..value import BufferValue, Value, _OpaqueValue


def _parameter_attribute(
    param: bool | int | str | DType, context: mlir.Context
) -> mlir.Attribute:
    """Converts a Python type to an MLIR attribute to parametrize a kernel."""
    if isinstance(param, bool):
        return BoolAttr.get(param, context)
    elif isinstance(param, int):
        return IntegerAttr.get(IndexType.get(context), param)
    elif isinstance(param, str):
        return StringAttr.get(param, context)
    elif isinstance(param, DType):
        # Wrap the MLIR type corresponding to dtype in a TypeAttr,
        # which MOToKGENLowering expects.
        dtype = _graph.dtype_type(param)
        attr = builtin.TypeAttr(dtype)
        return mlir.Attribute._CAPICreate(attr._CAPIPtr)  # type: ignore
    else:
        msg = f"unsupported parameter type {type(param)} for custom op"
        raise TypeError(msg)


def custom(
    name: str,
    device: DeviceRef,
    values: Sequence[Value],
    out_types: Sequence[Type],
    parameters: Mapping[str, bool | int | str | DType] | None = None,
) -> list[Value]:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with the
    [`@compiler.register`](/max/api/mojo-decorators/compiler-register/)
    decorator.

    Args:
        name: The op name provided to ``@compiler.register``.
        values: The op function's arguments.
        out_types: The list of op function's return type.
        parameters: Dictionary of extra parameters expected by the kernel.
        device: Device that the op is assigned to.
            This becomes a `target` parameter to the kernel.

    Returns:
        Symbolic values representing the outputs of the op in the graph.
        These correspond 1:1 with the types passed as ``out_types``.
    """
    graph = Graph.current
    symbol_attr = StringAttr.get(name, graph._context)

    if any(isinstance(val, (BufferValue, _OpaqueValue)) for val in values):
        msg = (
            "custom ops that take buffers or opaque values to do in-place "
            "updates should use ops.inplace_custom instead"
        )
        raise TypeError(msg)

    results, custom_op = graph._add_op_get_op_with_results(
        mo.custom, [t.to_mlir() for t in out_types], values, symbol=symbol_attr
    )

    if parameters is not None:
        custom_op.parameters = DictAttr.get(
            {
                name: _parameter_attribute(param, graph._context)
                for name, param in parameters.items()
            },
            graph._context,
        )

    custom_op.device = mlir.Attribute._CAPICreate(
        device.to_mlir()._CAPIPtr  # type: ignore
    )

    # Call the verifier, will throw if the call is invalid.
    graph._kernel_library.verify_custom_op(custom_op)

    return results


def inplace_custom(
    name: str,
    device: DeviceRef,
    values: Iterable[Value],
    out_types: Iterable[Type] | None = None,
    parameters: dict[str, bool | int | str | DType] | None = None,
) -> list[Value]:
    """Creates a node to execute an in-place custom graph operation in the graph.

    The custom op should be registered by annotating a function with the
    [`@compiler.register`](/max/api/mojo-decorators/compiler-register/)
    decorator.

    Args:
        name: The op name provided to ``@compiler.register``.
        device: Device that the op is assigned to.
            This becomes a `target` parameter to the kernel.
        values: The op function's arguments.
        parameters: Dictionary of extra parameters expected by the kernel.
    """
    # Unfortunately there's no existing way to mark a particular NDBuffer input
    # as needing to be backed by a `mo.buffer` value at the graph level.
    #
    # This will change as the new kernel API matures and has support added for
    # marking particular inputs as mutable.
    #
    # Until that switch is made check that at least one input to the custom op
    # is a BufferValue to provide some level of safety.
    if not any(isinstance(val, (BufferValue, _OpaqueValue)) for val in values):
        msg = (
            "expected at least one BufferValue or _OpaqueValue as input to an "
            "in-place custom op"
        )
        raise TypeError(msg)

    # Pass empty out_types if unspecified.
    out_mlir_types = [t.to_mlir() for t in out_types] if out_types else []

    graph = Graph.current
    current_chain = graph._current_chain

    (*results, out_chain), custom_op = graph._add_op_get_op_with_results(
        mo.custom,
        results_=[*out_mlir_types, _ChainType().to_mlir()],
        operands_=[*values, current_chain],
        symbol=StringAttr.get(name, graph._context),
    )
    graph._update_chain(out_chain)

    if parameters is not None:
        custom_op.parameters = DictAttr.get(
            {
                name: _parameter_attribute(param, graph._context)
                for name, param in parameters.items()
            },
            graph._context,
        )

    custom_op.device = mlir.Attribute._CAPICreate(
        device.to_mlir()._CAPIPtr  # type: ignore
    )

    # Call the verifier, will throw if the call is invalid.
    graph._kernel_library.verify_custom_op(custom_op)

    return results
