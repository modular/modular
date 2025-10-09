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
"""Operations for invoking user-defined operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from max import mlir
from max._core import graph as _graph
from max._core.dialects import builtin
from max.dtype import DType
from max.mlir import BoolAttr, DictAttr, IndexType, IntegerAttr, StringAttr
from max.mlir.dialects import mo

from ...driver import Device
from ..graph import Graph
from ..type import DeviceRef, Type, _ChainType
from ..value import (
    BufferValue,
    TensorValue,
    Value,
    _is_strong_tensor_value_like,
    _OpaqueValue,
)


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
        dtype = _graph.dtype_to_type(param)
        attr = builtin.TypeAttr(dtype)
        return mlir.Attribute._CAPICreate(attr._CAPIPtr)  # type: ignore
    else:
        raise TypeError(
            f"unsupported parameter type {type(param)} for custom op"
        )


def custom(
    name: str,
    device: Device | DeviceRef,
    values: Sequence[Value[Any]],
    out_types: Sequence[Type[Any]],
    parameters: Mapping[str, bool | int | str | DType] | None = None,
) -> list[Value[Any]]:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with the
    [`@compiler.register`](/mojo/manual/decorators/compiler-register/)
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
    device = DeviceRef.from_device(device)

    if any(isinstance(val, BufferValue | _OpaqueValue) for val in values):
        raise TypeError(
            "custom ops that take buffers or opaque values to do in-place "
            "updates should use ops.inplace_custom instead"
        )

    values = [
        TensorValue(v) if _is_strong_tensor_value_like(v) else v for v in values
    ]

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

    custom_op.device = mlir.Attribute._CAPICreate(  # type: ignore
        device.to_mlir()._CAPIPtr
    )

    # Call the verifier, will throw if the call is invalid.
    graph._kernel_library.verify_custom_op(custom_op)

    return results


def inplace_custom(
    name: str,
    device: DeviceRef,
    values: Sequence[Value[Any]],
    out_types: Sequence[Type[Any]] | None = None,
    parameters: dict[str, bool | int | str | DType] | None = None,
) -> list[Value[Any]]:
    """Creates a node to execute an in-place custom graph operation in the graph.

    The custom op should be registered by annotating a function with the
    [`@compiler.register`](/mojo/manual/decorators/compiler-register/)
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
    has_buffer_operand = any(isinstance(val, BufferValue) for val in values)
    if not has_buffer_operand and not any(
        isinstance(val, _OpaqueValue) for val in values
    ):
        raise TypeError(
            "expected at least one BufferValue or _OpaqueValue as input to an "
            "in-place custom op"
        )

    # Pass empty out_types if unspecified.
    out_mlir_types = [t.to_mlir() for t in out_types] if out_types else []

    graph = Graph.current

    values = [
        TensorValue(v) if _is_strong_tensor_value_like(v) else v for v in values
    ]

    chain_operand = graph.device_chains[device]
    if has_buffer_operand:
        chain_operand = graph._merge_chains(
            (graph._current_chain, chain_operand)
        )

    (*results, out_chain), custom_op = graph._add_op_get_op_with_results(
        mo.custom,
        results_=[*out_mlir_types, _ChainType().to_mlir()],
        operands_=[*values, chain_operand],
        symbol=StringAttr.get(name, graph._context),
    )
    graph.device_chains[device] = out_chain
    if has_buffer_operand:
        graph._update_chain(out_chain)

    if parameters is not None:
        custom_op.parameters = DictAttr.get(
            {
                name: _parameter_attribute(param, graph._context)
                for name, param in parameters.items()
            },
            graph._context,
        )

    custom_op.device = mlir.Attribute._CAPICreate(  # type: ignore
        device.to_mlir()._CAPIPtr
    )

    # Call the verifier, will throw if the call is invalid.
    graph._kernel_library.verify_custom_op(custom_op)

    return results


# TODO(GEX-2471): Cleanup the API for mo.custom with non-default chains.
