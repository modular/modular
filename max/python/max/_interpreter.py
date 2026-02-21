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

"""MO graph interpreter for eager execution.

This module provides an interpreter for MO (Modular Operation) graphs that can
execute operations directly without going through the full compilation pipeline.
This is useful for eager mode execution where compilation latency needs to be
minimized.

The interpreter walks through the MO graph in topological order and dispatches
each operation to an appropriate handler. Handlers can implement operations
using NumPy or by calling into Mojo kernels.

Example usage:
    from max._interpreter import MOInterpreter

    interp = MOInterpreter()
    outputs = interp.execute(graph, input_buffers)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from max import _core
from max._core.dialects import builtin, kgen, mo, mosh
from max.driver import Buffer
from max.graph import Graph

# Import handlers to register them (side effect import)
from ._interpreter_ops import lookup_handler

# Type alias for interpreter slots
InterpreterSlots = dict[Any, Buffer | None]


class MOInterpreter:
    """Interprets MO graphs by dispatching to ops directly.

    This interpreter walks through MO graph operations in a valid execution
    order and dispatches each operation to an appropriate handler. The handlers
    execute the operations and produce output buffers.
    """

    def _validate_inputs(self, graph: Graph, inputs: Sequence[Buffer]) -> None:
        """Validate input buffers match graph expectations.

        Args:
            graph: The graph being executed.
            inputs: Input buffers provided by caller.

        Raises:
            ValueError: If inputs don't match graph expectations.
        """
        graph_inputs = list(graph.inputs)
        if len(graph_inputs) != len(inputs):
            raise ValueError(
                f"Expected {len(graph_inputs)} inputs, got {len(inputs)}"
            )
        # TODO(EMF-93): Add dtype/shape validation once we have more complete
        # tensor type extraction. The MO type system provides dtype and
        # shape_attr but extracting static shapes requires handling
        # symbolic dimensions.

    # Op names that require the compiled execution path. When any of these
    # are present in a graph the interpreter falls back to compilation
    # instead of raising NotImplementedError.  Name-based matching is used
    # (like _is_dispatchable and the handler name-fallback) because
    # nanobind may create different class objects for the same MLIR op.
    _COMPILATION_REQUIRED_OP_NAMES: tuple[str, ...] = ("CustomOp",)

    def can_execute(self, graph: Graph) -> bool:
        """Check whether the graph contains ops that require compilation.

        Pre-scans the graph for ops listed in
        ``_COMPILATION_REQUIRED_OP_NAMES``.  Returns False if any such op
        is found, indicating the caller should use the compiled execution
        path instead.

        Args:
            graph: The graph to check.

        Returns:
            True if the interpreter can handle the graph, False if it
            contains ops that require compilation.
        """
        module: builtin.ModuleOp = _core.Operation._from_cmlir(
            graph._module.operation
        )  # type: ignore[assignment]
        for op in self._walk_ops(module):
            if type(op).__name__ in self._COMPILATION_REQUIRED_OP_NAMES:
                return False
        return True

    def execute(
        self,
        graph: Graph,
        inputs: Sequence[Buffer],
    ) -> Sequence[Buffer | None]:
        """Execute an MO graph and return output buffers.

        Args:
            graph: The finalized MO graph to execute.
            inputs: Input buffers corresponding to graph.inputs.

        Returns:
            List of output buffers.

        Raises:
            ValueError: If inputs don't match graph expectations.
            RuntimeError: If output value was not computed.
            NotImplementedError: If an operation has no handler.
        """
        # Create a new interpreter slots dictionary for this execution.
        slots: InterpreterSlots = {}

        # Validate inputs before execution
        self._validate_inputs(graph, inputs)

        # Map graph inputs to their buffers
        for graph_input, buffer in zip(graph.inputs, inputs, strict=True):
            slots[graph_input._mlir_value] = buffer

        # Walk ops in the graph body and dispatch
        # use _core bindings for type-safe access
        module: builtin.ModuleOp = _core.Operation._from_cmlir(
            graph._module.operation
        )  # type: ignore[assignment]
        output_op = None
        for op in self._walk_ops(module):
            if isinstance(op, mo.OutputOp):
                # Store the output op but don't dispatch it
                output_op = op
            else:
                self._dispatch_op(op, slots)

        # Collect outputs from the mo.output terminator
        if output_op is None:
            raise RuntimeError("Graph has no output terminator")
        outputs = []
        # mo.OutputOp.operands returns Value directly (not OpOperand)
        for operand in output_op.operands:
            try:
                outputs.append(slots[operand])
            except RuntimeError as e:
                raise RuntimeError(
                    f"Output value not computed: {operand}"
                ) from e
        return outputs

    def _walk_ops(self, module: builtin.ModuleOp) -> Iterator[_core.Operation]:
        """Walk operations in a valid execution order.

        Args:
            module: The MLIR module operation.

        Returns:
            Generator of dispatchable operations in execution order.
        """

        # MO graphs have the structure:
        # builtin.module -> mo.graph -> Region 0 -> Block 0 -> operations
        # SSA form guarantees operations are already in valid execution order.
        for top_level_op in module.body:
            if isinstance(top_level_op, mo.GraphOp):
                block = top_level_op.regions[0].front
                for op in block:
                    if self._is_dispatchable(op):
                        yield op

    def _is_dispatchable(self, op: _core.Operation) -> bool:
        """Check if an operation should be dispatched or collected.

        Skip function definitions and other structural ops.
        OutputOp is included so we can extract outputs from it.

        Args:
            op: The operation to check.

        Returns:
            True if the operation should be processed, False otherwise.
        """
        skip_types = (
            mo.ChainCreateOp,  # Sequencing (interpreter executes sequentially)
            kgen.ParamDeclareOp,  # Shape parameter declarations
            mosh.ParamFromValueOp,  # Records values into params (not needed)
        )
        if isinstance(op, skip_types):
            return False

        # TODO(EMF-104): Check type for these
        skip_names = (
            "ParamConstantOp",  # Constant parameter declarations
        )

        return type(op).__name__ not in skip_names

    def _dispatch_op(
        self, op: _core.Operation, slots: dict[Any, Buffer | None]
    ) -> None:
        """Dispatch a single MO operation to its handler.

        Args:
            op: The operation to dispatch.
            slots: The interpreter slots.

        Raises:
            NotImplementedError: If no handler exists for the operation.
        """
        # Check handler registry
        if (handler := lookup_handler(op)) is not None:
            # Operation.operands returns OpOperand, use .value to get the Value.
            # Use .get() with default None for chain values (ChainCreateOp is
            # skipped, so chain values are not stored in slots)
            input_buffers = [
                slots.get(operand.value) for operand in op.operands
            ]
            outputs = handler(op, input_buffers)
        else:
            raise NotImplementedError(f"No handler for op: {type(op).__name__}")

        # Store outputs
        for result, output_buf in zip(op.results, outputs, strict=True):
            slots[result] = output_buf
