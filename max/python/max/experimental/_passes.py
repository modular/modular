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
"""Transformation passes over a MAX Graph."""

import collections
from typing import TypeVar, overload

from max import _core as mlir
from max._core.dialects import builtin, kgen, mo, mosh
from max.driver import CPU
from max.dtype import DType
from max.graph import Graph, Shape, TensorType, Type, Value
from max.graph.dim import AlgebraicDim, Dim, StaticDim
from max.graph.graph import _location


def _graph_block(graph: Graph) -> mlir.Block:
    op = mlir.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(op, mo.GraphOp)
    return op.regions[0].front


def _fixup_graph(graph: Graph) -> None:
    """Updates the mo.GraphOp and Graph objects to match the internal graph
    structure.

    Assumes that the graph block definition is correct, and infers the remaining
    properties from there:
        - Graph.inputs
        - Graph._params
        - op.input_parameters
        - op.result_parameters
        - op.function_type
        - op.signature
        - op metadata: argument names
    """
    op = mlir.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(op, mo.GraphOp)
    block = op.regions[0].front

    with graph:
        # - use block.arguments as the source of truth for inputs
        inputs = [Value.from_mlir(arg) for arg in block.arguments]
        if isinstance(output_op := block[-1], mo.OutputOp):
            results = [Value.from_mlir(o) for o in output_op.operands]
        else:
            results = []

        # - reset graph.inputs
        graph.inputs = inputs

        # - reset op.input_parameters
        input_params = {
            dim: None
            for input in inputs
            for dim in getattr(input.type, "parameters", ())
        }
        si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
        op.input_parameters = kgen.ParamDeclArrayAttr(
            [kgen.ParamDeclAttr(str(dim), si64) for dim in input_params]
        )
        # - update graph._params
        graph._params.update(input_params)
        # - update argument names
        op.discardable_attributes["argument_names"] = builtin.ArrayAttr(
            [builtin.StringAttr(f"input{i}") for i in range(len(graph.inputs))]
        )

        result_params = {
            dim: None
            for result in results
            for dim in getattr(result.type, "parameters", ())
            if dim not in input_params
        }
        op.result_parameters = kgen.ParamDeclArrayAttr(
            [kgen.ParamDeclAttr(str(dim), si64) for dim in result_params]
        )

        # - reset op.function_type
        op.function_type = builtin.FunctionType(  # type: ignore
            [input.type.to_mlir() for input in inputs],
            [result.type.to_mlir() for result in results],
        )
        # - reset op.signature
        op.signature = kgen.FuncTypeGeneratorType([], op.function_type)  # type: ignore


def add_input(graph: Graph, type: Type) -> Value:
    """Adds a new input to an existing graph.

    Args:
        graph: The graph to which to add the new input
        type: The type of the new input to add
    Returns:
        The Value associated with the new input
    """
    block = _graph_block(graph)

    with graph:
        block.add_argument(type.to_mlir(), _location())

    _fixup_graph(graph)
    return graph.inputs[-1]


def remove_unused_arguments(graph: Graph) -> None:
    """Removes any unused arguments from the graph.

    Args:
        graph: The graph on which to apply the pass
    """
    block = _graph_block(graph)

    # reverse so indices don't during iteration+mutation
    for i, arg in reversed(list(enumerate(block.arguments))):
        if not arg.num_uses:
            block.erase_argument(i)

    _fixup_graph(graph)


T = TypeVar("T")


def remove_static_shape_info(graph: Graph) -> dict[int, Dim]:
    graph_op = mlir.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(graph_op, mo.GraphOp)
    block = graph_op.regions[0].front

    parameters: collections.defaultdict[int, Dim] = collections.defaultdict(
        lambda: Dim(f"D{len(parameters)}")
    )

    @overload
    def symbolic(v: builtin.IntegerAttr) -> kgen.ParamDeclRefAttr: ...
    @overload
    def symbolic(v: T) -> T: ...

    def symbolic(v):
        if isinstance(v, builtin.IntegerAttr):
            return symbolic(Dim.from_mlir(v)).to_mlir()
        if isinstance(v, kgen.ParamOperatorAttr):
            operands = [symbolic(o) for o in v.operands]
            assert v.type
            return kgen.ParamOperatorAttr(v.opcode, operands, v.type)
        if isinstance(v, StaticDim):
            return parameters[int(v)]
        if isinstance(v, AlgebraicDim):
            return AlgebraicDim(symbolic(v.attr))
        if isinstance(v, Shape):
            return Shape(symbolic(dim) for dim in v)
        if isinstance(v, Type):
            v_mlir = v.to_mlir()
            assert isinstance(v_mlir, mlir.Type)
            return Type.from_mlir(symbolic(v_mlir))
        if isinstance(v, (mo.TensorType, mo.BufferType)):
            return type(v)(
                symbolic(v.shape_attr), v.dtype, v.device_ref, v.metadata
            )
        if isinstance(v, mosh.ShapeAttr):
            return mosh.ShapeAttr([symbolic(d) for d in v.values], v.type)
        return v

    with graph:
        for arg in block.arguments:
            arg.type = symbolic(arg.type)

        for op in block:
            if isinstance(op, (mosh.ParamToValueOp, kgen.ParamDeclareOp)):
                op.value = symbolic(op.value)
                continue

            # ShapeOfOp verifies its result is a static shape
            # corresponding to a static rank input.
            if isinstance(op, (mo.ShapeOfOp, mo.ConstantOp)):
                continue

            # Only update op results, not operands. Operands will be
            # defined as results of other ops.
            for result in op.results:
                result.type = symbolic(result.type)

    add_input(graph, TensorType(DType.bool, [0, *parameters.values()], CPU()))
    return parameters
