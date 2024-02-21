# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives.

`Graph`s are callable routines in
[MAX Engine](/engine/), similar to functions in Mojo.
Like functions, graphs have a name and signature. Unlike functions, which
follow an imperative programming model, `Graph`s follow a
[dataflow](https://en.wikipedia.org/wiki/Dataflow_programming) programming
model, using lazily-executed, parallel operations instead of sequential
instructions. `Graph`s aren't called directly from Mojo, but are instead
compiled and executed by MAX Engine, for example using the MAX Engine API.
"""

from collections import Optional

import _mlir

from .attr import AttrMap
from .module import Module
from .symbol import Symbol, SymbolTuple
from .type import MOList, MOType, MOTensor, TypeTuple


# TODO: Add examples throughout.


@value
struct Graph(CollectionElement, Stringable):
    """Represents a single MAX graph."""

    var _op: _mlir.Operation

    fn __init__(
        inout self, name: String, in_types: TypeTuple, out_types: TypeTuple
    ):
        let module = Module()
        self = module.graph(name, in_types, out_types)

    fn __str__(self) -> String:
        return str(self._op)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn _body(self) raises -> _mlir.Block:
        return self._op.region(0).first_block()

    fn module(self) raises -> Module:
        """Returns the `Module` containing this `Graph`."""
        return Module(_mlir.Module.from_op(self._op.parent()))

    fn __getitem__(self, n: Int) raises -> Symbol:
        """Returns the `n`th argument of this `Graph`.

        The `Symbol` that the argument is returned as can be used as input to
        other `Graph` nodes.

        Args:
            n: The argument number. First argument is at position 0.

        Returns:
            A `Symbol` representing the argumen't symbolic value, as seen from
            within the `Graph`'s body.

        Raises:
            If `n` is not a valid argument number.
        """
        # TODO: Add an exmple.
        let num_args = self._body().num_arguments()
        if (n >= num_args) or (n < 0):
            raise "index out of bounds: " + str(
                n
            ) + ", graph has " + num_args + " arguments"
        return Symbol(self._body().argument(n))

    # ===------------------------------------------------------------------=== #
    # nvop - the most generic op builder
    # ===------------------------------------------------------------------=== #

    fn nvop(
        self,
        name: String,
        inputs: SymbolTuple = SymbolTuple(),
        out_types: TypeTuple = TypeTuple(),
        attrs: AttrMap = AttrMap(),
    ) raises -> SymbolTuple:
        """Adds a new `Node` to the `Graph`.

        The node represents a single MAX Graph operation. For a list of
        supported operations, see the
        [MO dialect reference](/engine/reference/mlir/mo).

        This is a very low level API meant to enable creating any supported op.
        In general, it's less ergonomic compared to the higher level helpers in
        the `ops` module.

        Note that these nodes don't take concrete values as inputs, but rather
        symbolic values representing the outputs of other nodes or the `Graph`s
        arguments.

        Args:
            name: The name of the operation to use.
            inputs: The list of symbolic operands.
            out_types: The list of output types.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic outputs of the newly-added `Node`.
        """
        # TODO: Add input verification.
        let ctx = self._op.context()

        let op = _mlir.Operation(
            name=name,
            location=_mlir.Location.unknown(ctx),
            operands=inputs.as_values(),
            results=out_types.to_mlir(self.module()),
            attributes=attrs.attrs,
        )

        let output_op = self._body().terminator()
        if output_op:
            self._body().insert_before(output_op.value(), op)
        else:
            self._body().append(op)

        var tup = SymbolTuple()
        for i in range(op.num_results()):
            tup.append(op.result(i))
        return tup

    # ===------------------------------------------------------------------=== #
    # op - shorthands for single-result ops
    # ===------------------------------------------------------------------=== #

    fn op(
        self, name: String, out_type: AnyMOType, attrs: AttrMap = AttrMap()
    ) raises -> Symbol:
        """Adds a new single-output, nullary `Node` to the `Graph`.

        See `Graph.nvop` for details. This overload can be used for operations
        that take no inputs and return a single result, such as `mo.constant`.

        Args:
            name: The name of the operation to use.
            out_type: The output types.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic output of the newly-added `Node`.
        """
        return self.nvop(name, out_types=out_type, attrs=attrs)[0]

    fn op(
        self,
        name: String,
        inputs: SymbolTuple,
        out_type: AnyMOType,
        attrs: AttrMap = AttrMap(),
    ) raises -> Symbol:
        """Adds a new single-output `Node` to the `Graph`.

        See `Graph.nvop` for details. This overload can be used for operations
        that return a single result.

        Args:
            name: The name of the operation to use.
            inputs: The list of symbolic operands.
            out_type: The output types.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic output of the newly-added `Node`.
        """
        return self.nvop(name, inputs, out_type, attrs)[0]

    # ===------------------------------------------------------------------=== #
    # Factories for various nullary ops
    # ===------------------------------------------------------------------=== #

    fn constant[
        dtype: DType = DType.float32
    ](self, owned value: Tensor[dtype]) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type `MOTensor` with the same
        shape and dtype as `value`.

        Params:
            dtype: The constant tensor's element type.

        Args:
            value: The constant's value.

        Returns:
            The symbolic output of this node.
        """
        return self.op(
            "mo.constant",
            MOTensor(value.spec()),
            AttrMap(self.module().tensor_attr("value", value)),
        )

    fn vector[
        dtype: DType = DType.float32
    ](self, values: DynamicVector[Scalar[dtype]]) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type `MOTensor` with 1D shape,
        consistent with the size of `values`.

        Params:
            dtype: The constant tensor's element type.

        Args:
            values: A vector represneting the constant's value.

        Returns:
            The symbolic output of this node.
        """
        return self.op(
            "mo.constant",
            MOTensor(dtype, len(values)),
            AttrMap(self.module().vector_attr[dtype]("value", values)),
        )

    fn scalar[
        dtype: DType = DType.float32
    ](self, value: Scalar[dtype], rank: Int = 0) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type scalar `MOTensor`
        (0D shape), when `rank` is 0, or a higher-rank `MOTensor` of a single
        element.

        Params:
            dtype: The constant tensor's element type.

        Args:
            value: The constant's value.
            rank: The output tensor's rank.

        Returns:
            The symbolic output of this node.
        """
        var shape = DynamicVector[Int](capacity=rank)
        for i in range(rank):
            shape.append(1)
        return self.constant[dtype](Tensor(shape, value))

    fn range[
        dtype: DType = DType.int64
    ](
        self, start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]
    ) raises -> Symbol:
        """Adds a node representing a `mo.range` operation.

        Params:
            dtype: The output tensor's element type.

        Args:
            start: The starting value.
            stop: The end value (exclusive).
            step: The step value.

        Returns:
            The symbolic output of this node.
        """
        return self.op(
            "mo.range",
            (
                self.scalar[dtype](start),
                self.scalar[dtype](stop),
                self.scalar[dtype](step),
            ),
            MOTensor(dtype, len(range(start, stop, step))),
        )

    fn output(self, outs: SymbolTuple) raises:
        """Adds a node representing a `mo.output` operation.

        This is a special node that all `Graph`s must have. The inputs must
        match the `Graph`s signature (specifically, its return values).

        Args:
            outs: The `Graph`s return values.
        """
        _ = self.nvop("mo.output", outs)
