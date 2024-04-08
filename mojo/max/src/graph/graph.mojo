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
from sys.info import has_neon
from tensor import Tensor

import _mlir

from .attr import AttrMap
from .module import _Module
from .symbol import Symbol, SymbolTuple
from .type import MOList, MOTensor, TypeTuple


# TODO: Add examples throughout.


@value
struct Graph(CollectionElement, Stringable):
    """Represents a single MAX graph."""

    var _op: _mlir.Operation

    fn __init__(inout self, in_types: TypeTuple, out_types: TypeTuple):
        """Constructs a new `Graph`.

        The constructed Graph will not be valid unless it has no outputs;
        a graph with outputs will need a `graph.output` call to tell it
        what to return. The graph's validity can be checked by calling
        `graph.verify()`.

        Args:
            in_types: The `Graph`'s input types.
            out_types: The `Graph`'s output types.
        """
        self.__init__("graph", in_types, out_types)

    fn __init__(
        inout self, name: String, in_types: TypeTuple, out_types: TypeTuple
    ):
        """Constructs a new `Graph`.

        The constructed Graph will not be valid unless it has no outputs;
        a graph with outputs will need a `graph.output` call to tell it
        what to return. The graph's validity can be checked by calling
        `graph.verify()`.

        Args:
            name: A name for the graph.
            in_types: The `Graph`'s input types.
            out_types: The `Graph`'s output types.
        """
        var module = _Module()
        self = module.graph(name, in_types, out_types)

    fn __str__(self) -> String:
        """Returns a `String` representation of this `Graph`.

        The representation uses a MLIR textual format. The format is subject to
        change and should only be used for debugging pruposes.

        Returns:
            A human-readable string representation of the graph.
        """
        try:
            return str(self._module()._module)
        except:
            return str(self._op)

    fn verify(self) raises:
        """Verifies the `Graph` and its contents.

        Examples of cases when a `Graph` may not be valid (the list is not
        exhaustive):
        1. it has an `output` op whose types don't match its `out_types`
        2. it has an op with an invalid name, number, type of operands,
            output types, etc.
        3. it contains cycles

        Raises:
            If the `Graph` did not pass verification. In this case it will also
            print a diagnostic message indicating the error.
        """
        if not self._op.verify():
            raise "graph did not verify"

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn _body(self) raises -> _mlir.Block:
        return self._op.region(0).first_block()

    fn _module(self) raises -> _Module:
        """Returns the `Graph`'s parent `Module`s.

        Returns:
            The `Module` that holds this `Graph`.
        """
        return _Module(_mlir.Module.from_op(self._op.parent()))

    fn __getitem__(self, n: Int) raises -> Symbol:
        """Returns the `n`th argument of this `Graph`.

        The `Symbol` that the argument is returned as can be used as input to
        other `Graph` `Node`s.

        Args:
            n: The argument number. First argument is at position 0.

        Returns:
            A `Symbol` representing the argumen't symbolic value, as seen from
            within the `Graph`'s body.

        Raises:
            If `n` is not a valid argument number.
        """
        # TODO: Add an exmple.
        var num_args = self._body().num_arguments()
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

        The `Node` represents a single MAX Graph operation.

        This is a very low level API meant to enable creating any supported op.
        In general, it's less ergonomic compared to the higher level helpers in
        the `ops` module.

        Note that these `Node`s don't take concrete values as inputs, but rather
        symbolic values representing the outputs of other `Node`s or the
        `Graph`s arguments.

        Args:
            name: The name of the operation to use.
            inputs: The list of symbolic operands.
            out_types: The list of output types.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic outputs of the newly-added `Node`.
        """
        # TODO: Add input verification.
        var ctx = self._op.context()

        var operands = List[_mlir.Value]()
        for i in range(len(inputs)):
            operands.append(inputs[i].handle)

        var op = _mlir.Operation(
            name=name,
            location=_mlir.Location.unknown(ctx),
            operands=operands,
            results=out_types.to_mlir(ctx),
            attributes=attrs.attrs,
        )

        var output_op = self._body().terminator()
        if output_op:
            self._body().insert_before(output_op.value(), op)
        else:
            self._body().append(op)

        var results = List[Symbol]()
        for i in range(op.num_results()):
            results.append(op.result(i))
        return results

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
        dtype: DType
    ](self, owned value: Tensor[dtype]) raises -> Symbol:
        """Adds a `Node` representing a `mo.constant` operation.

        The value of this constant will have the type `MOTensor` with the same
        shape and dtype as `value`.

        Parameters:
            dtype: The constant tensor's element type.

        Args:
            value: The constant's value.

        Returns:
            The symbolic output of this `Node`.
        """
        return self.op(
            "mo.constant",
            MOTensor(value.spec()),
            AttrMap(self._module().tensor_attr("value", value)),
        )

    fn vector[dtype: DType](self, values: List[Scalar[dtype]]) raises -> Symbol:
        """Adds a `Node` representing a `mo.constant` operation.

        The value of this constant will have the type `MOTensor` with 1-D shape,
        consistent with the size of `values`.

        Parameters:
            dtype: The constant tensor's element type.

        Args:
            values: A vector represneting the constant's value.

        Returns:
            The symbolic output of this `Node`.
        """
        return self.op(
            "mo.constant",
            MOTensor(dtype, len(values)),
            AttrMap(self._module().vector_attr[dtype]("value", values)),
        )

    fn scalar[
        dtype: DType
    ](self, value: Scalar[dtype], rank: Int = 0) raises -> Symbol:
        """Adds a `Node` representing a `mo.constant` operation.

        The value of this constant will have the type scalar `MOTensor`
        (0-D shape), when `rank` is 0, or a higher-rank `MOTensor` of a single
        element.

        Parameters:
            dtype: The constant tensor's element type.

        Args:
            value: The constant's value.
            rank: The output tensor's rank.

        Returns:
            The symbolic output of this `Node`.
        """
        var shape = List[Int](capacity=rank)
        for i in range(rank):
            shape.append(1)
        return self.constant[dtype](Tensor(shape, value))

    fn scalar(self, value: Int, dtype: ElementType) raises -> Symbol:
        """Adds a `Node` representing a `mo.constant` operation.

        The value of this constant will have the type `MOTensor` of the same
        element type as `dtype`, and scalar (0-D) shape.

        Args:
            value: The scalar value.
            dtype: The constant's element type.

        Returns:
            The symbolic output of this `Node`.

        Raises:
            If `value` cannot be instantiated as a tensor of element `dtype`.
        """
        if dtype.dtype == DType.uint8:
            return self.scalar(UInt8(value))
        if dtype.dtype == DType.uint16:
            return self.scalar(UInt16(value))
        if dtype.dtype == DType.uint32:
            return self.scalar(UInt32(value))
        if dtype.dtype == DType.uint64:
            return self.scalar(UInt64(value))

        if dtype.dtype == DType.int8:
            return self.scalar(Int8(value))
        if dtype.dtype == DType.int16:
            return self.scalar(Int16(value))
        if dtype.dtype == DType.int32:
            return self.scalar(Int32(value))
        if dtype.dtype == DType.int64:
            return self.scalar(Int64(value))

        # TODO(#30525): Enable once LLVM bfloat16 emulation support matures.
        @parameter
        if not has_neon():
            if dtype.dtype == DType.bfloat16:
                return self.scalar(BFloat16(value))

        # TODO(#33932): Enable once KGENCompilerRT provides __truncdfhf2.
        # if dtype.dtype == DType.float16:
        #     return self.scalar(Float16(value))

        if dtype.dtype == DType.float32:
            return self.scalar(Float32(value))
        if dtype.dtype == DType.float64:
            return self.scalar(Float64(value))

        raise "unimplemented Int conversion dtype: " + str(dtype.dtype)

    fn scalar(self, value: Float64, dtype: ElementType) raises -> Symbol:
        """Adds a `Node` representing a `mo.constant` operation.

        The value of this constant will have the type `MOTensor` of the same
        element type as `dtype`, and scalar (0-D) shape.

        Args:
            value: The scalar value.
            dtype: The constant's element type.

        Returns:
            The symbolic output of this `Node`.

        Raises:
            If `value` cannot be instantiated as a tensor of element `dtype`.
        """

        # TODO(#30525): Enable once LLVM bfloat16 emulation support matures.
        @parameter
        if not has_neon():
            if dtype.dtype == DType.bfloat16:
                return self.scalar(BFloat16(value))

        # TODO(#33932): Enable once KGENCompilerRT provides __truncdfhf2.
        # if dtype.dtype == DType.float16:
        #     return self.scalar(Float16(value))

        if dtype.dtype == DType.float32:
            return self.scalar(Float32(value))
        if dtype.dtype == DType.float64:
            return self.scalar(Float64(value))

        raise "unimplemented FloatLiteral conversion dtype: " + str(dtype.dtype)

    fn range[
        dtype: DType
    ](
        self, start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]
    ) raises -> Symbol:
        """Adds a `Node` representing a `mo.range` operation.

        Parameters:
            dtype: The output tensor's element type.

        Args:
            start: The starting value.
            stop: The end value (exclusive).
            step: The step value.

        Returns:
            The symbolic output of this `Node`.
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

    fn full[
        dtype: DType
    ](self, value: Scalar[dtype], dims: SymbolTuple) raises -> Symbol:
        """Creates a constant-valued symbolic tensor of a specified shape.

        Parameters:
            dtype: The output tensor's element type.

        Args:
            value: The value to fill the resulting tensor with.
            dims: The shape dimensions of the zero-valued tensor.

        Returns:
            A symbolic tensor of the specified shape and dtype, where
            every value is the specified fill value.
        """
        var out_dims = List[Dim]()
        var shape = List[Symbol]()
        for i in range(len(dims)):
            if dims[i].tensor_type().rank() != 0:
                raise "zeros inputs must be scalars"
            shape.append(dims[i])
            out_dims.append(Dim.dynamic())
        return self.op(
            "mo.broadcast_to",
            (self.scalar(value), ops.stack(shape)),
            MOTensor(dtype, out_dims),
        )

    fn output(self, outs: SymbolTuple) raises:
        """Adds a `Node` representing a `mo.output` operation.

        This is a special `Node` that all `Graph`s must have. The inputs must
        match the `Graph`s signature (specifically, its return values).

        Args:
            outs: The `Graph`s return values.
        """
        _ = self.nvop("mo.output", outs)
