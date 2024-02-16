# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

from collections import Optional

from .attr import AttrMap
from .module import Module
from .symbol import Symbol, SymbolTuple
from .type import MOList, MOType, MOTensor, TypeTuple


@value
struct Graph:
    """The core unit of computation in MAX Engine.

    `Graph`s are callable routines in MAX Engine, similar to functions in
    Mojo. Like functions, graphs have a name and signature. Unlike functions,
    which follow an imperative programming model, `Graph`s follow a dataflow
    programming model, using lazily-executed, parallel operations instead of
    sequential instructions. `Graph`s aren't called directly from Mojo, but are
    instead compiled and executed by MAX Engine, for example using the MAX
    Engine API.
    """

    # TODO: Add an exmple, after we cleaned up the TypeTuple thing.
    # TODO: Refer to the concepts doc for the meaning of Symbol, etc.
    # TODO: Link to max engine page, wikipedia, etc.

    var g: mlir.Operation
    """A handle to the `Graph`'s internal implementation."""

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn _body(self) raises -> mlir.Block:
        return self.g.region(0).first_block()

    fn module(self) raises -> Module:
        """Returns the `Module` containing this `Graph`.

        Returns:
            The `Module` containing this `Graph`.

        Raises:
            If the `Graph` is not owned by a `Module`. This can't happen
            in normal operation.
        """
        return Module(mlir.Module.from_op(self.g.parent()))

    fn __getitem__(self, n: Int) raises -> Symbol:
        """Returns the `n`th argument of this `Graph`.

        This allows ops inside the `Graph` to take this argument as input.

        Args:
            n: The argument's position.

        Returns:
            A `Symbol` representing the argumen't symbolic value, as seen from
            within the `Graph`'s body.

        Raises:
            If `n` is outside the bounds 0 <= `n` < #arguments.
        """
        # TODO: Add an exmple, after we cleaned up the Arity thing.
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
        name: StringRef,
        inputs: SymbolTuple = SymbolTuple(),
        out_types: TypeTuple = TypeTuple(),
        attrs: AttrMap = AttrMap(),
    ) raises -> SymbolTuple:
        """Constructs and adds an operation (op) node to the graph.

        Args:
            name: The name of the op. This must be a valid `mo` operation
                name with the dialect prefix, for instance `"mo.add"`.
            inputs: The inputs (operands) to the op. Inputs are symbolic graph
                values (`Symbol`s), and represent data during graph execution.
            out_types: The type of each output to the op. The number of outputs
                must match the length of out_types.
            attrs: Attributes to add to the op. Attributes are function
                parameters known at graph construction time, and can't
                depend on dynamic values.

        Returns:
            A `SymbolTuple` of symbolic graph values (`Symbol`s) representing
            the outputs of the graph operation. The length of the tuple is
            the number of outputs of the op. It has the same length as
            `out_types`, and satisfies `result[i].type() == out_types[i]`
            for each `0 <= i < len(out_types)`.

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        # We should also have this raise with a meaningful error message
        # based on the diagnostic run by `.verify` on the op construction.
        let ctx = self.g.context()

        let op = mlir.Operation(
            name=name,
            location=mlir.Location.unknown(ctx),
            operands=inputs.as_values(),
            results=out_types.to_mlir(self.module()),
            attributes=attrs.attrs,
        )

        let output_op = self._output_op()
        if output_op:
            self._body().insert_before(output_op.value(), op)
        else:
            self._body().append(op)

        var tup = SymbolTuple()
        for i in range(op.num_results()):
            tup.append(op.result(i))
        return tup

    fn _output_op(self) raises -> Optional[mlir.Operation]:
        # Terminator must always be an output op by graph definition
        return self._body().terminator()

    # ===------------------------------------------------------------------=== #
    # op - shorthands for single-result ops
    # ===------------------------------------------------------------------=== #

    fn op(
        self,
        name: StringRef,
        out_type: AnyMOType,
        attrs: AttrMap = AttrMap(),
    ) raises -> Symbol:
        """Constructs and adds a single-result operation (op) node to the graph.

        This overload is for ops with zero input operands (eg. `constant`).

        Args:
            name: The name of the op. This must be a valid `mo` operation
                name with the dialect prefix, for instance `"mo.add"`.
            out_type: The type of the operation's result. The returned
                `Symbol` will have `result.type() == out_type`.
            attrs: Attributes to add to the op. Attributes are function
                parameters known at graph construction time, and can't
                depend on dynamic values.

        Returns:
            A single symbolic graph value (`Symbol`) representing
            the output of the graph operation. It satisfies
            `result.type() == out_type`.

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        return self.nvop(name, out_types=out_type, attrs=attrs)[0]

    fn op(
        self,
        name: StringRef,
        inputs: SymbolTuple,
        out_type: AnyMOType,
        attrs: AttrMap = AttrMap(),
    ) raises -> Symbol:
        """Constructs and adds a single-result operation (op) node to the graph.

        Args:
            name: The name of the op. This must be a valid `mo` operation
                name with the dialect prefix, for instance `"mo.add"`.
            inputs: The inputs (operands) to the op. Inputs are symbolic graph
                values (`Symbol`s), and represent data during graph execution.
            out_type: The type of the operation's result. The returned
                `Symbol` will have `result.type() == out_type`.
            attrs: Attributes to add to the op. Attributes are function
                parameters known at graph construction time, and can't
                depend on dynamic values.

        Returns:
            A single symbolic graph value (`Symbol`) representing
            the output of the graph operation. It satisfies
            `result.type() == out_type`.

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        return self.nvop(name, inputs, out_type, attrs)[0]

    # ===------------------------------------------------------------------=== #
    # Factories for various nullary ops
    # ===------------------------------------------------------------------=== #

    fn constant[
        dtype: DType
    ](self, owned value: Tensor[dtype]) raises -> Symbol:
        """Creates a constant valued graph node from a data tensor.

        The input data will be stored into the graph as constant data
        available during graph execution.

        Args:
            value: A data tensor of data to store as a graph constant value.

        Returns:
            A symbolic graph value (`Symbol`) representing the data
            at graph execution type. The `Symbol` will be have its `type()`
            be a `MOTensor` with the same `dtype` and static shape equal
            to the input data's `spec`, or in other words:
            `symbol.tensor_type() == MOTensor(value.spec)`.

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        return self.op(
            "mo.constant",
            MOTensor(value.spec()),
            AttrMap(self.module().tensor_attr("value", value)),
        )

    fn vector[
        dtype: DType
    ](self, values: DynamicVector[Scalar[dtype]]) raises -> Symbol:
        """Creates a constant valued graph node from a data vector.

        The input data will be stored into the graph as constant data
        available during graph execution. It will be represented as a
        tensor of rank 1, with a static dimension equal to the length
        of the input vector.

        Args:
            values: A data vector of data to store as a graph constant value.

        Returns:
            A symbolic graph value (`Symbol`) representing the data
            at graph execution type. The `Symbol` will be have its `type()`
            be a `MOTensor` with the same `dtype` as the input data,
            rank 1, and a single static dimension equal to `len(values)`.
            In other words: `symbol.tensor_type() == MOTensor(dtype, len(values))`.

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        return self.op(
            "mo.constant",
            MOTensor(dtype, len(values)),
            AttrMap(self.module().vector_attr[dtype]("value", values)),
        )

    fn scalar[
        dtype: DType
    ](self, value: Scalar[dtype], rank: Int = 0) raises -> Symbol:
        """Creates a constant valued graph node from a scalar data value.

        The input data will be stored into the graph as constant data
        available during graph execution. It will be represented as a
        tensor of rank `rank`. The shape of the tensor will have `rank`
        dimensions of size 1 (0 dimensions for rank 0, the default).

        This is sufficient to represent larger tensors for common operators
        like `splat`, `fill`, `zeros` or `ones` through broadcasting.

        For instance:

        ```mojo
        # ones_like doesn't exist!
        # x + g.ones_like(x)
        from max.graph import *
        var g = Graph(...)
        g[0] + 1  # 1 is turned into a `scalar` constant, and then broadcast
               # to the same shape as x`
        ```

        Args:
            value: A single value to store as a graph constant value.
            rank: The rank of the tensor storing the single constant value.

        Returns:
            A symbolic graph value (`Symbol`) representing the data
            at graph execution type. The `Symbol` will be have its `type()`
            be a `MOTensor` with the same `dtype` as the input data,
            rank `rank`, and `rank` dimensions of size 1
            (0 dimensions for rank 0, the default).

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        var shape = DynamicVector[Int](capacity=rank)
        for i in range(rank):
            shape.append(1)
        return self.constant[dtype](Tensor(shape, value))

    fn range[
        dtype: DType
    ](
        self, start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]
    ) raises -> Symbol:
        """Creates a constant valued graph node representing a uniform range."""
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
        """Creates an `output` operation node for the `Graph`.

        This is a special op that all `Graph`s must have. The inputs
        of this op must have exactly the same length and type as the `out_types`
        provided when constructing the `Graph`.

        Args:
            outs: The values to return from graph execution. The types of these
                values must exactly match the `out_types` used to construct
                the graph.

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        _ = self.nvop("mo.output", outs)

    fn list(self, elements: SymbolTuple) raises -> Symbol:
        """Creates a symbolic list from a set of inputs.

        The values of this list must be symbolic graph values.
        The resulting list value may be used in graph list operations.
        See `Symbol` and `MOList` for more details.

        Args:
            elements: The initial elements of the list. These values currently
                must all be symoblic tensors.

        Raises:
            - On an empty input list. We can't infer the type of the list
                in this case, use the `Graph.list(MOTensor)` overload instead.
            - If any elements of the list are not tensor types. We currently
                only support non-nested lists of tensor values.
            - If the internal graph structure is invalid in certain ways.
                This can't happen during normal operation.
        """
        if not len(elements):
            raise "Can't create empty list without a type, use graph.list(MOTensor)"
        let tensor_type = elements[0].tensor_type()
        for i in range(len(elements)):
            if not elements[i].tensor_type() == tensor_type:
                raise "All list tensors must be the same type"
        return self.op(
            "mo.list.create", elements, MOList(elements[0].tensor_type())
        )

    fn list(self, tensor_type: MOTensor) raises -> Symbol:
        """Creates an empty symbolic list of a given tensor type.

        The resulting list value may be used in graph list operations.
        See `Symbol` and `MOList` for more details.

        Args:
            tensor_type: The type of elements which can be put into the
                list during graph execution. See `MOList` docs for more.

        Raises:
            If the internal graph structure is invalid in certain ways.
            This can't happen during normal operation.
        """
        return self.op("mo.list.create", (), MOList(tensor_type))
