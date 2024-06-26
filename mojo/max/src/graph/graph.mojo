# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

from collections import Set
from sys.info import has_neon

import _mlir
from _mlir.builtin_attributes import StringAttr, TypeAttr
from _mlir.builtin_types import FunctionType

from ._attributes import _tensor_attr, _vector_attr
from .error import error
from .symbol import Symbol
from .type import Dim, ListType, TensorType, Type

from max.graph.quantization import QuantizationEncoding
from max.tensor import Tensor, TensorShape


# TODO: Add examples throughout.


struct _OwnedGraph(Movable):
    var ctx: _mlir.Context
    var op: _mlir.Operation
    var layers: List[String]
    var parameters: Set[String]

    fn __init__(
        inout self,
        owned ctx: _mlir.Context,
        owned op: _mlir.Operation,
        in_types: List[Type],
    ):
        self.ctx = ctx
        self.op = op
        self.layers = List[String]()
        self.parameters = Set[String]()
        for type in in_types:
            for dim in type[].dims():
                if not dim[].is_symbolic():
                    continue
                var name = str(dim[])
                self.parameters.add(name)

    fn current_layer(self) -> String:
        var layer: String = ""
        for layer_name in self.layers:
            if layer:
                layer += "."
            layer += layer_name[]
        return layer^

    fn module(self) -> _mlir.Module:
        try:
            return _mlir.Module.from_op(self.op.parent())
        except:
            return abort[_mlir.Module]("invalid MLIR state for graph")

    fn __moveinit__(inout self, owned existing: Self):
        self.ctx = existing.ctx
        self.op = existing.op
        self.layers = existing.layers^
        self.parameters = existing.parameters^

    fn __del__(owned self):
        self.module().as_op().destroy()
        self.ctx.__exit__()


alias _GraphRef = Arc[_OwnedGraph]


@value
struct _GraphLayerContext:
    var graph: _GraphRef
    var name: String

    fn __enter__(inout self):
        self.graph[].layers.append(self.name)

    fn __exit__(inout self, _error: Error) -> Bool:
        self.__exit__()
        return False

    fn __exit__(inout self):
        var name = self.graph[].layers.pop()
        debug_assert(name == self.name, "non-hiercharchical graph layers")


@value
struct Graph(CollectionElement, Stringable):
    """Represents a single MAX graph.

    A `Graph` is a callable routine in [MAX Engine](/max/engine), similar to a
    function in Mojo. Like functions, graphs have a name and signature. Unlike
    a function, which follows an imperative programming model, a `Graph`
    follows a [dataflow](https://en.wikipedia.org/wiki/Dataflow_programming)
    programming model, using lazily-executed, parallel operations instead of
    sequential instructions.

    When you instantiate a graph, you must specify the input shapes
    as one or more [`TensorType`](/max/api/mojo/graph/type/TensorType) or
    [`ListType`](/max/api/mojo/graph/type/ListType) values. Then, build a
    sequence of ops and set the graph output with [`output()`](#output). For
    example:

    ```mojo
    from max.graph import Type, Graph, TensorType, ops
    from max.tensor import Tensor, TensorShape

    def build_model() -> Graph:
        var graph = Graph(TensorType(DType.float32, 2, 6))

        var matmul_constant_value = Tensor[DType.float32](TensorShape(6, 1), 0.15)
        var matmul_constant = graph.constant(matmul_constant_value)

        var matmul = graph[0] @ matmul_constant
        var relu = ops.elementwise.relu(matmul)
        var softmax = ops.softmax(relu)
        graph.output(softmax)

        return graph
    ```

    You can't call a `Graph` directly from Mojo. You must compile it and
    execute it with MAX Engine. For more detail, see the tutorial about how to
    [build a graph with MAX Graph](/max/graph/get-started).
    """

    var _graph: _GraphRef

    fn __init__(inout self, in_type: Type):
        """Constructs a new `Graph` with a single input type.

        Although a `Graph` is technically valid once constructed, it is not
        usable for inference until you specify outputs by calling `output()`.
        Check the graph validity by calling `verify()`.

        Args:
            in_type: The graph's input type, as a single
                [`TensorType`](/max/api/mojo/graph/type/TensorType) or
                [`ListType`](/max/api/mojo/graph/type/ListType) value.
        """
        self.__init__("graph", List[Type](in_type))

    fn __init__(
        inout self,
        in_types: List[Type],
        out_types: List[Type] = List[Type](),
    ):
        """Constructs a new `Graph` using the default graph name.

        Although a `Graph` is technically valid once constructed, it is not
        usable for inference until you specify outputs by calling `output()`.
        Check the graph validity by calling `verify()`.

        Args:
            in_types: The graph's input types, as one or more
                [`TensorType`](/max/api/mojo/graph/type/TensorType) or
                [`ListType`](/max/api/mojo/graph/type/ListType) values.
            out_types: The graph's output types, as one or more
                [`TensorType`](/max/api/mojo/graph/type/TensorType) or
                [`ListType`](/max/api/mojo/graph/type/ListType) values.
                Deprecated. This will be inferred by the `output` call.
        """
        self.__init__("graph", in_types, out_types)

    fn __init__(
        inout self,
        name: String,
        in_types: List[Type],
        out_types: List[Type] = List[Type](),
    ):
        """Constructs a new `Graph` with a custom graph name.

        Although a `Graph` is technically valid once constructed, it is not
        usable for inference until you specify outputs by calling `output()`.
        Check the graph validity by calling `verify()`.

        Args:
            name: A name for the graph.
            in_types: The graph's input types, as one or more
                [`TensorType`](/max/api/mojo/graph/type/TensorType) or
                [`ListType`](/max/api/mojo/graph/type/ListType) values.
            out_types: The graph's output types, as one or more
                [`TensorType`](/max/api/mojo/graph/type/TensorType) or
                [`ListType`](/max/api/mojo/graph/type/ListType) values.
                Deprecated.  This will be inferred by the `output` call.
        """
        var ctx = _mlir.Context()
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        var module = _mlir.Module(_mlir.Location.unknown(ctx))
        var loc = _mlir.Location.unknown(ctx)

        var in_types_mlir = List[_mlir.Type]()
        for type in in_types:
            in_types_mlir.append(type[].to_mlir(ctx))
        var out_types_mlir = List[_mlir.Type]()
        for type in out_types:
            out_types_mlir.append(type[].to_mlir(ctx))

        var op = _c.graph_new(
            module,
            loc,
            name,
            _mlir.builtin_types.FunctionType(
                ctx, in_types_mlir, out_types_mlir
            ),
        )

        self._graph = Arc(_OwnedGraph(ctx, op, in_types))

    fn __str__(self) -> String:
        """Returns a `String` representation of this `Graph`.

        The representation uses a MLIR textual format. The format is subject to
        change and should only be used for debugging pruposes.

        Returns:
            A human-readable string representation of the graph.
        """
        return str(self._module())

    @always_inline
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
        with self._context().diagnostic_error():
            if not self._graph[].op.verify():
                raise error(self, "graph did not verify")

    fn layer(inout self, name: String) -> _GraphLayerContext:
        """Creates a context manager for a graph layer.

        Graph layers don't have a functional meaning for graph execution.
        They help provide debug and visualization information, tagging
        nodes in the graph with informal information about the structure
        of the overall graph.

        Args:
            name: The name of the layer.

        Returns:
            A context manager. Inside this context, the layer will be "active".
        """
        return _GraphLayerContext(self._graph, name)

    fn current_layer(self) -> String:
        """Returns the full path of the current layer.

        This is a `.`-separated string of each nested layer context created
        by `Graph.layer()`.

        Returns:
            The full path of the current layer.
        """
        return self._graph[].current_layer()

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn _body(self) raises -> _mlir.Block:
        return self._graph[].op.region(0).first_block()

    fn _module(self) -> _mlir.Module:
        """Returns the `Graph`'s parent `Module`s.

        Returns:
            The `Module` that holds this `Graph`.
        """
        return self._graph[].module()

    fn _context(self) raises -> _mlir.Context:
        """Returns the `Graph`'s MLIR context."""
        return self._module().context()

    def _new_parameters(self, dims: List[Dim]) -> Optional[_mlir.Attribute]:
        """Create an `outputParamDecls` for all newly introduced parameters."""
        ctx = self._context()
        new_params = List[_mlir.Attribute]()
        for dim in dims:
            if dim[].is_symbolic():
                name = str(dim[])
                if name not in self._graph[].parameters:
                    self._graph[].parameters.add(name)
                    new_params.append(_c.attr_new_dim_param_decl(ctx, name))

        if new_params:
            return _c.attr_new_param_decl_array(ctx, new_params)

        return None

    fn __getitem__(self, n: Int) raises -> Symbol:
        """Returns the n'th argument of this `Graph`.

        By argument, we mean the graph input. For example, `graph[0]` gets the
        first input and `graph[1]` gets the second input (as specified with
        the `Graph` constructor's `in_types`).

        This provides the argument as a `Symbol`, which you can use as input to
        other nodes in the graph.

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
            raise error(
                self,
                "index out of bounds: "
                + str(n)
                + ", graph has "
                + str(num_args)
                + " arguments",
            )
        return Symbol(self._graph, self._body().argument(n))

    # ===------------------------------------------------------------------=== #
    # nvop - the most generic op builder
    # ===------------------------------------------------------------------=== #

    fn nvop(
        self,
        name: String,
        inputs: List[Symbol] = List[Symbol](),
        out_types: List[Type] = List[Type](),
        attrs: List[_mlir.NamedAttribute] = List[_mlir.NamedAttribute](),
        enable_result_type_inference: Bool = False,
    ) raises -> List[Symbol]:
        """Adds a new node to the `Graph`.

        The node represents a single MAX Graph operation.

        This is a very low level API meant to enable creating any supported op.
        In general, it's less ergonomic compared to the higher level helpers in
        the `ops` package.

        Note that these nodes don't take concrete values as inputs, but rather
        symbolic values representing the outputs of other nodes or the
        `Graph`s arguments.

        Args:
            name: The name of the operation to use.
            inputs: The list of symbolic operands.
            out_types: The list of output types.
            attrs: Any attributes that the operation might require.
            enable_result_type_inference: Will infer the result type if True.

        Returns:
            The symbolic outputs of the newly-added node.
        """
        # TODO: Add input verification.
        var ctx = self._graph[].ctx

        var operands = List[_mlir.Value]()
        for i in range(len(inputs)):
            operands.append(inputs[i].handle)

        var out_types_mlir = List[_mlir.Type]()
        for type in out_types:
            out_types_mlir.append(type[].to_mlir(ctx))

        var op = _mlir.Operation(
            name=name,
            location=_mlir.Location.unknown(ctx),
            operands=operands,
            results=out_types_mlir,
            attributes=attrs,
            enable_result_type_inference=enable_result_type_inference,
        )

        var all_dims = List[Dim]()
        for out_type in out_types:
            all_dims += out_type[].dims()

        var out_param_attr = self._new_parameters(all_dims)
        if out_param_attr:
            op.set_inherent_attr("outputParamDecls", out_param_attr.take())

        var layer = self._graph[].current_layer()
        if layer:
            op.set_discardable_attr("layer", StringAttr(ctx, layer^))

        var output_op = self._body().terminator()
        if output_op:
            self._body().insert_before(output_op.value(), op)
        else:
            self._body().append(op)

        var results = List[Symbol]()
        for i in range(op.num_results()):
            results.append(Symbol(self._graph, op.result(i)))
        return results

    # ===------------------------------------------------------------------=== #
    # op - shorthands for single-result ops
    # ===------------------------------------------------------------------=== #

    fn op(
        self,
        name: String,
        out_type: Type,
        attrs: List[_mlir.NamedAttribute] = List[_mlir.NamedAttribute](),
    ) raises -> Symbol:
        """Adds a new single-output, nullary node to the `Graph`.

        See `Graph.nvop` for details. This overload can be used for operations
        that take no inputs and return a single result, such as `mo.constant`.

        Args:
            name: The name of the operation to use.
            out_type: The output types.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic output of the newly-added node.
        """
        return self.nvop(name, out_types=out_type, attrs=attrs)[0]

    fn op(
        self,
        name: String,
        inputs: List[Symbol],
        out_type: Type,
        attrs: List[_mlir.NamedAttribute] = List[_mlir.NamedAttribute](),
    ) raises -> Symbol:
        """Adds a new single-output node to the `Graph`.

        See `Graph.nvop` for details. This overload can be used for operations
        that return a single result.

        Args:
            name: The name of the operation to use.
            inputs: The list of symbolic operands.
            out_type: The output types.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic output of the newly-added node.
        """
        return self.nvop(name, inputs, out_type, attrs)[0]

    # ===------------------------------------------------------------------=== #
    # Factories for various nullary ops
    # ===------------------------------------------------------------------=== #

    fn constant[
        dtype: DType
    ](self, owned value: Tensor[dtype]) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type `TensorType` with the same
        shape and dtype as `value`.

        Parameters:
            dtype: The constant tensor's element type.

        Args:
            value: The constant's value.

        Returns:
            The symbolic output of this node.
        """
        return self.op(
            "mo.constant",
            TensorType(value.spec()),
            List(_tensor_attr(self._context(), "value", value)),
        )

    def quantize[
        encoding: QuantizationEncoding
    ](self, owned value: Tensor[DType.float32]) -> Symbol:
        """Quantizes a tensor using a specific quantization encoding.

        This takes the full-precision `value` as owned data and frees it.
        The resulting quantized constant is allocated and owns its data.

        Parameters:
            encoding: Describes a specific quantization encoding such as Q4_0.

        Args:
            value: Full-precision value to quantize.

        Returns:
            Symbol representing the quantized constant.
        """
        return self.constant(encoding.quantize(value^))

    fn vector[dtype: DType](self, values: List[Scalar[dtype]]) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type `TensorType` with 1-D shape,
        consistent with the size of `values`.

        Parameters:
            dtype: The constant tensor's element type.

        Args:
            values: A vector represneting the constant's value.

        Returns:
            The symbolic output of this node.
        """
        return self.op(
            "mo.constant",
            TensorType(dtype, len(values)),
            List(_vector_attr[dtype](self._context(), "value", values)),
        )

    fn scalar[
        dtype: DType
    ](self, value: Scalar[dtype], rank: Int = 0) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type scalar `TensorType`
        (0-D shape), when `rank` is 0, or a higher-rank `TensorType` of a single
        element.

        Parameters:
            dtype: The constant tensor's element type.

        Args:
            value: The constant's value.
            rank: The output tensor's rank.

        Returns:
            The symbolic output of this node.
        """
        var shape = List[Int](capacity=rank)
        for _ in range(rank):
            shape.append(1)
        return self.constant[dtype](Tensor(shape, value))

    fn scalar(self, value: Int, dtype: DType) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type `TensorType` of the same
        element type as `dtype`, and scalar (0-D) shape.

        Args:
            value: The scalar value.
            dtype: The constant's element type.

        Returns:
            The symbolic output of this node.

        Raises:
            If `value` cannot be instantiated as a tensor of element `dtype`.
        """
        if dtype is DType.uint8:
            return self.scalar(UInt8(value))
        if dtype is DType.uint16:
            return self.scalar(UInt16(value))
        if dtype is DType.uint32:
            return self.scalar(UInt32(value))
        if dtype is DType.uint64:
            return self.scalar(UInt64(value))

        if dtype is DType.int8:
            return self.scalar(Int8(value))
        if dtype is DType.int16:
            return self.scalar(Int16(value))
        if dtype is DType.int32:
            return self.scalar(Int32(value))
        if dtype is DType.int64:
            return self.scalar(Int64(value))

        # TODO(KERN-228): support BF16 on neon systems.
        @parameter
        if not has_neon():
            if dtype is DType.bfloat16:
                return self.scalar(BFloat16(value))

        # TODO(#33932): Enable once KGENCompilerRT provides __truncdfhf2.
        # if dtype is DType.float16:
        #     return self.scalar(Float16(value))

        if dtype is DType.float32:
            return self.scalar(Float32(value))
        if dtype is DType.float64:
            return self.scalar(Float64(value))

        raise error(self, "unimplemented Int conversion dtype: " + str(dtype))

    fn scalar(self, value: Float64, dtype: DType) raises -> Symbol:
        """Adds a node representing a `mo.constant` operation.

        The value of this constant will have the type `TensorType` of the same
        element type as `dtype`, and scalar (0-D) shape.

        Args:
            value: The scalar value.
            dtype: The constant's element type.

        Returns:
            The symbolic output of this node.

        Raises:
            If `value` cannot be instantiated as a tensor of element `dtype`.
        """

        # TODO(KERN-228): support BF16 on neon systems.
        @parameter
        if not has_neon():
            if dtype is DType.bfloat16:
                return self.scalar(BFloat16(value))

        # TODO(#33932): Enable once KGENCompilerRT provides __truncdfhf2.
        # if dtype is DType.float16:
        #     return self.scalar(Float16(value))

        if dtype is DType.float32:
            return self.scalar(Float32(value))
        if dtype is DType.float64:
            return self.scalar(Float64(value))

        raise error(
            self, "unimplemented FloatLiteral conversion dtype: " + str(dtype)
        )

    fn range[
        dtype: DType
    ](
        self, start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]
    ) raises -> Symbol:
        """Adds a node representing a `mo.range` operation.

        Parameters:
            dtype: The output tensor's element type.

        Args:
            start: The starting value.
            stop: The end value (exclusive).
            step: The step value.

        Returns:
            The symbolic output of this node.
        """
        return self.op(
            "rmo.mo.range",
            List[Symbol](
                self.scalar[dtype](start),
                self.scalar[dtype](stop),
                self.scalar[dtype](step),
            ),
            TensorType(dtype, len(range(start, stop, step))),
        )

    fn full[
        dtype: DType
    ](self, value: Scalar[dtype], dims: List[Symbol]) raises -> Symbol:
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
                raise error(self, "zeros inputs must be scalars")
            shape.append(dims[i])
            out_dims.append(Dim.dynamic())
        return self.op(
            "rmo.mo.broadcast_to",
            List[Symbol](self.scalar(value), ops.stack(shape)),
            TensorType(dtype, out_dims),
        )

    fn output(inout self, outputs: List[Symbol]) raises:
        """Adds an output for the graph.

        This is a special node that all graphs must have in order to deliver
        inference results. The `outs` symbol given here must match the shape
        and type of the `out_types` given when constructing the graph.

        Args:
            outputs: The return values, usually the result from one or more ops.
        """
        var ctx = self._context()
        var results = List[_mlir.Type]()
        for output in outputs:
            results.append(output[].type().to_mlir(ctx))
        var op = self._graph[].op

        var function_type = FunctionType.from_mlir(
            TypeAttr.from_mlir(op.get_inherent_attr("functionType")).type
        )
        function_type.results = results^
        var signature = _mlir.Type.parse(
            ctx, "!kgen.signature<" + str(function_type.to_mlir()) + ">"
        )
        op.set_inherent_attr(
            "functionType", TypeAttr(function_type.to_mlir()).to_mlir()
        )
        op.set_inherent_attr("signature", TypeAttr(signature).to_mlir())

        _ = self.nvop("mo.output", outputs)
