# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

from collections import Optional, Set
from os import abort
from pathlib import Path
from sys.info import has_neon

import _mlir
from _mlir.builtin_attributes import StringAttr, TypeAttr
from _mlir.builtin_types import FunctionType
from _mlir.ir import Module, Operation
from builtin._location import __call_location, _SourceLocation
from max.graph.quantization import QuantizationEncoding
from max.tensor import Tensor, TensorShape
from memory import ArcPointer

from utils.write import _WriteBufferStack

from ._attributes import _tensor_attr, _vector_attr
from .error import error
from .symbol import Symbol
from .type import Dim, ListType, TensorType, Type

# TODO: Add examples throughout.


@value
struct LayerInfo:
    """Name and Location information for a layer."""

    var name: String
    """The name of the layer."""

    var op_count: Int
    """The current number of ops in this layer.

    Note: a sublayer is considered a single op.
    """


struct _OwnedGraph(Movable):
    var ctx: _mlir.Context
    var op: _mlir.Operation
    var layers: List[LayerInfo]
    var parameters: Set[String]

    fn __init__(
        out self,
        owned ctx: _mlir.Context,
        owned op: _mlir.Operation,
        in_types: List[Type],
    ):
        self.ctx = ctx
        self.op = op
        self.layers = List[LayerInfo](LayerInfo("", 0))
        self.parameters = Set[String]()
        for type in in_types:
            for dim in type[].dims():
                if not dim[].is_symbolic():
                    continue
                var name = String(dim[])
                self.parameters.add(name)

    fn current_layer(self) -> String:
        var layer: String = ""
        for layer_info in self.layers:
            if layer:
                layer += "."
            layer += layer_info[].name
        return layer^

    fn current_location(self) -> _mlir.Location:
        var full_name = self.current_layer()
        var current_layer = self.layers[-1]

        var layer_op_num = 0
        if len(self.layers) >= 2:
            layer_op_num = self.layers[-2].op_count

        return _mlir.Location(
            self.ctx,
            full_name,
            layer_op_num,
            current_layer.op_count,
        )

    fn inc_op_count(mut self):
        self.layers[-1].op_count += 1

    @no_inline
    fn module(self) -> _mlir.Module:
        try:
            return _mlir.Module.from_op(self.op.parent())
        except:
            return abort[_mlir.Module]("invalid MLIR state for graph")

    fn __moveinit__(out self, owned existing: Self):
        self.ctx = existing.ctx
        self.op = existing.op
        self.layers = existing.layers^
        self.parameters = existing.parameters^

    @no_inline
    fn __del__(owned self):
        self.module().as_op().destroy()
        self.ctx.__exit__()


alias _GraphRef = ArcPointer[_OwnedGraph]


@value
struct _GraphLayerContext:
    var graph: _GraphRef
    var name: String

    fn __enter__(mut self):
        self.graph[].layers.append(LayerInfo(self.name, 0))

    fn __exit__(mut self):
        var name = self.graph[].layers.pop().name
        self.graph[].inc_op_count()
        debug_assert(name == self.name, "non-hiercharchical graph layers")


@value
@deprecated(
    "the Mojo max.engine API has been deprecated in favor of the Python API. It"
    " will be open sourced in a future patch prior to being removed."
)
struct Graph(CollectionElement, Stringable, Writable):
    """Represents a single MAX graph.

    A `Graph` is a callable routine in MAX Engine, similar to a
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

    def main():
        var graph = Graph(TensorType(DType.float32, 2, 6))

        var matmul_constant_value = Tensor[DType.float32](TensorShape(6, 1), 0.15)
        var matmul_constant = graph.constant(matmul_constant_value)

        var matmul = graph[0] @ matmul_constant
        var relu = ops.elementwise.relu(matmul)
        var softmax = ops.softmax(relu)
        graph.output(softmax)

        print(graph)
    ```

    You can't call a `Graph` directly from Mojo. You must compile it and
    execute it with MAX Engine. For more detail, see the tutorial about how to
    [build a graph with MAX Graph](/max/tutorials/get-started-with-max-graph).
    """

    var _graph: _GraphRef

    @implicit
    fn __init__(out self, in_type: Type):
        """Constructs a new `Graph` with a single input type.

        Although a `Graph` is technically valid once constructed, it is not
        usable for inference until you specify outputs by calling `output()`.
        Check the graph validity by calling `verify()`.

        Args:
            in_type: The graph's input type, as a single
                [`TensorType`](/max/api/mojo/graph/type/TensorType) or
                [`ListType`](/max/api/mojo/graph/type/ListType) value.
        """
        self = Self("graph", List[Type](in_type))

    fn __init__(
        out self,
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
        self = Self("graph", in_types, out_types)

    fn __init__(
        out self,
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

        self._graph = ArcPointer(_OwnedGraph(ctx, op, in_types))

    fn __init__(out self, path: Path) raises:
        """Constructs a new `Graph` from a MLIR file.

        Experimental. Recreates a graph from an MLIR file.

        Args:
            path: The path of the MLIR file.
        """
        with open(path, "r") as f:
            var model_str = f.read()
            var ctx = _mlir.Context()
            ctx.load_modular_dialects()
            # Do we need to load all available dialects?
            ctx.load_all_available_dialects()
            var module = Module.parse(ctx, model_str)
            var first_op = module.body().first_operation()
            var in_list = List[Type]()
            var first_block = first_op.region(0).first_block()
            for i in range(first_block.num_arguments()):
                var arg_type = Type.from_mlir(first_block.argument(i).type())
                in_list.append(arg_type)
            self._graph = _OwnedGraph(ctx, first_op, in_list)

    fn debug_str(self, pretty_print: Bool = False) -> String:
        return self._module().debug_str(pretty_print)

    fn __str__(self) -> String:
        """Returns a `String` representation of this `Graph`.

        The representation uses a MLIR textual format. The format is subject to
        change and should only be used for debugging pruposes.

        Returns:
            A human-readable string representation of the graph.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self._module())

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

    fn layer(mut self, name: String) -> _GraphLayerContext:
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

    fn _name(self) raises -> String:
        """Returns the `Graph`'s name."""
        # Can use String() here because 'name' is a string attribute.
        return String(self._graph[].op.get_inherent_attr("name"))

    def _new_parameters(self, dims: List[Dim]) -> Optional[_mlir.Attribute]:
        """Create an `outputParamDecls` for all newly introduced parameters."""
        ctx = self._context()
        new_params = List[_mlir.Attribute]()
        for dim in dims:
            if dim[].is_symbolic():
                name = String(dim[])
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
                "index out of bounds: ",
                n,
                ", graph has ",
                num_args,
                " arguments",
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
            location=self._graph[].current_location(),
            operands=operands,
            results=out_types_mlir,
            attributes=attrs,
            enable_result_type_inference=enable_result_type_inference,
        )
        self._graph[].inc_op_count()

        var output_op = self._body().terminator()
        if output_op:
            self._body().insert_before(output_op.value(), op)
        else:
            self._body().append(op)

        var results = List[Symbol]()
        for i in range(op.num_results()):
            results.append(Symbol(self._graph, op.result(i)))

        var all_dims = List[Dim]()
        for res in results:
            all_dims += res[].type().dims()

        var out_param_attr = self._new_parameters(all_dims)
        if out_param_attr:
            op.set_inherent_attr("outputParamDecls", out_param_attr.take())

        # Now perform verification of the new operation
        with self._context().diagnostic_error():
            if not op.verify():
                raise error(self, "operation did not verify")

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
        return self.nvop(name, out_types=List(out_type), attrs=attrs)[0]

    fn op(
        self,
        name: String,
        input: Symbol,
        out_type: Type,
        attrs: List[_mlir.NamedAttribute] = List[_mlir.NamedAttribute](),
    ) raises -> Symbol:
        """Adds a new single-input, single-output node to the `Graph`.

        See `Graph.nvop` for details. This overload can be used for operations
        that take a single input, and return a single result.

        Args:
            name: The name of the operation to use.
            input: The symbolic operand.
            out_type: The output type.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic output of the newly-added node.
        """
        return self.nvop(name, List(input), List(out_type), attrs)[0]

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
        return self.nvop(name, inputs, List(out_type), attrs)[0]

    fn op(
        self,
        name: String,
        inputs: List[Symbol],
        attrs: List[_mlir.NamedAttribute] = List[_mlir.NamedAttribute](),
    ) raises -> Symbol:
        """Adds a new single-output node to the `Graph` with result type inference.

        See `Graph.nvop` for details. This overload can be used for operations
        that return a single result.

        Args:
            name: The name of the operation to use.
            inputs: The list of symbolic operands.
            attrs: Any attributes that the operation might require.

        Returns:
            The symbolic output of the newly-added node.
        """
        return self.nvop(
            name, inputs, attrs=attrs, enable_result_type_inference=True
        )[0]

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

        To quantize your model weights, follow these steps:

        1. Import your trained model weights
        2. Choose your [quantization
        encodings](/max/api/mojo/graph/quantization/encodings)
        3. Apply quantization using [`Graph.quantize()`](/max/api/mojo/graph/graph/Graph#quantize)

        ```mojo
        quantized_constant = graph.quantize[Q4_0Encoding](constant_value)
        ```
        4. Use [`qmatmul()`](/max/api/mojo/graph/ops/quantized_ops/qmatmul) for matrix
        operations with quantized weights

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
        var shape = List[Int, hint_trivial_type=True](capacity=rank)
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

        raise error(self, "unimplemented Int conversion dtype: ", dtype)

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
                return self.scalar(value.cast[DType.bfloat16]())

        # TODO(#33932): Enable once KGENCompilerRT provides __truncdfhf2.
        # if dtype is DType.float16:
        #     return self.scalar(Float16(value))

        if dtype is DType.float32:
            return self.scalar(value.cast[DType.float32]())
        if dtype is DType.float64:
            return self.scalar(value)

        raise error(
            self, "unimplemented FloatLiteral conversion dtype: ", dtype
        )

    fn range[
        dtype: DType
    ](
        self, start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]
    ) raises -> Symbol:
        """Creates a sequence of numbers. The sequence goes from `start` with
        increments of size `step` up to (but not including) `stop`. All arguments
        are mandatory and must have the same element type.

        Note the following restrictions on input values:
        1. `step` must be non-zero
        2. `stop - start` must be zero or have the same sign as `step`

        Parameters:
            dtype: The output tensor's element type.

        Args:
            start: The start of the range to generate.
            stop: The range will be generated up to, but not including, this value.
            step: The step size for the range.

        Returns:
            A symbolic tensor value containing the defined range of values.
        """
        return self.op(
            "rmo.mo.range",
            List[Symbol](
                self.scalar[dtype](start),
                self.scalar[dtype](stop),
                self.scalar[dtype](step),
            ),
            TensorType(dtype, len(range(Int(start), Int(stop), Int(step)))),
        )

    fn range(
        self, start: Symbol, stop: Symbol, step: Symbol, out_dim: Dim
    ) raises -> Symbol:
        """Creates a sequence of numbers. The sequence goes from `start` with
        increments of size `step` up to (but not including) `stop`. All arguments
        are mandatory and must have the same element type.

        Note the following restrictions on input values:
        1. `step` must be non-zero
        2. `stop - start` must be zero or have the same sign as `step`

        Args:
            start: The start of the range to generate.
            stop: The range will be generated up to, but not including, this value.
            step: The step size for the range.
            out_dim: The expected output dimensions returned by the range op.
              These will be assert at graph execution time to be correct.

        Returns:
            A symbolic tensor value containing the defined range of values.
        """
        return self.op(
            "rmo.mo.range",
            List[Symbol](
                start,
                stop,
                step,
            ),
            TensorType(start.tensor_type().dtype, out_dim),
        )

    @always_inline
    fn full[
        dtype: DType
    ](self, value: Scalar[dtype], *dims: Dim) raises -> Symbol:
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
        var shape = List[Dim]()
        for d in dims:
            shape.append(d[])

        return self.full(value, shape, __call_location())

    @always_inline
    fn full[
        dtype: DType
    ](
        self,
        value: Scalar[dtype],
        dims: List[Dim],
        location: Optional[_SourceLocation] = None,
    ) raises -> Symbol:
        """Creates a constant-valued symbolic tensor of a specified shape.

        Parameters:
            dtype: The output tensor's element type.

        Args:
            value: The value to fill the resulting tensor with.
            dims: The shape dimensions of the zero-valued tensor.
            location: An optional location for a more specific error message.

        Returns:
            A symbolic tensor of the specified shape and dtype, where
            every value is the specified fill value.
        """
        return self.scalar(value).broadcast_to(
            dims, location or __call_location()
        )

    fn output(mut self, output: Symbol) raises:
        """Adds an output for the graph.

        This is a special node that all graphs must have in order to deliver
        inference results. The `output` symbol given here must match the shape
        and type of the `out_types` given when constructing the graph.

        Args:
            output: The return value, usually the result from an op.
        """
        return self.output(List(output))

    fn output(mut self, outputs: List[Symbol]) raises:
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
            ctx, String("!kgen.generator<", function_type.to_mlir(), ">")
        )
        op.set_inherent_attr(
            "functionType", TypeAttr(function_type.to_mlir()).to_mlir()
        )
        op.set_inherent_attr("signature", TypeAttr(signature).to_mlir())

        # Set the result_names metadata on the staged op, which is needed by
        # the engine for execution.
        var result_names = String()
        var buffer = _WriteBufferStack(result_names)
        buffer.write("[")
        for i in range(len(outputs)):
            buffer.write('"output', i, '"')
            if i < (len(outputs) - 1):
                buffer.write(", ")
        buffer.write("]")
        buffer.flush()

        op.set_discardable_attr(
            "result_names", _mlir.Attribute.parse(ctx, result_names)
        )

        _ = self.nvop("mo.output", outputs)
