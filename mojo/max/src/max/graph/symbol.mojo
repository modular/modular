# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Symbolic value primitives."""

from collections.optional import Optional

import _mlir
from builtin._location import __call_location, _SourceLocation

from utils.variant import Variant

from ._attributes import _string_attr
from .error import error, format_error
from .graph import Graph, _GraphRef
from .ops import add, div, matmul, mul, pow, reshape, sub, transpose

# TODO: The overloads are incomplete, and make unverified assumptions about
# dtype, etc.


# TODO: Cull the sea of operator overloads. Should be simplifiable with traits.
#
# Some development notes - tried implementing a Reifiable trait, with
# Symbol implementing it, along with a few adapters, e.g. ReifiableInt, so that
# we can then implement operators like e.g.
#
#     fn __add__[T: Reifiable](self, rhs: T) raises -> Symbol:
#         var g = self.graph()
#         return add(self, rhs.to_symbol(g, self.tensor_type()))
#
# Alas that falls on its face for some reason. Bug?


@value
struct Symbol(Copyable, Movable, Stringable, Writable):
    """Represents a symbolic value within a `Graph`.

    A `Symbol` can represent the output of a node, the arguments of a `Graph`
    (as seen from within its body), and more generally any symbolic value
    available within the `Graph`. Other nodes receive `Symbol` values as inputs
    to form a computation graph.

    A `Symbol` may also refer to an existing input or output of a node, and you
    can change them, such as by swapping a new `Symbol`.

    Conceptually, a `Symbol` is the equivalent of a variable in Mojo. A `Symbol`
    can also be thought of as the end of an edge in the dataflow graph (the
    other end being one use of that symbol).

    Similar to a regular variable, a `Symbol` has a data type.

    Note: All the methods in this type are documented as "Creates foo". This is
    a shorthand notation for "Adds a node representing an op that returns foo".
    """

    var _graph: _GraphRef
    var handle: _mlir.Value
    """A handle to this `Symbol`'s internal representation."""

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn graph(self) -> Graph:
        """Returns the `Graph` owning this `Symbol`.

        Returns:
            The parent `Graph`.
        """
        return Graph(self._graph)

    fn type(self) raises -> Type:
        """Returns this `Symbol`'s type.

        Returns:
            The `Symbol`'s type.
        """
        return Type.from_mlir(self.handle.type())

    fn tensor_type(self) raises -> TensorType:
        """Returns this `Symbol`'s type, as `TensorType`.

        Implicitly asserts that the type is indeed `TensorType`, and raises an
        error otherwise.

        Returns:
            The tensor type of this `Symbol`.
        """
        return self.type().tensor()

    fn shape(self) raises -> List[Dim]:
        """Returns this `Symbol`'s tensor shape, as `List[Dim]`.

        Implicitly asserts that the type is indeed `TensorType`, and raises an
        error otherwise.

        Returns:
            The tensor shape of this `Symbol`.
        """
        return self.type().tensor().dims

    fn __str__(self) -> String:
        """Returns a `String` representation of this `Symbol`.

        The representation uses an internal MLIR Assembly format, and typically
        shows the node that outputs this `Symbol`. Its structure
        is subject to change without notice, and should not be used for
        serialization. For debugging purposes only.

        Returns:
            A textual representation of this `Symbol`.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this symbol to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write(String(self.handle))

    # ===------------------------------------------------------------------=== #
    # Casting and reshaping operators.
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn rebind(self, *dims: Dim) raises -> Symbol:
        return self._rebind_impl(dims, __call_location())

    fn _rebind_impl(
        self, dims: VariadicListMem[Dim, *_], call_loc: _SourceLocation
    ) raises -> Symbol:
        var out_dims = List[Dim]()
        for dim in dims:
            out_dims.append(dim[])

        var message = format_error(
            self.graph(),
            "Failed to rebind runtime shape",
            location=call_loc,
        )
        return ops.rebind(self, out_dims, message)

    @always_inline
    fn rebind(self, dims: List[Dim]) raises -> Symbol:
        return self._rebind_impl(dims, __call_location())

    fn _rebind_impl(
        self, dims: List[Dim], call_loc: _SourceLocation
    ) raises -> Symbol:
        var message = format_error(
            self.graph(),
            "failed to rebind runtime shape",
            call_loc,
        )
        return ops.rebind(self, dims, message)

    fn reshape(self) raises -> Symbol:
        return ops.reshape(
            self,
            self.graph().vector[DType.int64](List[Int64]()),
            List[Dim](),
        )

    # TODO(GEX-578): Once reshape with Variant[Symbol, Int] is removed, we can also remove this.
    # Will only need Variant[Dim, Int].
    @always_inline
    fn reshape(self, *dims: Int) raises -> Symbol:
        """Reshapes this `Symbol`.

        Uses the `mo.reshape` op. Requires the symbol to be a `TensorType`.

        Args:
            dims: The new dimensions.

        Returns:
            A new `Symbol` that has the given shape.
        """
        return self._reshape_impl(dims, __call_location())

    # TODO(GEX-578): Once reshape with Variant[Symbol, Int] is removed, we can also remove this.
    # Will only need Variant[Dim, Int].
    fn _reshape_impl(
        self, dims: VariadicList[Int], call_loc: _SourceLocation
    ) raises -> Symbol:
        if len(dims) == 0:
            return self.reshape()

        var shape = List[Dim]()
        for dim in dims:
            shape.append(Dim(dim))

        try:
            return ops.reshape(self, shape)
        except e:
            raise error(self.graph(), e, location=call_loc)

    @always_inline
    fn reshape(self, *dims: Dim) raises -> Symbol:
        """Reshapes this `Symbol`.

        Uses the `mo.reshape` op. Requires the symbol to be a `TensorType`.

        Args:
            dims: The new dimensions.

        Returns:
            A new `Symbol` that has the given shape.
        """
        return self._reshape_impl(dims, __call_location())

    fn _reshape_impl(
        self, dims: VariadicListMem[Dim, *_], call_loc: _SourceLocation
    ) raises -> Symbol:
        if len(dims) == 0:
            return self.reshape()

        var shape = List[Dim]()
        for dim in dims:
            shape.append(dim[])

        try:
            return ops.reshape(self, shape)
        except e:
            raise error(self.graph(), e, location=call_loc)

    fn reshape(self, *dims: Variant[Symbol, Int]) raises -> Symbol:
        """Reshapes this `Symbol`.

        Uses the `mo.reshape` op. Requires the symbol to be a `TensorType`.

        Args:
            dims: The new dimensions.

        Returns:
            A new `Symbol` that has the given shape.
        """
        if len(dims) == 0:
            return self.reshape()

        var static_shape = List[Dim]()
        var symbolic_dims = List[Symbol]()

        for dim in dims:
            if dim[].isa[Symbol]():
                static_shape.append(Dim.dynamic())
                symbolic_dims.append(dim[][Symbol])
            else:
                var d = dim[][Int]
                static_shape.append(d if d >= 0 else Dim.dynamic())
                symbolic_dims.append(self.graph().scalar[DType.int64](d))

        return ops.reshape(self, ops.stack(symbolic_dims), static_shape)

    fn swapaxes(self, axis1: Int, axis2: Int) raises -> Symbol:
        """Interchanges two axes of this `Symbol`.

        Uses the `mo.transpose` op. Negative values are allowed, and represent
        the axis number counting from the last.

        Args:
            axis1: One of the axes to swap.
            axis2: The other axis.

        Returns:
            A new transposed `Symbol`.
        """
        return transpose(self, axis1, axis2)

    @always_inline
    fn broadcast_to(self, *dims: Dim) raises -> Symbol:
        """Broadcasts this `Symbol` to the specified dims.

        Uses the `mo.broadcast_to` op. Requires the symbol to be a `TensorType`.

        Args:
            dims: The target dimensions.

        Returns:
            A new `Symbol` that is broadcast to the given shape.
        """
        var shape = List[Dim]()
        for dim in dims:
            shape.append(dim[])

        return self.broadcast_to(shape, __call_location())

    @always_inline
    fn broadcast_to(
        self, shape: List[Dim], location: Optional[_SourceLocation] = None
    ) raises -> Symbol:
        """Broadcasts this `Symbol` to the specified shape.

        Uses the `mo.broadcast_to` op. Requires the symbol to be a `TensorType`.

        Args:
            shape: The target shape.
            location: An optional location for a more specific error message.

        Returns:
            A new `Symbol` that is broadcast to the given shape.
        """

        return ops.broadcast_to(self, shape, location or __call_location())

    # ===------------------------------------------------------------------=== #
    # Slicing operators
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __getitem__(
        self, i: Variant[Symbol, Int], axis: Int = 0, keep_dims: Bool = False
    ) raises -> Symbol:
        """Symbolic slicing - indexes a value by a single index.

        Uses the `mo.slice` op.

        Args:
            i: The index value.
            axis: The axis to index at.
            keep_dims: Returns a tensor with the same rank as the input if set.

        Returns:
            The slicing result.
        """
        return self._getitem_impl(i, axis, keep_dims, __call_location())

    fn _getitem_impl(
        self,
        i: Variant[Symbol, Int],
        axis: Int,
        keep_dims: Bool,
        call_loc: _SourceLocation,
    ) raises -> Symbol:
        var index_sym: Symbol
        if i.isa[Int]():
            index_sym = self.graph().scalar(Int64(i[Int]))
        else:
            if i[Symbol].tensor_type().rank() != 0:
                raise error(
                    self.graph(),
                    "Slicing a tensor by Symbol requires a rank 0 index",
                    location=call_loc,
                )

            index_sym = i[Symbol]

        return ops.slice(self, index_sym, axis, keep_dims)

    @always_inline
    fn __getitem__(
        self, *s: SymbolicSlice, out_dims: List[Dim]
    ) raises -> Symbol:
        """Range-based slicing.

        Uses the `mo.slice` op. Slicing along multiple dimensions is
        supported.

        Args:
            s: The slice values. The `i`th `SymbolicSlice` in the variadic list
                represents the begin:end:step triple for axis `i`.
            out_dims: The expected output dimensions returned by slicing.
              These will be assert at graph execution time to be correct.

        Returns:
            The slicing result.
        """
        return self._getitem_impl(s, out_dims, __call_location())

    fn _getitem_impl(
        self,
        s: VariadicListMem[SymbolicSlice, *_],
        out_dims: List[Dim],
        call_loc: _SourceLocation,
    ) raises -> Symbol:
        var slices = List[SymbolicSlice]()
        for sval in s:
            slices.append(sval[])
        return ops.slice(self, slices, out_dims, call_loc)

    @always_inline
    fn __getitem__(
        self, *slices: Slice, out_dims: List[Dim] = List[Dim]()
    ) raises -> Symbol:
        """Shorthand for symbolic slicing with `Int` ranges.

        Args:
            slices: The slice values for each dimension respectively.
              If fewer than `rank()` slices are provided, the remaining
              dimensions will be trivially sliced. In other words
              `s[:, :2]` is equivalent to `s[:, :2, :, :]` for a tensor of
              rank 4. Currently indexing and slicing may not be mixed.
            out_dims: The expected output dimensions returned by slicing.
              These will be assert at graph execution time to be correct.

        Returns:
            The slicing result.

        Raises:
            An exception if out_dims is empty and can't be calculated at graph build time.
        """
        return ops.slice(self, slices, out_dims, __call_location())

    # ===------------------------------------------------------------------=== #
    # Arithmetic operators
    # ===------------------------------------------------------------------=== #

    # Note: Keep in alphabetic order.

    fn _consistent_scalar(self, value: Int) raises -> Symbol:
        return self.graph().scalar(value, self.type().tensor().dtype)

    fn _consistent_scalar(self, value: Float64) raises -> Symbol:
        return self.graph().scalar(value, self.type().tensor().dtype)

    @always_inline
    fn __add__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise addition.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self, rhs, __call_location())

    @always_inline
    fn __add__(self, rhs: Int) raises -> Symbol:
        """Element-wise addition by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __add__(self, rhs: Float64) raises -> Symbol:
        """Element-wise addition by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __matmul__(self, rhs: Symbol) raises -> Symbol:
        """Matrix multiplication.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return matmul(self, rhs, __call_location())

    @always_inline
    fn __mul__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise multiplication.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self, rhs, __call_location())

    @always_inline
    fn __mul__(self, rhs: Int) raises -> Symbol:
        """Element-wise multiplication by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __mul__(self, rhs: Float64) raises -> Symbol:
        """Element-wise multiplication by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self, self._consistent_scalar(rhs), __call_location())

    fn __neg__(self) raises -> Symbol:
        """Numerical negative, element-wise.

        Returns:
            The operation result.
        """
        return self.graph().op("rmo.mo.negative", self, self.tensor_type())

    @always_inline
    fn __pow__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise raise to power.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self, rhs, __call_location())

    @always_inline
    fn __pow__(self, rhs: Int) raises -> Symbol:
        """Element-wise raise to power by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __pow__(self, rhs: Float64) raises -> Symbol:
        """Element-wise raise to power by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __radd__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise addition.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(rhs, self, __call_location())

    @always_inline
    fn __radd__(self, rhs: Int) raises -> Symbol:
        """Element-wise addition by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __radd__(self, rhs: Float64) raises -> Symbol:
        """Element-wise addition by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rmul__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise multiplication.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(rhs, self, __call_location())

    @always_inline
    fn __rmul__(self, rhs: Int) raises -> Symbol:
        """Element-wise multiplication by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rmul__(self, rhs: Float64) raises -> Symbol:
        """Element-wise multiplication by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rpow__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise raise to power.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(rhs, self, __call_location())

    @always_inline
    fn __rpow__(self, rhs: Int) raises -> Symbol:
        """Element-wise raise to power by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rpow__(self, rhs: Float64) raises -> Symbol:
        """Element-wise raise to power by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rsub__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise subtraction.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(rhs, self, __call_location())

    @always_inline
    fn __rsub__(self, rhs: Int) raises -> Symbol:
        """Element-wise subtraction by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rsub__(self, rhs: Float64) raises -> Symbol:
        """Element-wise subtraction by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rtruediv__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise division.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(rhs, self, __call_location())

    @always_inline
    fn __rtruediv__(self, rhs: Int) raises -> Symbol:
        """Element-wise division by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __rtruediv__(self, rhs: Float64) raises -> Symbol:
        """Element-wise division by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self._consistent_scalar(rhs), self, __call_location())

    @always_inline
    fn __sub__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise subtraction.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self, rhs, __call_location())

    @always_inline
    fn __sub__(self, rhs: Int) raises -> Symbol:
        """Element-wise subtraction by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __sub__(self, rhs: Float64) raises -> Symbol:
        """Element-wise subtraction by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __truediv__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise division.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self, rhs, __call_location())

    @always_inline
    fn __truediv__(self, rhs: Int) raises -> Symbol:
        """Element-wise division by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self, self._consistent_scalar(rhs), __call_location())

    @always_inline
    fn __truediv__(self, rhs: Float64) raises -> Symbol:
        """Element-wise division by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self, self._consistent_scalar(rhs), __call_location())

    # ===------------------------------------------------------------------=== #
    # Other ops
    # ===------------------------------------------------------------------=== #

    fn print(self, label: String = "debug_tensor") raises:
        """Prints this `Symbol`'s value at runtime.

        This uses `mo.debug.tensor.unsafe.print` to enable printing the runtime value
        that this `Symbol` represents, at grpah execution time.

        Args:
            label: A label to accompany the printout.
        """
        var g = self.graph()
        var layer = g.current_layer()
        var label_prefix = (layer + ": ") if layer else ""
        # TODO(MSDK-1160): As with the Python implementation of debug printing
        # this should instead use the standard mo.debug.tensor.print which has
        # a chain as input and output enabling proper sequencing of prints.
        _ = g.nvop(
            "mo.debug.tensor.unsafe.print",
            List(self),
            List[Type](),
            List(_string_attr(g._context(), "label", label_prefix + label)),
        )

    # ===------------------------------------------------------------------=== #
    # Graph manipulation
    # ===------------------------------------------------------------------=== #

    fn insert_transformation(
        self, transform: fn (Symbol) raises -> Symbol
    ) raises:
        """Inserts nodes in between this `Symbol` and all its current uses.

        This enables inserting ops in between this `Symbol` and all its uses,
        for example to modify an existing `Graph`.

        Note: The function is called exactly once, even if `self` has no uses.

        Args:
            transform: A function that creates a unary subgraph (single input,
                single output) and returns the result of the final node. The
                function will be called with this `Symbol`, and its return value
                will replace all of its uses.
        """
        var dummy = self.graph().constant[DType.float32](0)
        self.handle.replace_all_uses_with(dummy.handle)
        var replacement = transform(self)
        dummy.handle.replace_all_uses_with(replacement.handle)


@value
struct SymbolicSlice(Copyable, Movable):
    """`Slice`-like struct with `Symbol` fields.

    This struct enables the range slice (`start:stop:end`) operator with
    `Symbol` indices.
    """

    var start: Optional[Symbol]
    """The slice's start index."""

    var stop: Optional[Symbol]
    """The slice's start index, exclusive."""

    var step: Optional[Symbol]
    """The slice's step."""

    def __init__(out self, start: Optional[Symbol], stop: Optional[Symbol]):
        """Convenience constructor from `start` and `step` that doesn't require `step`.
        """
        self.start = start
        self.stop = stop
        self.step = Optional[Symbol]()

    def __init__(out self, g: Graph, s: Slice):
        """Convenience constructor from a `Slice`.

        This wraps any indices in `s` into constant nodes (using `mo.constant`).

        Args:
            g: The `Graph` into which the constant nodes are created.
            s: The `Slice` to turn into `SymbolicSlice`.
        """
        self.start = Optional[Symbol]()
        self.stop = Optional[Symbol]()
        self.step = Optional[Symbol]()
        if s.start:
            self.start = g.scalar(Int64(s.start.value()))
        if s.end:
            self.stop = g.scalar(Int64(s.end.value()))
        if s.step:
            self.step = g.scalar(Int64(s.step.value()))
