# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Symbolic value primitives.

A `Symbol` can represent the output of a `Node`, the arguments of a `Graph` (as
seen from within its body), and more generally any symbolic value available
within the `Graph`. Other `Node`s receive `Symbol`s as inputs to form a
computation graph.

`Symbol`s may also be used to refer to an existing input or output of a `Node`,
and one may change these, for instance by swapping a new `Symbol`.

Conceptually, `Symbol`s are the equivalent of variables in Mojo. A `Symbol` can
also be thought of as the end of an edge in the dataflow graph (the other end
being one use of that symbol).

Similar to regular variables, `Symbol`s have a data type.

Note: All the helpers in this module are documented as "Creates foo". This is
a shorthand notation for "Adds a node representing an op that returns foo".
"""

from collections.optional import Optional
from memory.unsafe import _LITRef
from tensor import Tensor, TensorShape, TensorSpec
from utils.variant import Variant

import _mlir

from .attr import AttrMap
from .graph import Graph
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
struct Symbol(CollectionElement, Stringable):
    """Represents a symbolic value within a `Graph`."""

    var handle: _mlir.Value
    """An handle to this `Symbol`'s internal representation."""

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn graph(self) -> Graph:
        """Returns the `Graph` owning this `Symbol`.

        Returns:
            The parent `Graph`.
        """
        var parent = self.handle.parent()
        var block: _mlir.Block
        if parent.isa[_mlir.Block]():
            block = parent.get[_mlir.Block]()[]
        else:
            var op = parent.get[_mlir.Operation]()[]
            block = op.block()

        var graph_op = block.parent()
        return Graph(graph_op)

    fn type(self) raises -> AnyMOType:
        """Returns this `Symbol`'s type.

        Returns:
            The `Symbol`'s type.
        """
        return AnyMOType.from_mlir(self.handle.type())

    fn tensor_type(self) raises -> MOTensor:
        """Returns this `Symbol`'s type, as `MOTensor`.

        Implicitly asserts that the type is indeed `MOTensor`, and raises an
        error otherwise.

        Returns:
            The tensor type of this `Symbol`.
        """
        return self.type().tensor()

    fn __str__(self) -> String:
        """Returns a `String` representation of this `Symbol`.

        The representation uses an internal MLIR Assembly format, and typically
        shows the `Node` that outputs this `Symbol`. Its structure
        is subject to change without notice, and should not be used for
        serialization. For debugging purposes only.

        Returns:
            A textual representation of this `Symbol`.
        """
        return str(self.handle)

    # ===------------------------------------------------------------------=== #
    # Casting and reshaping operators.
    # ===------------------------------------------------------------------=== #

    fn reshape(self, *dims: Variant[Symbol, Int]) raises -> Symbol:
        """Reshapes this `Symbol`.

        Uses the `mo.reshape` op. Requires the symbol to be a `MOTensor`.

        Args:
            dims: The new dimensions.

        Returns:
            A new `Symbol` that has the given shape.
        """
        var static_shape = List[Dim]()
        var symbolic_dims = List[Symbol]()

        if len(dims) == 0:
            return ops.reshape(
                self,
                self.graph().vector[DType.int64](List[Int64]()),
                static_shape,
            )

        for dim in dims:
            if dim[].isa[Symbol]():
                static_shape.append(Dim.dynamic())
                symbolic_dims.append(dim[].get[Symbol]()[])
            else:
                var d = dim[].get[Int]()[]
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

    # ===------------------------------------------------------------------=== #
    # Slicing operators
    # ===------------------------------------------------------------------=== #

    fn __getitem__(self, i: Symbol, axis: Int = 0) raises -> Symbol:
        """Symbolic slicing - indexes a value by a single index.

        Uses the `mo.slice` op.

        Args:
            i: The index value.
            axis: The axis to index at.

        Returns:
            The slicing result.
        """
        return ops.slice(self, i, axis=axis)

    fn __getitem__(self, i: Int, axis: Int = 0) raises -> Symbol:
        """Symbolic slicing - indexes a value by a single constant index.

        Uses the `mo.slice` op and automatically wraps `i` inside a
        `mo.constant`.

        Args:
            i: The index value.
            axis: The axis to index at.

        Returns:
            The slicing result.
        """
        return ops.slice(self, self.graph().scalar(Int64(i)), axis=axis)

    fn __getitem__(self, *s: SymbolicSlice) raises -> Symbol:
        """Range-based slicing.

        Uses the `mo.slice` op. Slicing along multiple dimensions is
        supported.

        Args:
            s: The slice values. The `i`th `SymbolicSlice` in the variadic list
                represents the begin:end:step triple for axis `i`.

        Returns:
            The slicing result.
        """
        var slices = List[SymbolicSlice]()
        for sval in s:
            slices.append(sval[])
        return ops.slice(self, slices)

    fn __getitem__(self, s: Slice) raises -> Symbol:
        """Shorthand for symbolic slicing with an `Int` range.

        This overload only supports slicing along the first axis.

        Args:
            s: The slice value.

        Returns:
            The slicing result.
        """
        return ops.slice(self, s)

    # ===------------------------------------------------------------------=== #
    # Arithmetic operators
    # ===------------------------------------------------------------------=== #

    # Note: Keep in alphabetic order.

    fn _consistent_scalar(self, value: Int) raises -> Symbol:
        return self.graph().scalar(value, self.type().tensor().dtype)

    fn _consistent_scalar(self, value: Float64) raises -> Symbol:
        return self.graph().scalar(value, self.type().tensor().dtype)

    fn __add__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise addition.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self, rhs)

    fn __add__(self, rhs: Int) raises -> Symbol:
        """Element-wise addition by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self, self._consistent_scalar(rhs))

    fn __add__(self, rhs: Float64) raises -> Symbol:
        """Element-wise addition by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self, self._consistent_scalar(rhs))

    fn __matmul__(self, rhs: Symbol) raises -> Symbol:
        """Matrix multiplication.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return matmul(self, rhs)

    fn __mul__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise multiplication.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self, rhs)

    fn __mul__(self, rhs: Int) raises -> Symbol:
        """Element-wise multiplication by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self, self._consistent_scalar(rhs))

    fn __mul__(self, rhs: Float64) raises -> Symbol:
        """Element-wise multiplication by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self, self._consistent_scalar(rhs))

    fn __neg__(self) raises -> Symbol:
        """Numerical negative, element-wise.

        Returns:
            The operation result.
        """
        return self.graph().op("mo.negative", self, self.tensor_type())

    fn __pow__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise raise to power.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self, rhs)

    fn __pow__(self, rhs: Int) raises -> Symbol:
        """Element-wise raise to power by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self, self._consistent_scalar(rhs))

    fn __pow__(self, rhs: Float64) raises -> Symbol:
        """Element-wise raise to power by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self, self._consistent_scalar(rhs))

    fn __radd__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise addition.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(rhs, self)

    fn __radd__(self, rhs: Int) raises -> Symbol:
        """Element-wise addition by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self._consistent_scalar(rhs), self)

    fn __radd__(self, rhs: Float64) raises -> Symbol:
        """Element-wise addition by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return add(self._consistent_scalar(rhs), self)

    fn __rmul__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise multiplication.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(rhs, self)

    fn __rmul__(self, rhs: Int) raises -> Symbol:
        """Element-wise multiplication by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self._consistent_scalar(rhs), self)

    fn __rmul__(self, rhs: Float64) raises -> Symbol:
        """Element-wise multiplication by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return mul(self._consistent_scalar(rhs), self)

    fn __rpow__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise raise to power.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(rhs, self)

    fn __rpow__(self, rhs: Int) raises -> Symbol:
        """Element-wise raise to power by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self._consistent_scalar(rhs), self)

    fn __rpow__(self, rhs: Float64) raises -> Symbol:
        """Element-wise raise to power by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return pow(self._consistent_scalar(rhs), self)

    fn __rsub__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise subtraction.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(rhs, self)

    fn __rsub__(self, rhs: Int) raises -> Symbol:
        """Element-wise subtraction by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self._consistent_scalar(rhs), self)

    fn __rsub__(self, rhs: Float64) raises -> Symbol:
        """Element-wise subtraction by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self._consistent_scalar(rhs), self)

    fn __rtruediv__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise division.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(rhs, self)

    fn __rtruediv__(self, rhs: Int) raises -> Symbol:
        """Element-wise division by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self._consistent_scalar(rhs), self)

    fn __rtruediv__(self, rhs: Float64) raises -> Symbol:
        """Element-wise division by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self._consistent_scalar(rhs), self)

    fn __sub__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise subtraction.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self, rhs)

    fn __sub__(self, rhs: Int) raises -> Symbol:
        """Element-wise subtraction by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self, self._consistent_scalar(rhs))

    fn __sub__(self, rhs: Float64) raises -> Symbol:
        """Element-wise subtraction by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return sub(self, self._consistent_scalar(rhs))

    fn __truediv__(self, rhs: Symbol) raises -> Symbol:
        """Element-wise division.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self, rhs)

    fn __truediv__(self, rhs: Int) raises -> Symbol:
        """Element-wise division by an `Int` literal.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self, self._consistent_scalar(rhs))

    fn __truediv__(self, rhs: Float64) raises -> Symbol:
        """Element-wise division by a `Float64`.

        Args:
            rhs: The right hand side operand.

        Returns:
            The operation result.
        """
        return div(self, self._consistent_scalar(rhs))

    # ===------------------------------------------------------------------=== #
    # Other ops
    # ===------------------------------------------------------------------=== #

    fn print(self, label: String = "debug_tensor") raises:
        """Prints this `Symbol`'s value at runtime.

        This uses `mo.debug.tensor.print` to enable printing the runtime value
        that this `Symbol` represents, at grpah execution time.

        Args:
            label: A label to accompany the printout.
        """
        var g = self.graph()
        var attrs = AttrMap(g._module().string_attr("label", label))
        _ = g.nvop("mo.debug.tensor.print", self, TypeTuple(), attrs)

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
struct SymbolicSlice(CollectionElement):
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

    def __init__(inout self, g: Graph, s: Slice):
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
            self.start = g.scalar(Int64(s.start))
        if s.end:
            self.stop = g.scalar(Int64(s.end))
        if s.step:
            self.step = g.scalar(Int64(s.step))


@value
struct SymbolTuple(Sized):
    """A tuple of `Symbol`s.

    This struct mainly offers the convenience of building a tuple of `Symbol`s
    using the tuple literal (`(a, b, c)`) notation. It is largely equivalent
    to a `List[Symbol]`.
    """

    # TODO: Drop this once List can do this.

    var symbols: List[Symbol]
    """The actual list of symbols."""

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *symbols: Symbol):
        """Constructor from a variadic list of `Symbol`s.

        Args:
            symbols: `Symbol`s to initializae the tuple with.
        """
        self.symbols = List[Symbol]()
        for symbol in symbols:
            self.symbols.append(symbol[])

    fn __init__(inout self, owned symbols: ()):
        """Convenience constructor for an empty tuple.

        Args:
            symbols: The empty tuple.
        """
        self.__init__()

    fn __init__(inout self, owned symbols: (Symbol, Symbol)):
        """Convenience constructor from a 2-tuple.

        Args:
            symbols: A tuple of two `Symbol`s.
        """
        var ptr = Pointer.address_of(symbols).bitcast[Int8]()
        self.__init__(
            ptr.bitcast[Symbol]()[],
            ptr.offset(symbols._offset[1]()).bitcast[Symbol]()[],
        )

    fn __init__(inout self, owned symbols: (Symbol, Symbol, Symbol)):
        """Convenience constructor from a 3-tuple.

        Args:
            symbols: A tuple of three `Symbol`s.
        """
        var ptr = Pointer.address_of(symbols).bitcast[Int8]()
        self.__init__(
            ptr.bitcast[Symbol]()[],
            ptr.offset(symbols._offset[1]()).bitcast[Symbol]()[],
            ptr.offset(symbols._offset[2]()).bitcast[Symbol]()[],
        )

    fn __init__(inout self, owned symbols: (Symbol, Symbol, Symbol, Symbol)):
        """Convenience constructor from a 4-tuple.

        Args:
            symbols: A tuple of four `Symbol`s.
        """
        var ptr = Pointer.address_of(symbols).bitcast[Int8]()
        self.__init__(
            ptr.bitcast[Symbol]()[],
            ptr.offset(symbols._offset[1]()).bitcast[Symbol]()[],
            ptr.offset(symbols._offset[2]()).bitcast[Symbol]()[],
            ptr.offset(symbols._offset[3]()).bitcast[Symbol]()[],
        )

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #
    fn __len__(self) -> Int:
        """Returns the length of this `SymbolTuple`."""
        return len(self.symbols)

    fn __getitem__(self, i: Int) -> Symbol:
        """Returns the `Symbol` at the specified index.

        Args:
            i: The index to retrieve from.

        Returns:
            The `Symbol` at position `i`.
        """
        return self.symbols[i]
